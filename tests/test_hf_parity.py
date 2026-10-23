# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Parity between a torchtitan DomynEdge model and its HuggingFace conversion."""

import importlib.util
import pathlib
import typing

import pytest
import rich.console
import torch
import torch.distributed.checkpoint.state_dict_saver
import transformers

from domyn_forge.models.domynedge import DomynEdgeModel, model_registry

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HF_ASSETS_DIR = REPO_ROOT / "assets" / "domyn-edge" / "hf"
TOKENIZER_DIR = REPO_ROOT / "assets" / "domyn-edge" / "tokenizers" / "pretraining"

FLAVOR = "debugmodel"
VOCAB_SIZE = 2048
MAX_SEQ_LEN = 2048
# Longer than the debug model's 128-token sliding window, so windowed layers see truncated context.
SEQ_LEN = 384
BATCH_SIZE = 2


def load_convert_module():
    spec = importlib.util.spec_from_file_location("convert", REPO_ROOT / "scripts" / "huggingface" / "convert.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_titan_model(seed: int = 0) -> DomynEdgeModel:
    """Build the debug model with random weights, norm weights included.

    Norm weights are initialized to one, which would hide a norm whose weight is dropped
    by the conversion, so they are moved away from one.
    """
    torch.manual_seed(seed)
    with torch.device("cpu"):
        model = model_registry(FLAVOR, max_seq_len=MAX_SEQ_LEN).model.build()
    assert isinstance(model, DomynEdgeModel)
    model.init_states()
    with torch.no_grad():
        for name, param in model.named_parameters():
            if "norm" in name:
                param.copy_(1.0 + 0.1 * torch.randn_like(param))
    return model.eval()


def convert_to_hf(model: torch.nn.Module, workdir: pathlib.Path) -> pathlib.Path:
    """Save ``model`` as a DCP checkpoint and convert it with ``convert.py`` in the model's own dtype."""
    convert = load_convert_module()
    console = rich.console.Console(quiet=True)
    dtype = str(next(model.parameters()).dtype).removeprefix("torch.")

    dcp_dir = workdir / "dcp"
    hf_dir = workdir / "hf"
    hf_dir.mkdir()
    torch.distributed.checkpoint.state_dict_saver.save(model.state_dict(), checkpoint_id=dcp_dir)

    convert.save_model(console, TOKENIZER_DIR, dcp_dir, hf_dir, dtype, "domynedge", FLAVOR, VOCAB_SIZE)
    convert.save_tokenizer(console, TOKENIZER_DIR, hf_dir)
    convert.save_modeling(console, hf_dir, assets_dir=HF_ASSETS_DIR)
    convert.save_config(console, TOKENIZER_DIR, hf_dir, "domynedge", FLAVOR, dtype, MAX_SEQ_LEN, VOCAB_SIZE)
    return hf_dir


def load_hf_model(hf_dir: pathlib.Path, dtype: torch.dtype) -> torch.nn.Module:
    """Load the converted checkpoint as ``generate.py`` does, failing on any weight it does not map one to one."""
    # The stubs do not model the (model, loading_info) tuple returned with `output_loading_info=True`.
    model, loading_info = typing.cast(
        tuple[torch.nn.Module, dict[str, typing.Any]],
        transformers.AutoModelForCausalLM.from_pretrained(
            hf_dir, trust_remote_code=True, dtype=dtype, output_loading_info=True
        ),
    )
    for kind in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        assert not loading_info[kind], f"{kind} when loading the converted checkpoint: {loading_info[kind]}"
    return model.eval()


def test_debugmodel_hf_config(tmp_path):
    """The converted config exposes per-layer head sizes and the RoPE settings in their transformers 5 form."""
    # The registry types `.model` as the base `Config`, which declares no `layers`.
    layers = typing.cast(DomynEdgeModel.Config, model_registry(FLAVOR, max_seq_len=MAX_SEQ_LEN).model).layers
    config = transformers.AutoConfig.from_pretrained(
        convert_to_hf(build_titan_model(), tmp_path), trust_remote_code=True
    )

    head_dims = [layer.attention.head_dim for layer in layers]
    assert config.head_sizes == head_dims
    assert [layer_config.head_dim for layer_config in config.per_layer_config] == head_dims
    # Head sizes live only in `per_layer_config`, with no global `head_dim`.
    assert "head_dim" not in config.to_dict()
    # The HF config has a single `rope_parameters`, so every RoPE layer must share one theta.
    rope_thetas = {layer.attention.rope.theta for layer in layers if layer.attention.use_rope}
    assert len(rope_thetas) == 1, f"RoPE layers must share one theta. Got {rope_thetas}"
    assert config.rope_parameters == {"rope_type": "default", "rope_theta": rope_thetas.pop()}
    assert not hasattr(config, "rope_theta")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen attention requires a CUDA device")
@torch.no_grad()
def test_debugmodel_hf_logits_match(tmp_path):
    device = torch.device("cuda")
    titan_model = build_titan_model()
    dtype = next(titan_model.parameters()).dtype
    hf_model = load_hf_model(convert_to_hf(titan_model, tmp_path), dtype).to(device)
    titan_model = titan_model.to(device)

    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=generator).to(device)
    positions = torch.arange(SEQ_LEN, device=device).expand(BATCH_SIZE, -1)

    titan_logits = titan_model(tokens, positions, titan_model.get_attention_masks(positions))
    hf_logits = hf_model(input_ids=tokens, position_ids=positions).logits

    assert titan_logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    torch.testing.assert_close(hf_logits, titan_logits, rtol=1e-3, atol=1e-2)
