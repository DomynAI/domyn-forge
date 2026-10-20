# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Parity between a torchtitan DomynEdge model and its HuggingFace conversion.

A randomly initialized debug model is saved as a DCP checkpoint, converted with
``scripts/huggingface/convert.py`` exactly as a trained checkpoint would be, loaded back
through ``AutoModelForCausalLM`` with the bundled remote code, and both models must
produce the same logits on the same tokens.

Each model runs the way it is used: torchtitan with float32 weights under bfloat16 autocast,
as in training, and the HF model loaded in bfloat16 without autocast, as ``generate.py`` and
the benchmarks run it.

Both models use varlen (flash) attention, so the comparison needs a CUDA device.
"""

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
    """Save ``model`` as a DCP checkpoint and convert it with ``convert.py``."""
    convert = load_convert_module()
    console = rich.console.Console(quiet=True)

    dcp_dir = workdir / "dcp"
    hf_dir = workdir / "hf"
    hf_dir.mkdir()
    torch.distributed.checkpoint.state_dict_saver.save(model.state_dict(), checkpoint_id=dcp_dir)

    convert.save_model(console, TOKENIZER_DIR, dcp_dir, hf_dir, "bfloat16", "domynedge", FLAVOR, VOCAB_SIZE)
    convert.save_tokenizer(console, TOKENIZER_DIR, hf_dir)
    convert.save_modeling(console, hf_dir, assets_dir=HF_ASSETS_DIR)
    convert.save_config(console, TOKENIZER_DIR, hf_dir, "domynedge", FLAVOR, "bfloat16", MAX_SEQ_LEN, VOCAB_SIZE)
    return hf_dir


def load_hf_model(hf_dir: pathlib.Path) -> torch.nn.Module:
    """Load the converted checkpoint as ``generate.py`` does, failing on any weight it does not map one to one."""
    # The stubs do not model the (model, loading_info) tuple returned with `output_loading_info=True`.
    model, loading_info = typing.cast(
        tuple[torch.nn.Module, dict[str, typing.Any]],
        transformers.AutoModelForCausalLM.from_pretrained(
            hf_dir, trust_remote_code=True, dtype=torch.bfloat16, output_loading_info=True
        ),
    )
    for kind in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        assert not loading_info[kind], f"{kind} when loading the converted checkpoint: {loading_info[kind]}"
    return model.eval()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen attention requires a CUDA device")
@torch.no_grad()
def test_debugmodel_hf_logits_match(tmp_path):
    device = torch.device("cuda")
    titan_model = build_titan_model()
    hf_model = load_hf_model(convert_to_hf(titan_model, tmp_path)).to(device)
    titan_model = titan_model.to(device)

    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQ_LEN), generator=generator).to(device)
    positions = torch.arange(SEQ_LEN, device=device).expand(BATCH_SIZE, -1)

    with torch.autocast("cuda", dtype=torch.bfloat16):
        titan_logits = titan_model(tokens, positions, titan_model.get_attention_masks(positions)).float()
    hf_logits = hf_model(input_ids=tokens, position_ids=positions).logits.float()

    assert titan_logits.shape == hf_logits.shape == (BATCH_SIZE, SEQ_LEN, VOCAB_SIZE)
    # The HF residual stream is bfloat16 while torchtitan's is float32, so logits drift by up to
    # ~0.12 (~2% relative); so the checks are relative rather than element-wise.
    relative_error = (hf_logits - titan_logits).norm() / titan_logits.norm()
    assert relative_error < 5e-2, f"relative L2 error of the logits is {relative_error:.4f}"

    # A random model has many near-ties at the top, which that drift can flip; compare the argmax
    # only where titan's top-1 logit leads the runner-up by about twice the largest drift.
    top2 = titan_logits.topk(2, dim=-1).values
    decisive = (top2[..., 0] - top2[..., 1]) > 0.25
    assert decisive.float().mean() > 0.3, "too few decisive positions for the argmax check to mean anything"
    assert torch.equal(hf_logits.argmax(-1)[decisive], titan_logits.argmax(-1)[decisive])
