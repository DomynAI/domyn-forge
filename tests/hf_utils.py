# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Helpers shared by the HF tests: build and convert the torchtitan debug model, load and compare HF models."""

import importlib.util
import pathlib
import typing

import pytest
import torch
import torch.distributed.checkpoint.state_dict_saver
import transformers

from domyn_forge.models.domynedge import DomynEdgeModel, model_registry

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
HF_ASSETS_DIR = REPO_ROOT / "assets" / "domyn-edge" / "hf"
TOKENIZER_DIR = REPO_ROOT / "assets" / "domyn-edge" / "tokenizers" / "pretraining"


ATTN_IMPLEMENTATIONS = [
    pytest.param("eager", id="eager"),
    pytest.param("sdpa", id="sdpa"),
    pytest.param(
        "flex_attention",
        id="flex_attention",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="flex attention is only tested on CUDA"),
    ),
    pytest.param(
        "torch_varlen",
        id="torch_varlen",
        # The modeling's backend for PyTorch's built-in flash kernel, the one torchtitan's varlen attention runs.
        # This should be byte identical.
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen attention requires a CUDA device"),
    ),
]


def load_convert_module():
    spec = importlib.util.spec_from_file_location("convert", REPO_ROOT / "scripts" / "huggingface" / "convert.py")
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_titan_model(
    dtype: torch.dtype = torch.bfloat16,
    seed: int = 0,
    flavor: str = "debugmodel",
    max_seq_len: int = 2048,
    device: str = "cpu",
) -> DomynEdgeModel:
    """Build the debug model on ``device`` with random ``dtype`` weights, norm weights included.

    The default bfloat16 is the dtype the model is shipped in.

    The model is built on the meta device and materialized on ``device``, as the trainer does, so the RoPE cache is
    computed there: on the GPU, as in training and as the HF model computes it (the CPU one differs in the last float32
    bit, which flips the bfloat16 rounding of a few rotated queries and keys).
    """
    torch.manual_seed(seed)
    with torch.device("meta"):
        model = model_registry(flavor, max_seq_len=max_seq_len).model.build()
    assert isinstance(model, DomynEdgeModel)
    model.to_empty(device=device)
    model.init_states()
    with torch.no_grad():
        for name, tensor in [*model.named_parameters(), *model.named_buffers()]:
            if name.endswith(".attention.rope.cache"):
                # The RoPE caches stay in float32, as in training.
                assert tensor.dtype == torch.float32, f"RoPE cache {name} is {tensor.dtype}, not float32"
                continue
            if "norm" in name:
                # Norm weights are initialized to one, which would hide a norm whose weight is dropped by the conversion
                tensor.copy_(1.0 + 0.1 * torch.randn_like(tensor))
            tensor.data = tensor.data.to(dtype)
    return model.eval()


def convert_to_hf(
    model: torch.nn.Module,
    workdir: pathlib.Path,
    flavor: str = "debugmodel",
    vocab_size: int = 2048,
    max_seq_len: int = 2048,
) -> pathlib.Path:
    """Save ``model`` as a DCP checkpoint and convert it with the ``convert.py`` CLI in the model's own dtype."""
    dcp_dir = workdir / "dcp"
    hf_dir = workdir / "hf"
    torch.distributed.checkpoint.state_dict_saver.save(model.state_dict(), checkpoint_id=dcp_dir)

    # `standalone_mode=False` returns instead of calling `sys.exit` when the command completes.
    load_convert_module().main(
        [
            f"--model-dir={dcp_dir}",
            f"--tokenizer-dir={TOKENIZER_DIR}",
            f"--output-dir={hf_dir}",
            f"--assets-dir={HF_ASSETS_DIR}",
            f"--dtype={str(next(model.parameters()).dtype).removeprefix('torch.')}",
            "--model-name=domynedge",
            f"--model-flavor={flavor}",
            f"--max-seq-len={max_seq_len}",
            f"--vocab-size={vocab_size}",
        ],
        standalone_mode=False,
    )
    return hf_dir


def load_hf_model(hf_dir: pathlib.Path, attn_implementation: str | None = None) -> torch.nn.Module:
    """Load the converted checkpoint as the model card does, failing on any weight it does not map one to one."""
    # The stubs do not model the (model, loading_info) tuple returned with `output_loading_info=True`.
    model, loading_info = typing.cast(
        tuple[torch.nn.Module, dict[str, typing.Any]],
        transformers.AutoModelForCausalLM.from_pretrained(
            hf_dir,
            dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation=attn_implementation,
            output_loading_info=True,
        ),
    )
    for kind in ("missing_keys", "unexpected_keys", "mismatched_keys"):
        assert not loading_info[kind], f"{kind} when loading the converted checkpoint: {loading_info[kind]}"
    return model.eval()


def assert_top_tokens_match(
    logits: torch.Tensor, other_logits: torch.Tensor, name: str, other_name: str, top_n: int = 2, top_k: int = 5
) -> None:
    """At every position, the top-``top_n`` tokens of each logits are in the top-``top_k`` of the other.

    Unlike the logits themselves, this does not depend on their scale, and tolerates bfloat16 rounding that swaps
    nearly equal logits. A token tied with the ``top_k``-th logit counts as top-``top_k``.
    """
    kth = logits.topk(top_k).values[..., -1:]
    other_kth = other_logits.topk(top_k).values[..., -1:]
    top = logits.topk(top_n).indices
    other_top = other_logits.topk(top_n).indices
    assert (other_logits.gather(-1, top) >= other_kth).all(), (
        f"a {name} top-{top_n} token is not in the {other_name} top-{top_k}"
    )
    assert (logits.gather(-1, other_top) >= kth).all(), (
        f"a {other_name} top-{top_n} token is not in the {name} top-{top_k}"
    )
