# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Parity between a torchtitan DomynEdge model and its HuggingFace conversion."""

import contextlib
import typing

import pytest
import torch
import transformers
from hf_utils import ATTN_IMPLEMENTATIONS, assert_top_tokens_match, build_titan_model, convert_to_hf, load_hf_model

from domyn_forge.models.domynedge import DomynEdgeModel, model_registry


@contextlib.contextmanager
def _record_outputs(modules: list[torch.nn.Module]) -> typing.Generator[list[torch.Tensor]]:
    """Record the output of each module, in call order, while the context is open."""
    outputs = []
    handles = [
        module.register_forward_hook(lambda _module, _args, output: outputs.append(output)) for module in modules
    ]
    try:
        yield outputs
    finally:
        for handle in handles:
            handle.remove()


def test_debugmodel_hf_config(tmp_path):
    """The converted config exposes per-layer head sizes, attention types and RoPE in their transformers 5 form."""
    # The registry types `.model` as the base `Config`, which declares no `layers`.
    layers = typing.cast(DomynEdgeModel.Config, model_registry("debugmodel", max_seq_len=2048).model).layers
    titan_model = build_titan_model()
    config = transformers.AutoConfig.from_pretrained(convert_to_hf(titan_model, tmp_path), trust_remote_code=True)

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

    # Titan's flash-attn left window = many tokens back + the current one, transformers' window = many tokens back.
    window_sizes = [layer.attention.sliding_window_size for layer in layers]
    sliding_window_sizes = {window_size for window_size in window_sizes if window_size != -1}
    assert len(sliding_window_sizes) == 1, f"Sliding window layers must share one size. Got {sliding_window_sizes}"
    assert config.sliding_window == sliding_window_sizes.pop() + 1
    assert "window_sizes" not in config.to_dict()

    # Global layers have no window (-1)
    titan_layer_types = ["full_attention" if window_size == -1 else "sliding_attention" for window_size in window_sizes]
    assert config.layer_types == titan_layer_types
    assert set(config.layer_types) == {"full_attention", "sliding_attention"}


@pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen attention requires a CUDA device")
@pytest.mark.parametrize("attn_implementation", ATTN_IMPLEMENTATIONS)
@torch.no_grad()
def test_debugmodel_hf_logits_match(tmp_path, attn_implementation):
    """The converted model gives the torchtitan residual stream and logits with every attention implementation.

    Both models are bfloat16 on the GPU, with float32 RoPE computed on the GPU as in training, so every step before the
    attention core is bit-identical.

    `torch_varlen` runs torchtitan's own flash kernel and window on the same shapes, so it must be bit-identical.
    `eager`, `sdpa` and `flex_attention` round differently from torchtitan's kernel in bfloat16, and that
    rounding accumulates in the residual stream: they are compared at the same rtol, with an atol that grows linearly
    through the layers (0.07 + 0.03 per layer), and an atol of 0.15 on the logits.
    """
    batch_size, seq_len, vocab_size = 2, 384, 2048
    device = torch.device("cuda")
    titan_model = build_titan_model(device="cuda")
    hf_model = load_hf_model(convert_to_hf(titan_model, tmp_path), attn_implementation)
    assert typing.cast(transformers.PreTrainedConfig, hf_model.config)._attn_implementation == attn_implementation

    generator = torch.Generator().manual_seed(0)
    tokens = torch.randint(0, vocab_size, (batch_size, seq_len), generator=generator).to(device)
    positions = torch.arange(seq_len, device=device).expand(batch_size, -1)

    # The stubs type submodules as `Tensor | Module`.
    hf_decoder = typing.cast(torch.nn.Module, hf_model.model)
    hf_modules = [
        typing.cast(torch.nn.Module, hf_decoder.embed_tokens),
        *typing.cast(torch.nn.ModuleList, hf_decoder.layers),
    ]
    with (
        _record_outputs([titan_model.tok_embeddings, *titan_model.layers.values()]) as titan_stream,
        _record_outputs(hf_modules) as hf_stream,
    ):
        titan_logits = titan_model(tokens, positions, titan_model.get_attention_masks(positions))
        hf_logits = hf_model(input_ids=tokens, position_ids=positions).logits

    def _assert_matches(name: str, hf_tensor: torch.Tensor, titan_tensor: torch.Tensor, atol: float) -> None:
        if attn_implementation == "torch_varlen":
            torch.testing.assert_close(hf_tensor, titan_tensor, rtol=0, atol=0, msg=lambda msg: f"{name}: {msg}")
        else:
            torch.testing.assert_close(
                hf_tensor, titan_tensor, rtol=1.6e-2, atol=atol, msg=lambda msg: f"{name}: {msg}"
            )

    # Compare the residual stream layer by layer. With PyTorch's bfloat16 rtol, the atol eager, sdpa and flex_attention
    # need grows about linearly through the layers, as their rounding accumulates in the residual stream.
    assert len(hf_stream) == len(titan_stream)
    for index, (hf_hidden, titan_hidden) in enumerate(zip(hf_stream, titan_stream, strict=True)):
        layer = index - 1  # the embeddings come first
        name = "embeddings" if layer < 0 else f"layer {layer}"
        _assert_matches(name, hf_hidden, titan_hidden, atol=0.07 + 0.03 * max(layer, 0))

    # The logits come after the final norm, much smaller than the deep residual stream: measured atol 0.093.
    assert titan_logits.shape == (batch_size, seq_len, vocab_size)
    _assert_matches("logits", hf_logits, titan_logits, atol=0.15)

    assert_top_tokens_match(hf_logits, titan_logits, "HF", "torchtitan")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen attention requires a CUDA device")
@torch.no_grad()
def test_debugmodel_hf_packed_documents_match(tmp_path):
    """Documents packed in one row give the torchtitan logits through HF `torch_varlen`.

    torchtitan trains on packed rows: positions restart at 0 at each document, and its varlen attention keeps each
    document separate. One document is longer than the 128-token sliding window, and the window must not reach
    back into the previous document. HF gets the same packed row and runs the same kernel, so the logits must be
    bit-identical.
    """
    document_lengths, vocab_size = [200, 100, 84], 2048
    device = torch.device("cuda")
    titan_model = build_titan_model(device="cuda")
    hf_model = load_hf_model(convert_to_hf(titan_model, tmp_path), "torch_varlen")

    generator = torch.Generator().manual_seed(0)
    documents = [torch.randint(0, vocab_size, (length,), generator=generator).to(device) for length in document_lengths]

    # torchtitan: the documents packed in one row, with positions restarting at 0 at each of them.
    tokens = torch.cat(documents)[None]
    positions = torch.cat([torch.arange(length, device=device) for length in document_lengths])[None]
    titan_logits = titan_model(tokens, positions, titan_model.get_attention_masks(positions))[0]

    # HF: the same packed row and positions, without cache (packed sequences are prefill only).
    hf_logits = hf_model(input_ids=tokens, position_ids=positions, use_cache=False).logits[0]

    start = 0
    for index, document in enumerate(documents):
        torch.testing.assert_close(
            hf_logits[start : start + len(document)],
            titan_logits[start : start + len(document)],
            rtol=0,
            atol=0,
            msg=lambda msg, index=index: f"document {index}: {msg}",
        )
        start += len(document)
