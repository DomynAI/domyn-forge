# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the DomynEdge model, its flavors, state dict adapter and sharding config.

The config, tying, adapter and sharding tests run on CPU. The forward tests need a CUDA device:
varlen attention is CUDA only, and flex attention compiles its block masks with inductor.
"""

import dataclasses
import types
import typing

import pytest
import torch
import torchtitan.models.common.attention
from torchtitan.config import ParallelismConfig

from domyn_forge.models.domynedge import (
    DomynEdgeModel,
    DomynEdgeStateDictAdapter,
    DomynEdgeTransformerBlock,
    domynedge_configs,
    model_registry,
)

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="forward tests need a CUDA device")

MAX_SEQ_LEN = 512
DEBUG_VOCAB_SIZE = 2048
DEBUG_WINDOW = 128

# Per-layer (head_dim, sliding_window_size, use_rope) of each flavor. RoPE is on the sliding-window layers in both
# flavors (matching the HF config's default `position_layers`).
DEBUG_LAYERS = [
    (head_dim, window, window != -1)
    for head_dim, window in zip(
        [64, 64] + [32, 32, 32, 64] * 2 + [64, 64], [-1, -1] + [128, 128, 128, -1] * 2 + [-1, -1], strict=True
    )
]
EDGETHINKER_LAYERS = [
    (head_dim, window, window != -1)
    for head_dim, window in zip(
        [256, 256] + [64, 64, 64, 256] * 8 + [256, 256], [-1, -1] + [1024, 1024, 1024, -1] * 8 + [-1, -1], strict=True
    )
]


def build_debug_model(attn_backend: str = "varlen", seed: int = 0) -> DomynEdgeModel:
    torch.manual_seed(seed)
    with torch.device("cpu"):
        model = model_registry("debugmodel", attn_backend=attn_backend, max_seq_len=MAX_SEQ_LEN).model.build()
    assert isinstance(model, DomynEdgeModel)
    model.init_states()
    return model


def blocks(model: DomynEdgeModel) -> list[DomynEdgeTransformerBlock]:
    """Return the transformer blocks of ``model`` in layer order, typed for attribute access."""
    layers = list(model.layers.values())
    assert all(isinstance(layer, DomynEdgeTransformerBlock) for layer in layers)
    return typing.cast(list[DomynEdgeTransformerBlock], layers)


def adapter_for(model: DomynEdgeModel) -> DomynEdgeStateDictAdapter:
    config = model.config
    assert isinstance(config, DomynEdgeModel.Config)
    return DomynEdgeStateDictAdapter(config, None)


def update_from_parallelism(config: DomynEdgeModel.Config, **parallelism) -> None:
    config.update_from_config(config=types.SimpleNamespace(parallelism=ParallelismConfig(**parallelism)))


# Flavors


@pytest.mark.parametrize(
    ("flavor", "layers", "dim", "vocab_size", "q_heads", "kv_heads"),
    [
        ("debugmodel", DEBUG_LAYERS, 128, DEBUG_VOCAB_SIZE, 4, 4),
        ("edgethinker", EDGETHINKER_LAYERS, 2560, 115328, 32, 8),
    ],
)
def test_flavor_layer_pattern(flavor, layers, dim, vocab_size, q_heads, kv_heads):
    config = domynedge_configs(flavor, "varlen", max_seq_len=MAX_SEQ_LEN, vocab_size=None)

    assert config.dim == dim
    assert config.vocab_size == vocab_size
    assert config.enable_weight_tying
    assert [
        (layer.attention.head_dim, layer.attention.sliding_window_size, layer.attention.use_rope)
        for layer in config.layers
    ] == layers
    for layer in config.layers:
        attention = layer.attention
        assert (attention.q_heads, attention.kv_heads) == (q_heads, kv_heads)
        assert attention.n_heads == q_heads
        assert attention.wq.out_features == q_heads * attention.head_dim
        assert attention.wk.out_features == attention.wv.out_features == kv_heads * attention.head_dim
        assert attention.wo.in_features == q_heads * attention.head_dim
        assert attention.q_norm.normalized_shape == attention.k_norm.normalized_shape == attention.head_dim
        assert attention.rope.dim == attention.head_dim
        assert attention.rope.max_seq_len == MAX_SEQ_LEN


@pytest.mark.parametrize("flavor", ["debugmodel", "edgethinker"])
def test_vocab_size_override(flavor):
    config = domynedge_configs(flavor, "varlen", max_seq_len=MAX_SEQ_LEN, vocab_size=1000)
    assert config.vocab_size == config.tok_embeddings.num_embeddings == config.lm_head.out_features == 1000


@pytest.mark.parametrize(
    ("attn_backend", "inner_attention_cls"),
    [
        ("varlen", torchtitan.models.common.attention.VarlenAttention.Config),
        ("flex", torchtitan.models.common.attention.FlexAttention.Config),
    ],
)
def test_attention_backend(attn_backend, inner_attention_cls):
    config = domynedge_configs("debugmodel", attn_backend, max_seq_len=MAX_SEQ_LEN, vocab_size=None)
    assert all(isinstance(layer.attention.inner_attention, inner_attention_cls) for layer in config.layers)


def test_unsupported_flavor_raises():
    with pytest.raises(ValueError, match="Unsupported flavor"):
        domynedge_configs("nope", "varlen", max_seq_len=MAX_SEQ_LEN, vocab_size=None)


def test_unsupported_attention_backend_raises():
    with pytest.raises(ValueError, match="Unsupported attention backend"):
        domynedge_configs("debugmodel", "ring", max_seq_len=MAX_SEQ_LEN, vocab_size=None)


def test_untied_embeddings_raise():
    config = domynedge_configs("debugmodel", "varlen", max_seq_len=MAX_SEQ_LEN, vocab_size=None)
    with pytest.raises(NotImplementedError, match="Non-weight-tied"):
        dataclasses.replace(config, enable_weight_tying=False)


# Built model


def test_rope_built_only_where_used():
    model = build_debug_model()
    for layer, (_, _, use_rope) in zip(blocks(model), DEBUG_LAYERS, strict=True):
        assert (layer.attention.rope is not None) == use_rope


def test_embeddings_tied_after_init_states():
    model = build_debug_model()
    assert model.tok_embeddings.weight is model.lm_head.weight
    # Re-initializing must keep the tie and refill the one shared tensor.
    before = model.lm_head.weight.detach().clone()
    model.init_states()
    assert model.tok_embeddings.weight is model.lm_head.weight
    assert not torch.equal(model.lm_head.weight, before)


def test_parameter_count_counts_tied_weights_once():
    model = build_debug_model()
    nparams, _ = model.config.get_nparams_and_flops(model, seq_len=MAX_SEQ_LEN)
    assert nparams == sum(param.numel() for param in model.parameters())


# State dict adapter


def test_state_dict_adapter_to_hf_keys():
    model = build_debug_model()
    adapter = adapter_for(model)
    state_dict = model.state_dict()
    hf_state_dict = adapter.to_hf(state_dict)

    # Every weight is exported except the head, which HF ties to the embeddings.
    assert "lm_head.weight" not in hf_state_dict
    assert len(hf_state_dict) == len(state_dict) - 1
    assert {"model.embed_tokens.weight", "model.norm.weight"} <= set(hf_state_dict)
    for i in range(len(DEBUG_LAYERS)):
        assert f"model.layers.{i}.attention.q_proj.weight" in hf_state_dict
        assert f"model.layers.{i}.post_feedforward_norm.weight" in hf_state_dict
    assert not any("rope" in key for key in hf_state_dict)


def test_state_dict_adapter_round_trip():
    model = build_debug_model()
    adapter = adapter_for(model)
    state_dict = model.state_dict()

    restored = adapter.from_hf(adapter.to_hf(state_dict))

    assert set(restored) == set(state_dict)
    for key, value in state_dict.items():
        assert torch.equal(restored[key], value), key
    # The head comes back from the embeddings.
    assert torch.equal(restored["lm_head.weight"], state_dict["tok_embeddings.weight"])


def test_state_dict_adapter_loads_into_fresh_model():
    source = build_debug_model(seed=0)
    target = build_debug_model(seed=1)
    adapter = adapter_for(source)

    result = target.load_state_dict(adapter.from_hf(adapter.to_hf(source.state_dict())), strict=True)

    assert not result.missing_keys
    assert not result.unexpected_keys
    for (name, expected), (_, actual) in zip(source.state_dict().items(), target.state_dict().items(), strict=True):
        assert torch.equal(expected, actual), name


# update_from_config and sharding


def test_tensor_parallel_must_divide_kv_heads():
    # edgethinker has 32 query heads and 8 KV heads: TP 16 divides the former but not the latter.
    config = domynedge_configs("edgethinker", "varlen", max_seq_len=MAX_SEQ_LEN, vocab_size=None)
    with pytest.raises(ValueError, match=r"n_kv_heads \(8\)"):
        update_from_parallelism(config, tensor_parallel_degree=16)


def test_context_parallel_with_varlen_raises():
    config = domynedge_configs("debugmodel", "varlen", max_seq_len=MAX_SEQ_LEN, vocab_size=None)
    with pytest.raises(NotImplementedError, match="Context Parallel with varlen"):
        update_from_parallelism(config, context_parallel_degree=2)


def test_context_parallel_with_flex_is_allowed():
    config = domynedge_configs("debugmodel", "flex", max_seq_len=MAX_SEQ_LEN, vocab_size=None)
    update_from_parallelism(config, context_parallel_degree=2)


@pytest.mark.parametrize("enable_sp", [False, True])
def test_sharding_config_is_set(enable_sp):
    config = domynedge_configs("debugmodel", "varlen", max_seq_len=MAX_SEQ_LEN, vocab_size=None)
    update_from_parallelism(config, tensor_parallel_degree=2, enable_sequence_parallel=enable_sp)

    for layer in config.layers:
        attention = layer.attention
        for sub_config in (
            layer.attention_norm,
            layer.post_attn_norm,
            layer.ffn_norm,
            layer.post_ffn_norm,
            attention,
            attention.wq,
            attention.wk,
            attention.wv,
            attention.wo,
            attention.q_norm,
            attention.k_norm,
            layer.feed_forward.w1,
            layer.feed_forward.w2,
            layer.feed_forward.w3,
        ):
            assert sub_config.sharding_config is not None
        assert (attention.rope.sharding_config is not None) == attention.use_rope


# Forward (CUDA)


def forward(model: DomynEdgeModel, tokens: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    with torch.autocast("cuda", dtype=torch.bfloat16):
        return model(tokens, positions, model.get_attention_masks(positions)).float()


def random_tokens(batch_size: int, seq_len: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, DEBUG_VOCAB_SIZE, (batch_size, seq_len), generator=generator).cuda()


def arange_positions(batch_size: int, seq_len: int) -> torch.Tensor:
    return torch.arange(seq_len, device="cuda").expand(batch_size, -1)


def relative_error(actual: torch.Tensor, expected: torch.Tensor) -> float:
    """Relative L2 error; robust to the near-zero elements that make element-wise bfloat16 checks flaky."""
    return ((actual - expected).norm() / expected.norm()).item()


# Different kernels (varlen vs flex, or different sequence lengths) round differently in bfloat16; over the 12 debug
# layers the logits drift by about 1% relative L2.
BF16_RELATIVE_TOLERANCE = 2e-2


@requires_cuda
@pytest.mark.parametrize("attn_backend", ["varlen", "flex"])
@torch.no_grad()
def test_forward_shape(attn_backend):
    model = build_debug_model(attn_backend).cuda().eval()
    tokens = random_tokens(2, 256)

    logits = forward(model, tokens, arange_positions(2, 256))

    assert logits.shape == (2, 256, DEBUG_VOCAB_SIZE)
    assert torch.isfinite(logits).all()


@requires_cuda
def test_varlen_and_flex_match():
    """Both backends must give the same logits and gradients for the same weights and packed batch."""
    varlen_model = build_debug_model("varlen").cuda()
    flex_model = build_debug_model("flex").cuda()
    flex_model.load_state_dict(varlen_model.state_dict())

    # Longer than the sliding window, with two packed documents in the second row.
    seq_len = 384
    tokens = random_tokens(2, seq_len)
    positions = arange_positions(2, seq_len).clone()
    positions[1, 200:] = torch.arange(seq_len - 200, device="cuda")

    varlen_logits = forward(varlen_model, tokens, positions)
    flex_logits = forward(flex_model, tokens, positions)
    logits_error = relative_error(flex_logits, varlen_logits)
    assert logits_error < BF16_RELATIVE_TOLERANCE, f"relative L2 error of the logits is {logits_error:.4f}"

    grad = torch.randn_like(varlen_logits)
    varlen_logits.backward(grad)
    flex_logits.backward(grad)
    # Compared as one flattened vector: per-parameter checks fail on noise for tiny gradients such as norm weights.
    varlen_grads, flex_grads = [], []
    flex_params = dict(flex_model.named_parameters())
    for name, param in varlen_model.named_parameters():
        flex_grad = flex_params[name].grad
        assert param.grad is not None, name
        assert flex_grad is not None, name
        varlen_grads.append(param.grad.flatten())
        flex_grads.append(flex_grad.flatten())
    grad_error = relative_error(torch.cat(flex_grads), torch.cat(varlen_grads))
    assert grad_error < BF16_RELATIVE_TOLERANCE, f"relative L2 error of the gradients is {grad_error:.4f}"


@requires_cuda
@pytest.mark.parametrize("attn_backend", ["varlen", "flex"])
@torch.no_grad()
def test_forward_is_causal(attn_backend):
    model = build_debug_model(attn_backend).cuda().eval()
    tokens = random_tokens(1, 256)
    changed = tokens.clone()
    changed[0, 200:] = (changed[0, 200:] + 1) % DEBUG_VOCAB_SIZE
    positions = arange_positions(1, 256)

    logits = forward(model, tokens, positions)
    changed_logits = forward(model, changed, positions)

    torch.testing.assert_close(logits[:, :200], changed_logits[:, :200])
    assert not torch.allclose(logits[:, 200:], changed_logits[:, 200:])


@requires_cuda
@pytest.mark.parametrize("attn_backend", ["varlen", "flex"])
@torch.no_grad()
def test_packed_documents_match_separate_documents(attn_backend):
    """Positions restarting at 0 start a new document that must not attend to the previous one."""
    model = build_debug_model(attn_backend).cuda().eval()
    first, second = random_tokens(1, 256, seed=0), random_tokens(1, 256, seed=1)
    packed_positions = torch.cat([arange_positions(1, 256), arange_positions(1, 256)], dim=1)

    both = torch.cat([first, second], dim=1)
    packed = forward(model, both, packed_positions)
    alone = forward(model, second, arange_positions(1, 256))
    # The same tokens as one document: the second half now attends to the first, which the tolerance must catch.
    continuous = forward(model, both, arange_positions(1, 512))

    packed_error = relative_error(packed[:, 256:], alone)
    continuous_error = relative_error(continuous[:, 256:], alone)
    assert packed_error < BF16_RELATIVE_TOLERANCE, f"relative L2 error of the logits is {packed_error:.4f}"
    assert continuous_error > BF16_RELATIVE_TOLERANCE, f"a single document only differs by {continuous_error:.4f}"


@requires_cuda
@pytest.mark.parametrize("attn_backend", ["varlen", "flex"])
@torch.no_grad()
def test_sliding_window_layer_ignores_tokens_outside_window(attn_backend):
    """A sliding-window block only sees the last ``DEBUG_WINDOW`` tokens; a global block sees all of them."""
    model = build_debug_model(attn_backend).cuda().eval()
    seq_len = 384
    positions = arange_positions(1, seq_len)
    masks = model.get_attention_masks(positions)
    hidden = torch.randn(1, seq_len, 128, device="cuda")
    changed = hidden.clone()
    changed[0, 0] += 1.0
    # First position whose window no longer contains position 0.
    outside = DEBUG_WINDOW + 1

    windowed_block, global_block = blocks(model)[2], blocks(model)[0]
    assert windowed_block.attention.sliding_window_size == DEBUG_WINDOW
    assert global_block.attention.sliding_window_size == -1

    with torch.autocast("cuda", dtype=torch.bfloat16):
        windowed = windowed_block(hidden, masks, positions)
        windowed_changed = windowed_block(changed, masks, positions)
        global_ = global_block(hidden, masks, positions)
        global_changed = global_block(changed, masks, positions)

    torch.testing.assert_close(windowed[:, outside:], windowed_changed[:, outside:])
    assert not torch.allclose(windowed[:, 1:outside], windowed_changed[:, 1:outside])
    assert not torch.allclose(global_[:, outside:], global_changed[:, outside:])
