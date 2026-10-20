# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import copy
import functools

import torch
import torchtitan.components.loss
import torchtitan.distributed.pipeline_parallel
import torchtitan.models.common
import torchtitan.models.common.param_init
import torchtitan.models.llama3
import torchtitan.protocols

from .model import DomynEdgeAttention, DomynEdgeModel, DomynEdgeTransformerBlock
from .state_dict_adapter import DomynEdgeStateDictAdapter


def get_norm_init():
    """Return the param init spec for RMSNorm layers (weights initialized to one)."""
    return {
        "weight": torch.nn.init.ones_,
    }


def get_output_init(dim):
    """Return the param init spec for the embedding / output projection.

    Args:
        dim: Model hidden dimension; weights use a truncated normal with std ``dim**-0.5`` clipped at 3 std.

    Returns:
        Mapping from parameter name to init function.
    """
    return {
        "weight": functools.partial(
            torch.nn.init.trunc_normal_,
            std=dim**-0.5,
            a=-3 * dim**-0.5,
            b=3 * dim**-0.5,
        ),
        "bias": torch.nn.init.zeros_,
    }


def linear_init(layer):
    """Return the param init spec for a linear layer.

    Args:
        layer: Layer index used for depth-scaled std, or ``None`` for a fixed std of 0.02.

    Returns:
        Mapping from parameter name to init function.
    """
    return {
        "weight": functools.partial(
            torch.nn.init.trunc_normal_,
            std=0.02 if layer is None else torchtitan.models.common.param_init.depth_scaled_std(0.02, layer_id=layer),
        ),
        "bias": torch.nn.init.zeros_,
    }


def domynedge_configs(
    flavor: str, attn_backend: str, max_seq_len: int, vocab_size: int | None
) -> DomynEdgeModel.Config:
    """Build the DomynEdge model config for a given flavor.

    DomynEdge is a hybrid dense decoder: each layer has its own ``head_dim`` and either a sliding-window
    or a global (``sliding_window_size=-1``) attention. RoPE is applied only on the global layers.

    Args:
        flavor: Model flavor name (``"debugmodel"`` or ``"edgethinker"``).
        attn_backend: Inner attention backend (``"varlen"`` or ``"flex"``).
        max_seq_len: Maximum sequence length used for the RoPE cache.
        vocab_size: Vocabulary size; ``None`` selects the flavor default.

    Returns:
        The model config.

    Raises:
        ValueError: If ``attn_backend`` or ``flavor`` is not supported.
    """
    match attn_backend:
        case "varlen":
            inner_attention_cls = torchtitan.models.common.attention.VarlenAttention.Config
        case "flex":
            inner_attention_cls = torchtitan.models.common.attention.FlexAttention.Config
        case _:
            msg = f"Unsupported attention backend: {attn_backend}"
            raise ValueError(msg)

    # Per-layer factory; the ring backend would consume these args, varlen/flex ignore them.
    def inner_attention(*_args, **_kwargs):
        return inner_attention_cls()

    match flavor:
        case "debugmodel":
            vocab_size = 2048 if vocab_size is None else vocab_size
            return DomynEdgeModel.Config(
                norm=torchtitan.models.common.RMSNorm.Config(
                    normalized_shape=(dim := 128), eps=1e-6, param_init=get_norm_init()
                ),
                tok_embeddings=torchtitan.models.common.Embedding.Config(
                    num_embeddings=(vocab_size := 2048 if vocab_size is None else vocab_size),
                    embedding_dim=dim,
                    param_init=get_output_init(dim),
                ),
                lm_head=torchtitan.models.common.Linear.Config(
                    in_features=dim, out_features=vocab_size, param_init=get_output_init(dim)
                ),
                layers=[
                    DomynEdgeTransformerBlock.Config(
                        attention_norm=torchtitan.models.common.RMSNorm.Config(
                            normalized_shape=dim, eps=1e-6, param_init=get_norm_init()
                        ),
                        post_attn_norm=torchtitan.models.common.RMSNorm.Config(
                            normalized_shape=dim, eps=1e-6, param_init=get_norm_init()
                        ),
                        ffn_norm=torchtitan.models.common.RMSNorm.Config(
                            normalized_shape=dim, eps=1e-6, param_init=get_norm_init()
                        ),
                        post_ffn_norm=torchtitan.models.common.RMSNorm.Config(
                            normalized_shape=dim, eps=1e-6, param_init=get_norm_init()
                        ),
                        feed_forward=torchtitan.models.common.FeedForward.Config(
                            w1=torchtitan.models.common.Linear.Config(
                                in_features=dim, out_features=3072, param_init=linear_init(None)
                            ),
                            w2=torchtitan.models.common.Linear.Config(
                                in_features=3072, out_features=dim, param_init=linear_init(i)
                            ),
                            w3=torchtitan.models.common.Linear.Config(
                                in_features=dim, out_features=3072, param_init=linear_init(i)
                            ),
                        ),
                        attention=DomynEdgeAttention.Config(
                            q_heads=(q_heads := 4),
                            kv_heads=(kv_heads := 4),
                            head_dim=(head_dim := hs),
                            use_rope=hs == 64,
                            rope=torchtitan.models.common.CosSinRoPE.Config(
                                dim=head_dim,
                                max_seq_len=max_seq_len,
                                theta=10000.0,
                            ),
                            sliding_window_size=ws,
                            q_norm=torchtitan.models.common.RMSNorm.Config(
                                normalized_shape=head_dim,
                                eps=1e-6,
                                elementwise_affine=False,
                                param_init=get_norm_init(),
                            ),
                            k_norm=torchtitan.models.common.RMSNorm.Config(
                                normalized_shape=head_dim, eps=1e-6, elementwise_affine=True, param_init=get_norm_init()
                            ),
                            wq=torchtitan.models.common.Linear.Config(
                                in_features=dim, out_features=q_heads * head_dim, param_init=linear_init(None)
                            ),
                            wk=torchtitan.models.common.Linear.Config(
                                in_features=dim, out_features=kv_heads * head_dim, param_init=linear_init(i)
                            ),
                            wv=torchtitan.models.common.Linear.Config(
                                in_features=dim, out_features=kv_heads * head_dim, param_init=linear_init(i)
                            ),
                            wo=torchtitan.models.common.Linear.Config(
                                in_features=q_heads * head_dim, out_features=dim, param_init=linear_init(i)
                            ),
                            inner_attention=inner_attention(
                                cp_comm_type="a2a" if ws != -1 else "p2p",
                                layer_num=i,
                                head_size=head_dim,
                                kv_heads=kv_heads,
                                q_heads=q_heads,
                            ),
                        ),
                    )
                    for i, (hs, ws) in enumerate(
                        zip(
                            [64, 64] + [32, 32, 32, 64] * 2 + [64, 64],
                            [-1, -1] + [128, 128, 128, -1] * 2 + [-1, -1],
                            strict=True,
                        )
                    )
                ],
            )
        case "edgethinker":
            return DomynEdgeModel.Config(
                tok_embeddings=torchtitan.models.common.Embedding.Config(
                    num_embeddings=(vocab_size := 115328 if vocab_size is None else vocab_size),
                    embedding_dim=(dim := 2560),
                    param_init=get_output_init(dim),
                ),
                lm_head=torchtitan.models.common.Linear.Config(
                    in_features=dim, out_features=vocab_size, param_init=get_output_init(dim)
                ),
                norm=torchtitan.models.common.RMSNorm.Config(
                    normalized_shape=2560, eps=1e-6, param_init=get_norm_init()
                ),
                layers=[
                    DomynEdgeTransformerBlock.Config(
                        attention_norm=torchtitan.models.common.RMSNorm.Config(
                            normalized_shape=dim, eps=1e-6, param_init=get_norm_init()
                        ),
                        post_attn_norm=torchtitan.models.common.RMSNorm.Config(
                            normalized_shape=dim, eps=1e-6, param_init=get_norm_init()
                        ),
                        ffn_norm=torchtitan.models.common.RMSNorm.Config(
                            normalized_shape=dim, eps=1e-6, param_init=get_norm_init()
                        ),
                        post_ffn_norm=torchtitan.models.common.RMSNorm.Config(
                            normalized_shape=dim, eps=1e-6, param_init=get_norm_init()
                        ),
                        feed_forward=torchtitan.models.common.FeedForward.Config(
                            w1=torchtitan.models.common.Linear.Config(
                                in_features=dim, out_features=9728, param_init=linear_init(None)
                            ),
                            w2=torchtitan.models.common.Linear.Config(
                                in_features=9728, out_features=dim, param_init=linear_init(i)
                            ),
                            w3=torchtitan.models.common.Linear.Config(
                                in_features=dim, out_features=9728, param_init=linear_init(i)
                            ),
                        ),
                        attention=DomynEdgeAttention.Config(
                            q_heads=(q_heads := 32),
                            kv_heads=(kv_heads := 8),
                            head_dim=(head_dim := hs),
                            use_rope=hs == 64,
                            rope=torchtitan.models.common.CosSinRoPE.Config(
                                dim=head_dim,
                                max_seq_len=max_seq_len,
                                theta=10000.0,
                            ),
                            sliding_window_size=ws,
                            q_norm=torchtitan.models.common.RMSNorm.Config(
                                normalized_shape=head_dim,
                                eps=1e-6,
                                elementwise_affine=False,
                                param_init=get_norm_init(),
                            ),
                            k_norm=torchtitan.models.common.RMSNorm.Config(
                                normalized_shape=head_dim, eps=1e-6, elementwise_affine=True, param_init=get_norm_init()
                            ),
                            wq=torchtitan.models.common.Linear.Config(
                                in_features=dim, out_features=q_heads * head_dim, param_init=linear_init(None)
                            ),
                            wk=torchtitan.models.common.Linear.Config(
                                in_features=dim, out_features=kv_heads * head_dim, param_init=linear_init(i)
                            ),
                            wv=torchtitan.models.common.Linear.Config(
                                in_features=dim, out_features=kv_heads * head_dim, param_init=linear_init(i)
                            ),
                            wo=torchtitan.models.common.Linear.Config(
                                in_features=q_heads * head_dim, out_features=dim, param_init=linear_init(i)
                            ),
                            inner_attention=inner_attention(
                                cp_comm_type="a2a" if ws != -1 else "p2p",
                                layer_num=i,
                                head_size=head_dim,
                                kv_heads=kv_heads,
                                q_heads=q_heads,
                            ),
                        ),
                    )
                    for i, (hs, ws) in enumerate(
                        zip(
                            [256, 256] + [64, 64, 64, 256] * 8 + [256, 256],
                            [-1, -1] + [1024, 1024, 1024, -1] * 8 + [-1, -1],
                            strict=True,
                        )
                    )
                ],
            )
        case _:
            msg = f"Unsupported flavor: {flavor}"
            raise ValueError(msg)


def model_registry(
    flavor: str, attn_backend: str = "varlen", max_seq_len=131072, vocab_size=None
) -> torchtitan.protocols.ModelSpec:
    """Return the torchtitan ``ModelSpec`` for a DomynEdge flavor.

    Args:
        flavor: Model flavor name.
        attn_backend: Inner attention backend (``"varlen"`` or ``"flex"``).
        max_seq_len: Maximum sequence length used for the RoPE cache.
        vocab_size: Vocabulary size; ``None`` selects the flavor default.

    Returns:
        The model spec, using the Llama3 parallelize function and the DomynEdge state dict adapter.
    """
    model = copy.deepcopy(domynedge_configs(flavor, attn_backend, max_seq_len=max_seq_len, vocab_size=vocab_size))
    return torchtitan.protocols.ModelSpec(
        name="domynedge",
        flavor=flavor,
        model=model,
        parallelize_fn=torchtitan.models.llama3.parallelize_llama,
        pipelining_fn=torchtitan.distributed.pipeline_parallel.pipeline_llm,
        post_optimizer_build_fn=None,
        state_dict_adapter=DomynEdgeStateDictAdapter,
    )
