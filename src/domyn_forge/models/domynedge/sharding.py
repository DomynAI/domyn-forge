# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0
#
# This file includes third-party code:
#
# - The sharding helpers are adapted from torchtitan
#   (https://github.com/pytorch/torchtitan/blob/main/torchtitan/models/common/decoder_sharding.py),
#   Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#   Licensed under the BSD 3-Clause License (https://github.com/pytorch/torchtitan/blob/main/LICENSE).
#   Modified for DomynEdge.

from typing import TYPE_CHECKING

import spmd_types as spmd
from torchtitan.models.common.decoder_sharding import (
    colwise_config,
    dense_activation_placement,
    dense_param_placement,
    dense_sequence_parallel_placement,
    norm_config,
    rowwise_config,
    set_decoder_sharding_config,
    set_dense_ffn_sharding,
    set_gqa_inner_attention_local_map,
)
from torchtitan.protocols.sharding import ShardingConfig

if TYPE_CHECKING:
    from domyn_forge.models.domynedge.model import (
        DomynEdgeAttention,
        DomynEdgeModel,
        DomynEdgeTransformerBlock,
    )


def set_domynedge_sharding_config(
    config: "DomynEdgeModel.Config",
    *,
    enable_sp: bool,
) -> None:
    """Fill ``sharding_config`` on all DomynEdge sub-configs.

    Specs are populated unconditionally -- the mesh actually passed to
    ``Module.parallelize()`` at runtime determines which declarations
    apply. Declarations for mesh axes that aren't enabled (e.g. ``TP``
    placements under FSDP-only) are skipped at parallelize time.

    ``enable_sp`` controls SequenceParallel (decoupled from TP).
    """
    set_decoder_sharding_config(config, enable_sp=enable_sp)
    for layer_cfg in config.layers:
        _set_domynedge_layer_sharding(layer_cfg, enable_sp=enable_sp)


def set_domynedge_attention_sharding(attention_cfg: "DomynEdgeAttention.Config", *, enable_sp: bool) -> None:
    """DomynEdge attention TP sharding.

    Mirrors the shared ``set_gqa_attention_sharding`` (``x`` gathered to
    Replicate internally, RoPE ``cache`` buffer Replicate, ``wo`` rowwise),
    but DomynEdge uses three separate ``wq``/``wk``/``wv`` projections instead
    of a fused ``qkv_linear``, so it can't reuse the GQAttention helper. The
    ``q_norm``/``k_norm`` sharding is set by the caller.
    """
    attn_x_layout = dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.R)
    attention_cfg.sharding_config = ShardingConfig(
        in_src_shardings={"x": attn_x_layout},
        in_dst_shardings={"x": dense_activation_placement(tp=spmd.R)},
    )
    # Only layers that apply RoPE build a rope module (and its cache buffer);
    # skip the cache sharding for use_rope=False layers.
    if attention_cfg.use_rope:
        attention_cfg.rope.sharding_config = ShardingConfig(
            state_shardings={"cache": dense_param_placement(tp=spmd.R)},
        )
    attention_cfg.wq.sharding_config = colwise_config()
    attention_cfg.wk.sharding_config = colwise_config()
    attention_cfg.wv.sharding_config = colwise_config()
    attention_cfg.wo.sharding_config = rowwise_config(output_sp=enable_sp)


def _set_domynedge_layer_sharding(
    layer_cfg: "DomynEdgeTransformerBlock.Config",
    *,
    enable_sp: bool,
) -> None:
    """Set sharding on one DomynEdge transformer layer.

    ``enable_sp=True``  -> SP norms and Shard(1) activations around attention/FFN;
    ``attention.wo`` and ``feed_forward.w2`` reduce-scatter to Shard(1).
    ``enable_sp=False`` -> norms stay Replicate (no parallelism), activations
    stay Replicate; ``attention.wo`` and ``feed_forward.w2`` all-reduce to
    Replicate.
    """
    norm = norm_config(enable_sp=enable_sp)
    layer_cfg.attention_norm.sharding_config = norm
    layer_cfg.ffn_norm.sharding_config = norm
    layer_cfg.post_attn_norm.sharding_config = norm
    layer_cfg.post_ffn_norm.sharding_config = norm

    set_domynedge_attention_sharding(layer_cfg.attention, enable_sp=enable_sp)
    set_gqa_inner_attention_local_map(layer_cfg.attention.inner_attention)

    # QK norms: shard on head dim (dim=2) — independent of SP.
    # TODO: also set out_src_shardings (same head-sharded placement as out_dst_shardings). Without it the
    # spmd_types backend, torchtitan's default, raises "RMSNorm: SPMD output redistribution requires explicit
    # out_src_shardings" on every multi-GPU run, so run configs use spmd_backend="partial_dtensor" for now.
    qk_norm = ShardingConfig(
        state_shardings={"weight": dense_param_placement(tp=spmd.R)},
        in_src_shardings={"input": dense_activation_placement(tp=spmd.S(2))},
        in_dst_shardings={"input": dense_activation_placement(tp=spmd.S(2))},
        out_dst_shardings=dense_activation_placement(tp=spmd.S(2)),
    )
    layer_cfg.attention.q_norm.sharding_config = qk_norm
    layer_cfg.attention.k_norm.sharding_config = qk_norm

    attn_x_layout = dense_sequence_parallel_placement() if enable_sp else dense_activation_placement(tp=spmd.R)
    assert layer_cfg.feed_forward is not None
    set_dense_ffn_sharding(
        layer_cfg.feed_forward,
        attn_x_layout=attn_x_layout,
        enable_sp=enable_sp,
    )
