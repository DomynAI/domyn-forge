# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import dataclasses

import torch
import torch.nn.attention.flex_attention
import torchtitan.models.common
import torchtitan.models.utils
import torchtitan.protocols.module
from torchtitan.distributed.utils import is_in_batch_invariant_mode
from torchtitan.models.common.decoder import Decoder


class DomynEdgeAttention(torchtitan.models.common.attention.BaseAttention):
    """Grouped-query attention with per-layer head_dim, QK norm, optional RoPE and sliding window.

    Layers with ``sliding_window_size > 0`` attend within a local window; global layers use ``-1``.
    RoPE is applied only when ``use_rope`` is set (the global layers in the registered flavors).
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(torchtitan.models.common.attention.BaseAttention.Config):
        """Config for ``DomynEdgeAttention``."""

        q_heads: int
        kv_heads: int
        head_dim: int
        sliding_window_size: int
        q_norm: torchtitan.models.common.RMSNorm.Config
        k_norm: torchtitan.models.common.RMSNorm.Config
        # Each attention layer owns its RoPE config/cache (the supported
        # per-module pattern). ``use_rope`` gates whether it is applied; the
        # cache is still built so the base ``Decoder`` machinery can read and
        # resize ``attention.rope`` uniformly across all layers.
        rope: torchtitan.models.common.RoPE.Config
        use_rope: bool = False
        wq: torchtitan.models.common.Linear.Config
        wk: torchtitan.models.common.Linear.Config
        wv: torchtitan.models.common.Linear.Config
        wo: torchtitan.models.common.Linear.Config
        n_heads: int = dataclasses.field(init=False)
        moe = None

        def __post_init__(self):
            """Set ``n_heads`` from ``q_heads`` for the base attention machinery."""
            self.n_heads = self.q_heads

    def __init__(self, config: Config):
        """Build projections, QK norms, optional RoPE and the inner attention from ``config``."""
        super().__init__()
        self.config = config
        self.scaling = config.head_dim**-0.5
        self.head_dim = config.head_dim
        self.q_heads = config.q_heads
        self.kv_heads = config.kv_heads
        self.use_rope = config.use_rope
        # Only build the RoPE module (and its precomputed cache) when this layer
        # actually applies rotary embeddings. The ``rope`` config is still
        # present on every layer so the base ``Decoder.Config`` machinery
        # (``max_seq_len``, RoPE cache resize) stays uniform, but layers with
        # ``use_rope=False`` allocate no cache.
        self.rope = config.rope.build() if config.use_rope else None
        self.sliding_window_size = config.sliding_window_size
        inner_attention_config = config.inner_attention
        if hasattr(inner_attention_config, "window_size"):
            inner_attention_config = dataclasses.replace(
                inner_attention_config, window_size=(self.sliding_window_size, 0)
            )
        self.inner_attention = inner_attention_config.build()
        self.enable_gqa = self.q_heads > self.kv_heads

        self.q_norm = config.q_norm.build()
        self.k_norm = config.k_norm.build()

        self.wq = config.wq.build()
        self.wk = config.wk.build()
        self.wv = config.wv.build()
        self.wo = config.wo.build()

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: torchtitan.models.common.attention.AttentionMasksType,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Compute attention.

        Args:
            x: Input of shape ``(batch, seq_len, dim)``.
            attention_masks: Masks / metadata consumed by the inner attention.
            positions: Token positions used by RoPE.

        Returns:
            Output of shape ``(batch, seq_len, dim)``.
        """
        bs, seqlen, _ = x.shape
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)

        # reshape for multi-head attention
        xq = xq.view(bs, seqlen, self.q_heads, self.head_dim)
        xk = xk.view(bs, seqlen, self.kv_heads, self.head_dim)
        xv = xv.view(bs, seqlen, self.kv_heads, self.head_dim)

        # QK normalization
        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        # Apply rotary embeddings
        if self.rope is not None:
            xq, xk = self.rope(xq, xk, positions)

        output = self.inner_attention(
            xq,
            xk,
            xv,
            attention_masks=attention_masks,
            scale=self.scaling,
            enable_gqa=self.enable_gqa,
        ).contiguous()
        output = output.view(bs, seqlen, self.q_heads * self.head_dim)
        return self.wo(output)


class DomynEdgeTransformerBlock(torchtitan.protocols.module.Module):
    """Dense transformer block with pre- and post-norms around both attention and feed-forward.

    Computes ``x + post_attn_norm(attn(attention_norm(x)))`` then ``x + post_ffn_norm(ffn(ffn_norm(x)))``.
    """

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(torchtitan.protocols.module.Module.Config):
        """Config for ``DomynEdgeTransformerBlock``."""

        depth_init: bool = True
        moe_enabled: bool = False
        moe: None = None
        attention: DomynEdgeAttention.Config
        feed_forward: torchtitan.models.common.FeedForward.Config
        attention_norm: torchtitan.models.common.RMSNorm.Config
        post_attn_norm: torchtitan.models.common.RMSNorm.Config
        ffn_norm: torchtitan.models.common.RMSNorm.Config
        post_ffn_norm: torchtitan.models.common.RMSNorm.Config

    def __init__(self, config: Config):
        """Build the attention, feed-forward and norm submodules from ``config``.

        Raises:
            NotImplementedError: If ``config.moe_enabled`` is set.
        """
        super().__init__()

        self.moe_enabled = False

        if config.moe_enabled:
            msg = "MoE is not implemented for DomynEdgeTransformerBlock."
            raise NotImplementedError(msg)

        self.attention = config.attention.build()
        self.feed_forward = config.feed_forward.build()

        self.attention_norm = config.attention_norm.build()
        self.post_attn_norm = config.post_attn_norm.build()
        self.ffn_norm = config.ffn_norm.build()
        self.post_ffn_norm = config.post_ffn_norm.build()
        self._sliding_window_mask_key = f"sliding_window_mask_{config.attention.sliding_window_size}"

    def forward(
        self,
        x: torch.Tensor,
        attention_masks: torchtitan.models.common.attention.AttentionMasksType,
        positions: torch.Tensor,
    ):
        """Apply the block.

        Args:
            x: Input of shape ``(batch, seq_len, dim)``.
            attention_masks: Varlen metadata, or a dict of flex masks keyed by sliding-window size.
            positions: Token positions used by RoPE.

        Returns:
            Output of shape ``(batch, seq_len, dim)``.
        """
        # FlexAttention produces a dict of masks keyed by sliding-window size
        # ("basic_mask" plus per-window "sliding_window_mask_<n>"); select this
        # layer's mask here. The varlen path instead passes a plain
        # VarlenMetadata (not a dict), so this branch is skipped for it.
        if isinstance(attention_masks, dict) and "basic_mask" in attention_masks:
            layer_mask = attention_masks.get(self._sliding_window_mask_key, attention_masks["basic_mask"])
            assert layer_mask is not None
            attention_masks = layer_mask

        x = x + self.post_attn_norm(self.attention(self.attention_norm(x), attention_masks, positions))

        return x + self.post_ffn_norm(self.feed_forward(self.ffn_norm(x)))


class DomynEdgeModel(Decoder):
    """DomynEdge hybrid sliding-window / global attention decoder with tied input and output embeddings."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(Decoder.Config):
        """Config for ``DomynEdgeModel``."""

        enable_weight_tying: bool = True
        dim: int = dataclasses.field(init=False)
        vocab_size: int = dataclasses.field(init=False)

        def __post_init__(self):
            """Derive ``dim`` and ``vocab_size`` from the embedding config and require weight tying."""
            self.dim = self.tok_embeddings.embedding_dim
            self.vocab_size = self.tok_embeddings.num_embeddings
            if not self.enable_weight_tying:
                msg = "Non-weight-tied models are not implemented for DomynEdgeModel."
                raise NotImplementedError(msg)

        def get_nparams_and_flops(self, model: torch.nn.Module, seq_len: int) -> tuple[int, int]:
            """Estimate parameter count and FLOPs per token.

            Uses the average ``q_heads`` and ``head_dim`` across layers since they vary per layer.
            """
            avg_q_heads = sum(layer.attention.q_heads for layer in self.layers) // len(self.layers)
            avg_head_dim = sum(layer.attention.head_dim for layer in self.layers) // len(self.layers)
            return torchtitan.models.utils.get_dense_model_nparams_and_flops(
                model,
                n_layers=len(self.layers),
                n_heads=avg_q_heads,
                head_dims=avg_head_dim,
                seq_len=seq_len,
                enable_weight_tying=True,
            )

        def update_from_config(
            self,
            *,
            config,
            **kwargs,
        ) -> None:
            """Update the config from the job config, validate CP / TP support and set sharding.

            Raises:
                NotImplementedError: If context parallel is used with varlen attention.
                ValueError: If the TP degree does not divide the number of KV heads.
            """
            Decoder.Config.update_from_config(self, config=config, **kwargs)
            parallelism = config.parallelism

            if parallelism.context_parallel_degree > 1:
                inner_attention = self.layers[0].attention.inner_attention
                if isinstance(
                    inner_attention,
                    torchtitan.models.common.attention.VarlenAttention.Config,
                ):
                    msg = "Context Parallel with varlen attention is not supported. Use attn_backend='flex'."
                    raise NotImplementedError(msg)

            # The base TP check covers n_heads; domynedge names KV heads kv_heads, so validate divisibility explicitly.
            tp = parallelism.tensor_parallel_degree
            if tp > 1:
                attention = self.layers[0].attention
                n_kv_heads = attention.kv_heads or attention.q_heads
                if n_kv_heads % tp != 0:
                    msg = f"tensor_parallel_degree ({tp}) must divide n_kv_heads ({n_kv_heads})."
                    raise ValueError(msg)

            from domyn_forge.models.domynedge.sharding import set_domynedge_sharding_config

            set_domynedge_sharding_config(
                self,
                enable_sp=parallelism.enable_sequence_parallel,
            )

    def __init__(self, config: Config):
        """Build the decoder and tie ``tok_embeddings.weight`` to ``lm_head.weight``."""
        super().__init__(config)
        self.enable_weight_tying = config.enable_weight_tying
        if self.enable_weight_tying:
            self.tok_embeddings.weight = self.lm_head.weight

    def init_states(
        self,
        *,
        buffer_device: torch.device | None = None,
    ) -> None:
        """Re-tie embedding weights, then run the base ``init_states``."""
        if self.enable_weight_tying:
            # FSDP/TP sharding can replace the parameter assignment, so re-tie
            # before parameter init runs. Initialization of lm_head.weight then
            # populates the shared tensor seen by both tok_embeddings and lm_head.
            assert self.tok_embeddings is not None
            assert self.lm_head is not None
            self.tok_embeddings.weight = self.lm_head.weight
        super().init_states(buffer_device=buffer_device)

    def get_attention_masks(
        self,
        positions: torch.Tensor,
    ) -> torchtitan.models.common.attention.AttentionMasksType:
        """Build attention masks for the configured inner attention backend.

        Args:
            positions: Token positions (restarting per packed document) of shape ``(batch, seq_len)``.

        Returns:
            Varlen metadata, or a dict of flex block masks (``"basic_mask"`` plus one per sliding-window size).

        Raises:
            ValueError: If the inner attention type is not supported.
        """
        attn_config = self.config.first_attention
        assert attn_config is not None, "DomynEdgeModel has no attention layers"
        inner_attention = attn_config.inner_attention
        if isinstance(inner_attention, torchtitan.models.common.attention.VarlenAttention.Config):
            return torchtitan.models.common.attention.create_varlen_metadata_for_document(positions)
        if isinstance(inner_attention, torchtitan.models.common.attention.FlexAttention.Config):
            return self._get_flex_attention_masks(positions, attn_config)
        msg = f"Unknown attention type: {type(inner_attention)}"
        raise ValueError(msg)

    def _get_flex_attention_masks(
        self,
        positions: torch.Tensor,
        attn_config: torchtitan.models.common.attention.BaseAttention.Config,
    ) -> torchtitan.models.common.attention.AttentionMasksType:
        assert isinstance(attn_config.inner_attention, torchtitan.models.common.attention.FlexAttention.Config)
        batch_size, seq_len = positions.shape
        mask_mods = [
            torchtitan.models.common.attention.get_causal_mask_mod(),
            torchtitan.models.common.attention.get_efficient_causal_mask_mod_for_packed_document(positions),
        ]
        block_size = attn_config.inner_attention.block_size
        separate_full_blocks = not is_in_batch_invariant_mode()
        sliding_window_mask_mods = {
            layer_cfg.attention.sliding_window_size: torchtitan.models.common.get_sliding_window_mask_mod(
                layer_cfg.attention.sliding_window_size
            )
            for layer_cfg in self.config.layers
            if layer_cfg.attention.sliding_window_size > 0
        }
        sliding_window_masks = {
            f"sliding_window_mask_{k}": torchtitan.models.common.create_attention_mask(
                torch.nn.attention.flex_attention.and_masks(*mask_mods, v),
                batch_size,
                None,
                seq_len,
                seq_len,
                device=positions.device,
                BLOCK_SIZE=block_size,
                separate_full_blocks=separate_full_blocks,
            )
            for k, v in sliding_window_mask_mods.items()
        }
        masks: dict[str, torchtitan.models.common.attention.BlockMask] = {
            "basic_mask": torchtitan.models.common.attention.create_attention_mask(
                torch.nn.attention.flex_attention.and_masks(*mask_mods),
                batch_size,
                None,
                seq_len,
                seq_len,
                device=positions.device,
                BLOCK_SIZE=block_size,
                separate_full_blocks=separate_full_blocks,
            ),
            **sliding_window_masks,
        }
        return masks
