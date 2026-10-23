# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

from transformers.configuration_utils import PreTrainedConfig
from transformers.modeling_rope_utils import RopeParameters


class DomynEdgeConfig(PreTrainedConfig):
    """Configuration of a DomynEdge model, with per-layer position encoding, sliding window and head size."""

    model_type: str = "domynedge"
    default_theta: float = 10_000

    def __init__(
        self,
        vocab_size: int = 115269,
        hidden_size: int = 2560,
        intermediate_size: int = 9728,
        num_hidden_layers: int = 36,
        num_attention_heads: int = 32,
        num_key_value_heads: int = 8,
        max_position_embeddings: int = 8192,
        initializer_range: float = 0.02,
        rms_norm_eps: float = 1e-6,
        *,
        use_cache: bool = True,
        tie_word_embeddings: bool = True,
        rope_parameters: RopeParameters | None = None,
        max_window_layers: int | None = None,
        attention_bias: bool = False,
        attention_dropout: float = 0.0,
        position_layers: list[str] = ["nope", "nope"] + ["rope", "rope", "rope", "nope"] * 8 + ["nope", "nope"],
        window_sizes: list[tuple[int, int]] = [(-1, 0), (-1, 0)]
        + [(1024, 0), (1024, 0), (1024, 0), (-1, 0)] * 8
        + [(-1, 0), (-1, 0)],
        head_sizes: list[int] = [256, 256] + [64, 64, 64, 256] * 8 + [256, 256],
        pad_token_id: int | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        **kwargs,
    ):
        """Initialize and validate the per-layer settings, which must all have `num_hidden_layers` entries."""
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout

        self.position_layers = position_layers
        self.window_sizes = window_sizes
        self.head_sizes = head_sizes

        # Validate position_layers
        if len(self.position_layers) != self.num_hidden_layers:
            msg = (
                f"Length of position_layers ({len(self.position_layers)}) "
                f"must be equal to num_hidden_layers ({self.num_hidden_layers})"
            )
            raise ValueError(msg)
        if any(pl not in ["rope", "nope"] for pl in self.position_layers):
            msg = f"position_layers must be one of 'rope', 'nope'. Got {self.position_layers}"
            raise ValueError(msg)
        # Validate window_sizes
        if len(self.window_sizes) != self.num_hidden_layers:
            msg = (
                f"Length of window_sizes ({len(self.window_sizes)}) "
                f"must be equal to num_hidden_layers ({self.num_hidden_layers})"
            )
            raise ValueError(msg)
        # Validate head_sizes
        if len(head_sizes) != self.num_hidden_layers:
            msg = (
                f"Length of head_sizes ({len(head_sizes)}) "
                f"must be equal to num_hidden_layers ({self.num_hidden_layers})"
            )
            raise ValueError(msg)
        if any(not isinstance(hs, int) for hs in head_sizes):
            msg = f"head_sizes must be a list of integers. Got {head_sizes}"
            raise ValueError(msg)
        # Only one rope dim
        rope_dims = [head_size for rope, head_size in zip(position_layers, head_sizes, strict=True) if rope == "rope"]
        if len(set(rope_dims)) > 1:
            msg = f"All rope layers must have the same head size. Got {rope_dims}"
            raise ValueError(msg)

        kwargs.pop("rope_dim", None)  # derived from rope_dims, never read from config.json. It won't be passed to super
        self.rope_dim = rope_dims[0]

        # If None, `super().__init__` calls `RotaryEmbeddingConfigMixin.convert_rope_params_to_dict`, which builds
        # `{"rope_type": "default", "rope_theta": ...}` with the legacy top-level `rope_theta` kwarg if present,
        # else `default_theta`.
        self.rope_parameters = rope_parameters

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings

        self.head_dim = self.rope_dim
        if "per_layer_config" not in kwargs:
            overrides = {
                layer_idx: {"head_dim": head_size}
                for layer_idx, head_size in enumerate(head_sizes)
                if head_size != self.head_dim
            }
            if overrides:
                kwargs["per_layer_config"] = overrides

        super().__init__(**kwargs)


__all__ = ["DomynEdgeConfig"]
