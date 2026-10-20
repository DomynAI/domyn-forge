# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import transformers


class RopeParameters:
    def __init__(
        self,
        rope_type: str = "default",
        rope_theta: float = 10_000,
    ):
        self.rope_type = rope_type
        self.rope_theta = rope_theta


# to support both transformers > 5 and older
PretrainedConfig = (
    transformers.modeling_utils.PreTrainedConfig
    if hasattr(transformers.modeling_utils, "PreTrainedConfig")
    else transformers.configuration_utils.PretrainedConfig
)


class DomynEdgeConfig(PretrainedConfig):
    """Configuration of a DomynEdge model, with per-layer position encoding, sliding window and head size."""

    model_type: str = "domynedge"

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
        self.rope_dim = rope_dims[0]

        # Set default rope_parameters if not provided
        if rope_parameters is None:
            self.rope_parameters = RopeParameters(
                rope_type="default",
                rope_theta=10_000,
            )

        self.pad_token_id = pad_token_id
        self.bos_token_id = bos_token_id
        self.eos_token_id = eos_token_id
        self.tie_word_embeddings = tie_word_embeddings
        self.rope_parameters = rope_parameters

        super().__init__(**kwargs)


__all__ = ["DomynEdgeConfig"]
