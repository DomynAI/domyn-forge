# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0
#
# This file includes third-party code:
#
# - `DomynEdgeRotaryEmbedding.compute_default_rope_parameters` and `apply_rotary_pos_emb` are adapted from
#   HuggingFace transformers
#   (https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py),
#   Copyright The HuggingFace Team, licensed under the Apache License, Version 2.0.
#   Modified for DomynEdge.

import typing

import torch
import torch.nn.attention.varlen
import transformers
import transformers.modeling_outputs
from transformers.masking_utils import create_causal_mask, create_sliding_window_causal_mask, flash_attention_mask
from transformers.modeling_flash_attention_utils import (
    _is_packed_sequence,
    _pad_input,
    _prepare_from_posids,
    _unpad_input,
    _upad_input,
)
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
from transformers.models.llama.modeling_llama import eager_attention_forward, rotate_half
from transformers.utils.generic import can_return_tuple, merge_with_config_defaults
from transformers.utils.output_capturing import capture_outputs

from .configuration_domynedge import DomynEdgeConfig


class DomynEdgeFeedForward(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class DomynEdgeRotaryEmbedding(torch.nn.Module):
    inv_freq: torch.Tensor  # fix linting for `register_buffer`

    def __init__(self, config: DomynEdgeConfig, device=None):
        super().__init__()
        self.max_seq_len_cached = config.max_position_embeddings
        self.original_max_seq_len = config.max_position_embeddings

        self.config = config

        assert self.config.rope_parameters is not None
        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: typing.Callable = self.compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: DomynEdgeConfig,
        device: typing.Optional["torch.device"] = None,
    ) -> tuple["torch.Tensor", float]:
        """Computes the inverse frequencies according to the original RoPE implementation.

        Args:
            config ([`~transformers.PreTrainedConfig`]):
                The model configuration.
            device (`torch.device`):
                The device to use for initialization of the inverse frequencies.

        Returns:
            Tuple of (`torch.Tensor`, `float`), containing the inverse frequencies for the RoPE embeddings and the
            post-processing scaling factor applied to the computed cos/sin (unused in this type of RoPE).
        """
        assert config.rope_parameters is not None
        assert config.rope_parameters["rope_theta"] is not None

        base = config.rope_parameters["rope_theta"]
        dim = config.rope_dim

        attention_factor = 1.0  # Unused in this type of RoPE

        # Compute the inverse frequencies
        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        return inv_freq, attention_factor

    @torch.no_grad()
    def forward(self, x, position_ids):
        # Recomputed on the input's device instead of reading the `inv_freq` buffer, to match the training
        # implementation.
        inv_freq, _ = self.compute_default_rope_parameters(self.config, x.device)
        # An elementwise product, not a matmul, so torch.autocast(bfloat16) does not lower it.
        freqs = position_ids[..., None].float() * inv_freq
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The dimension along which to unsqueeze cos and sin so that they broadcast to q and k. Use 1 when q and k
            have the shape [batch_size, heads, seq_len, head_dim], 2 for [batch_size, seq_len, heads, head_dim].

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    qf, kf = q.float(), k.float()
    q_embed = (qf * cos) + (rotate_half(qf) * sin)
    k_embed = (kf * cos) + (rotate_half(kf) * sin)
    # Main difference with respect to transformers.models.llama.modeling_llama, llama does not do the upcast
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


def torch_varlen_attention_forward(
    _module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float | None = None,
    sliding_window: int | None = None,
    position_ids: torch.Tensor | None = None,
    **_kwargs,
) -> tuple[torch.Tensor, None]:
    """Attention through PyTorch's built-in flash kernel, the one the model is trained with.

    Selected with `attn_implementation="torch_varlen"`, it needs no `flash_attn` package.
    """
    batch_size, _, q_len, head_dim = query.shape
    query, key, value = query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2)
    indices_q = None
    if attention_mask is not None:
        query, key, value, indices_q, (cu_seq_q, cu_seq_k), (max_q, max_k) = _upad_input(
            query, key, value, attention_mask, q_len, _unpad_input
        )
    elif _is_packed_sequence(position_ids, batch_size):
        query, key, value, (cu_seq_q, cu_seq_k), (max_q, max_k) = _prepare_from_posids(query, key, value, position_ids)
        max_q, max_k = int(max_q), int(max_k)
    else:
        k_len = key.shape[1]
        query, key, value = query.flatten(0, 1), key.flatten(0, 1), value.flatten(0, 1)
        cu_seq_q = torch.arange(0, (batch_size + 1) * q_len, q_len, dtype=torch.int32, device=query.device)
        cu_seq_k = torch.arange(0, (batch_size + 1) * k_len, k_len, dtype=torch.int32, device=query.device)
        max_q, max_k = q_len, k_len

    output = torch.nn.attention.varlen.varlen_attn(
        query,
        key,
        value,
        cu_seq_q,
        cu_seq_k,
        max_q,
        max_k,
        scale=scaling,
        # transformers' `sliding_window` counts the current token, flash attention's left window does not.
        window_size=(-1 if sliding_window is None else sliding_window - 1, 0),
        enable_gqa=query.shape[1] != key.shape[1],
    )
    assert isinstance(output, torch.Tensor)
    if indices_q is None:
        return output.view(batch_size, q_len, -1, head_dim), None
    return _pad_input(output, indices_q, batch_size, q_len), None


transformers.AttentionInterface.register("torch_varlen", torch_varlen_attention_forward)
# Without a registered mask function, transformers passes no mask at all to a custom implementation, padding included.
transformers.AttentionMaskInterface.register("torch_varlen", flash_attention_mask)


class DomynEdgeAttention(torch.nn.Module):
    """Grouped Query Attention with per-layer head size, optional RoPE and optional sliding window.

    Attention is dispatched through `ALL_ATTENTION_FUNCTIONS`, so any transformers attention implementation
    (eager, sdpa, flash_attention_2, flex) and vLLM's Transformers backend can serve it.
    """

    def __init__(self, config: DomynEdgeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.layer_type = config.layer_types[layer_idx]
        self.head_dim = config.per_layer_config[layer_idx].head_dim
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim**-0.5
        self.attention_dropout = config.attention_dropout
        self.is_causal = True
        self.use_rope = config.position_layers[layer_idx] == "rope"
        self.sliding_window = config.sliding_window if self.layer_type == "sliding_attention" else None

        self.q_proj = torch.nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = torch.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = torch.nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = torch.nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        self.q_norm = torch.nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=False)
        self.k_norm = torch.nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps, elementwise_affine=True)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
        past_key_values: transformers.Cache | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        batch_size, sequence_length, _ = hidden_states.shape
        query_shape = (batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
        key_value_shape = (batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)

        query_states = self.q_norm(self.q_proj(hidden_states).view(query_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(key_value_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(key_value_shape).transpose(1, 2)

        if self.use_rope:
            cos, sin = position_embeddings
            query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # `get_interface` handles a None implementation, despite its `str` annotation
        attention_interface: typing.Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation,  # noqa: SLF001  # pyright: ignore[reportArgumentType]
            eager_attention_forward,
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )

        attn_output = attn_output.reshape(
            batch_size, sequence_length, self.config.num_attention_heads * self.head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


class DomynEdgeDecoderLayer(torch.nn.Module):
    def __init__(self, config: DomynEdgeConfig, layer_idx: int):
        super().__init__()
        self.config = config

        self.attention = DomynEdgeAttention(config=config, layer_idx=layer_idx)
        self.feedforward = DomynEdgeFeedForward(config)

        self.pre_attention_norm = torch.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.pre_feedforward_norm = torch.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_norm = torch.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_feedforward_norm = torch.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_values: transformers.Cache | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        # Self Attention
        attn_output, _ = self.attention(
            hidden_states=self.pre_attention_norm(hidden_states),
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        hidden_states = hidden_states + self.post_attention_norm(attn_output)

        ## Fully Connected
        return hidden_states + self.post_feedforward_norm(self.feedforward(self.pre_feedforward_norm(hidden_states)))


class DomynEdgePreTrainedModel(transformers.PreTrainedModel):
    """Base class wiring DomynEdge into the HuggingFace `PreTrainedModel` machinery."""

    config: DomynEdgeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _no_split_modules = ["DomynEdgeDecoderLayer"]
    _skip_keys_device_placement = ["past_key_values"]
    _supports_flash_attn = True
    _supports_sdpa = True
    _supports_flex_attn = True

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DomynEdgeDecoderLayer,
        "attentions": DomynEdgeAttention,
    }


class DomynEdgeModel(DomynEdgePreTrainedModel):
    """DomynEdge decoder stack without the language modeling head."""

    def __init__(self, config: DomynEdgeConfig):
        """Build the embeddings, decoder layers, final norm and rotary embedding from `config`."""
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size

        self.embed_tokens = torch.nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = torch.nn.ModuleList(
            [DomynEdgeDecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        self.norm = torch.nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = DomynEdgeRotaryEmbedding(config=config)
        self.has_sliding_layers = "sliding_attention" in config.layer_types

        # Initialize weights and apply final processing
        self.post_init()

    @merge_with_config_defaults
    @capture_outputs
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | dict[str, torch.Tensor] | None = None,
        past_key_values: transformers.Cache | None = None,
        inputs_embeds: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> transformers.modeling_outputs.BaseModelOutputWithPast:
        """Run the decoder stack and return the final hidden states (and the cache when `use_cache` is set)."""
        if (input_ids is None) ^ (inputs_embeds is not None):
            msg = "You must specify exactly one of input_ids or inputs_embeds"
            raise ValueError(msg)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        assert inputs_embeds is not None

        if use_cache and past_key_values is None:
            past_key_values = transformers.DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        # It may already have been prepared by the caller, e.g. `generate`
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            mask_kwargs = {
                "config": self.config,
                "inputs_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {"full_attention": create_causal_mask(**mask_kwargs)}
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for i, decoder_layer in enumerate(self.layers):
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[self.config.layer_types[i]],
                position_embeddings=position_embeddings,
                position_ids=position_ids,
                past_key_values=past_key_values,
                **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return transformers.modeling_outputs.BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


class DomynEdgeForCausalLM(DomynEdgePreTrainedModel, transformers.GenerationMixin):
    """DomynEdge model with a language modeling head tied to the input embeddings."""

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config):
        """Build the decoder stack and the language modeling head from `config`."""
        super().__init__(config)
        self.model = DomynEdgeModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    @can_return_tuple
    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values: transformers.Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        **kwargs,
    ) -> transformers.modeling_outputs.CausalLMOutputWithPast:
        """Compute logits for the last `logits_to_keep` positions, plus the loss when `labels` are given."""
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            **kwargs,
        )

        hidden_states = outputs.last_hidden_state
        # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        loss = (
            self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)
            if labels is not None
            else None
        )

        return transformers.modeling_outputs.CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


__all__ = [
    "DomynEdgeForCausalLM",
    "DomynEdgeModel",
    "DomynEdgePreTrainedModel",
]
