# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0
#
# This file includes third-party code:
#
# - `IndexFirstAxis`, `IndexPutFirstAxis`, `unpad_input` and `pad_input` are adapted from
#   https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/bert_padding.py
#   and are distributed under the following license:
#
#   BSD 3-Clause License
#
#   Copyright (c) 2022, the respective contributors, as shown by the AUTHORS file.
#   All rights reserved.
#
#   Redistribution and use in source and binary forms, with or without
#   modification, are permitted provided that the following conditions are met:
#
#   * Redistributions of source code must retain the above copyright notice, this
#     list of conditions and the following disclaimer.
#
#   * Redistributions in binary form must reproduce the above copyright notice,
#     this list of conditions and the following disclaimer in the documentation
#     and/or other materials provided with the distribution.
#
#   * Neither the name of the copyright holder nor the names of its
#     contributors may be used to endorse or promote products derived from
#     this software without specific prior written permission.
#
#   THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
#   AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
#   IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
#   DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
#   FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
#   DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
#   SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
#   CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
#   OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
#   OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# - `DomynEdgeRotaryEmbedding.compute_default_rope_parameters`, `rotate_half` and
#   `apply_rotary_pos_emb` are adapted from HuggingFace transformers
#   (https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py),
#   Copyright The HuggingFace Team, licensed under the Apache License, Version 2.0.
#   Modified for DomynEdge.

import dataclasses
import typing

import einops
import torch
import torch.nn.attention.varlen
import transformers
import transformers.modeling_outputs

from .configuration_domynedge import DomynEdgeConfig


@dataclasses.dataclass
class VarlenMetadataMask:
    cu_seq_q: torch.Tensor
    cu_seq_k: torch.Tensor
    max_q: int
    max_k: int


class IndexFirstAxis(torch.autograd.Function):
    """Gather the rows of the first axis at `indices`, scattering the gradient back in backward."""

    @staticmethod
    def forward(ctx, input_, indices):
        ctx.save_for_backward(indices)
        assert input_.ndim >= 2
        ctx.first_axis_dim, other_shape = input_.shape[0], input_.shape[1:]
        second_dim = other_shape.numel()
        return torch.gather(
            einops.rearrange(input_, "b ... -> b (...)"), 0, einops.repeat(indices, "z -> z d", d=second_dim)
        ).reshape(-1, *other_shape)

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        assert grad_output.ndim >= 2
        other_shape = grad_output.shape[1:]
        grad_output = einops.rearrange(grad_output, "b ... -> b (...)")
        grad_input = torch.zeros(
            [ctx.first_axis_dim, grad_output.shape[1]],
            device=grad_output.device,
            dtype=grad_output.dtype,
        )
        grad_input.scatter_(0, einops.repeat(indices, "z -> z d", d=grad_output.shape[1]), grad_output)
        return grad_input.reshape(ctx.first_axis_dim, *other_shape), None


class IndexPutFirstAxis(torch.autograd.Function):
    """Scatter `values` into a zero tensor at the first-axis `indices`, gathering the gradient back in backward."""

    @staticmethod
    def forward(ctx, values, indices, first_axis_dim):
        ctx.save_for_backward(indices)
        assert indices.ndim == 1
        assert values.ndim >= 2
        output = torch.zeros(first_axis_dim, *values.shape[1:], device=values.device, dtype=values.dtype)
        output[indices] = values
        return output

    @staticmethod
    def backward(ctx, grad_output):
        (indices,) = ctx.saved_tensors
        grad_values = grad_output[indices]
        return grad_values, None, None


def unpad_input(
    hidden_states: torch.Tensor, attention_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Remove the padding tokens, packing the valid tokens of all sequences along one axis.

    Arguments:
        hidden_states: (batch, seqlen, ...)
        attention_mask: (batch, seqlen), bool / int, 1 means valid and 0 means not valid.

    Return:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens selected in attention_mask + unused_mask.
        indices: (total_nnz), the indices of masked tokens from the flattened input sequence.
        cu_seqlens: (batch + 1), the cumulative sequence lengths, used to index into hidden_states.
        max_seqlen_in_batch: int

    """
    seqlens_in_batch = attention_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(attention_mask.flatten(), as_tuple=False).flatten()
    max_seqlen = seqlens_in_batch.max().item()
    cu_seqlens = torch.nn.functional.pad(torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0))
    return (
        IndexFirstAxis.apply(einops.rearrange(hidden_states, "b s ... -> (b s) ..."), indices),
        indices,
        cu_seqlens,
        max_seqlen,
    )


def pad_input(hidden_states: torch.Tensor, indices: torch.Tensor, batch_size: int, seqlen: int) -> torch.Tensor:
    """Restore the padded (batch, seqlen, ...) layout from packed tokens, the inverse of `unpad_input`.

    Arguments:
        hidden_states: (total_nnz, ...), where total_nnz = number of tokens in selected in attention_mask.
        indices: (total_nnz), the indices that represent the non-masked tokens of the original padded input sequence.
        batch_size: int, batch size for the padded sequence.
        seqlen: int, maximum sequence length for the padded sequence.

    Return:
        hidden_states: (batch, seqlen, ...)

    """
    output = IndexPutFirstAxis.apply(hidden_states, indices, batch_size * seqlen)
    return einops.rearrange(output, "(b s) ... -> b s ...", b=batch_size)


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

        self.rope_type = self.config.rope_parameters["rope_type"]
        rope_init_fn: typing.Callable = self.compute_default_rope_parameters
        inv_freq, self.attention_scaling = rope_init_fn(self.config, device)

        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("original_inv_freq", inv_freq.clone(), persistent=False)

    @staticmethod
    def compute_default_rope_parameters(
        config: DomynEdgeConfig | None = None,
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
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        # TODO: the angles lose precision in two cases, both wrong at large positions:
        # - under torch.autocast(bfloat16) the matmul below runs in bf16 despite the `.float()` casts;
        # - after model.to(torch.bfloat16) / model.bfloat16() the `inv_freq` buffer itself is cast to bf16.
        # Loading with `from_pretrained(dtype=torch.bfloat16)` and no autocast is not affected. Candidate fix:
        # recompute `inv_freq` in fp32 here, with autocast disabled, instead of reading the buffer.
        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos, sin


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.

    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    qf, kf = q.float(), k.float()
    q_embed = (qf * cos) + (rotate_half(qf) * sin)
    k_embed = (kf * cos) + (rotate_half(kf) * sin)
    return q_embed.to(q.dtype), k_embed.to(k.dtype)


class DomynEdgeAttention(torch.nn.Module):
    """Grouped Query Attention with support for sliding window."""

    def __init__(self, config: DomynEdgeConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.head_dim = config.head_sizes[layer_idx]
        self.scaling = self.head_dim**-0.5
        self.window_size = config.window_sizes[layer_idx]
        self.apply_rope = (
            apply_rotary_pos_emb
            if config.position_layers[layer_idx] == "rope"
            else lambda q, k, cos, sin, unsqueeze_dim: (q, k)  # noqa: ARG005
        )
        self.layer_idx = layer_idx
        self.enable_gqa = config.num_attention_heads > config.num_key_value_heads

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
        attention_mask: torch.Tensor,
        past_key_values: transformers.Cache | None = None,
        **_kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:

        batch_size, sequence_length, _ = hidden_states.shape

        q = self.q_norm(
            self.q_proj(hidden_states).view(batch_size, sequence_length, self.config.num_attention_heads, self.head_dim)
        )
        k = self.k_norm(
            self.k_proj(hidden_states).view(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)
        )
        v = self.v_proj(hidden_states).view(batch_size, sequence_length, self.config.num_key_value_heads, self.head_dim)

        cos, sin = position_embeddings
        q, k = self.apply_rope(q, k, cos, sin, unsqueeze_dim=2)

        if past_key_values is not None:
            k, v = k.transpose(1, 2), v.transpose(1, 2)
            k, v = past_key_values.update(k, v, self.layer_idx)
            k, v = k.transpose(1, 2), v.transpose(1, 2)

        if q.size(1) > 1:
            q, q_indices, q_cu_seqlen, q_maxseq = unpad_input(q, attention_mask)
        else:
            q_maxseq = 1
            q_cu_seqlen = torch.arange(batch_size + 1, dtype=torch.int32, device=q.device)
            q_indices = q_cu_seqlen[:-1]
            q = q.squeeze(1)

        k, _, k_cu_seqlen, k_maxseq = unpad_input(k, attention_mask)
        v, *_ = unpad_input(v, attention_mask)

        attn_output = torch.nn.attention.varlen.varlen_attn(
            q if q.dtype in [torch.float16, torch.bfloat16] else q.to(torch.bfloat16),
            k if k.dtype in [torch.float16, torch.bfloat16] else k.to(torch.bfloat16),
            v if v.dtype in [torch.float16, torch.bfloat16] else v.to(torch.bfloat16),
            cu_seq_q=q_cu_seqlen,
            cu_seq_k=k_cu_seqlen,
            max_q=q_maxseq,
            max_k=k_maxseq,
            scale=self.scaling,
            window_size=self.window_size,
            enable_gqa=self.enable_gqa,
        ).to(q.dtype)

        attn_output = pad_input(attn_output, q_indices, batch_size, sequence_length)
        attn_output = attn_output.view(batch_size, sequence_length, self.config.num_attention_heads * self.head_dim)
        return self.o_proj(attn_output)


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
        hidden_states = hidden_states + self.post_attention_norm(
            self.attention(
                hidden_states=self.pre_attention_norm(hidden_states),
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                position_embeddings=position_embeddings,
                **kwargs,
            )
        )

        ## Fully Connected
        return hidden_states + self.post_feedforward_norm(self.feedforward(self.pre_feedforward_norm(hidden_states)))


class DomynEdgePreTrainedModel(transformers.PreTrainedModel):
    """Base class wiring DomynEdge into the HuggingFace `PreTrainedModel` machinery."""

    config: DomynEdgeConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = False
    _supports_flash_attn = True
    _supports_sdpa = False
    _supports_flex_attn = False

    _can_compile_fullgraph = True
    _supports_attention_backend = True
    _can_record_outputs = {
        "hidden_states": DomynEdgeDecoderLayer,
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

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor | None = None,
        attention_mask: torch.Tensor | None = None,
        past_key_values: transformers.Cache | None = None,
        inputs_embeds: torch.FloatTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        use_cache: bool | None = None,
        **kwargs,
    ) -> transformers.modeling_outputs.BaseModelOutputWithPast:
        """Run the decoder stack and return the final hidden states (and the cache when `use_cache` is set)."""
        if input_ids is not None and inputs_embeds is not None:
            msg = "You must specify exactly one of input_ids or inputs_embeds"
            raise ValueError(msg)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            past_key_values = transformers.DynamicCache(config=self.config)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        if attention_mask is None:
            attention_mask = torch.ones(hidden_states.shape[:2], device=hidden_states.device, dtype=torch.bool)

        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_embeddings=position_embeddings,
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
