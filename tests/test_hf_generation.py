# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Generation with the KV cache of the HuggingFace DomynEdge model matches generation without it."""

import pytest
import torch
import transformers
from test_hf_parity import VOCAB_SIZE, build_titan_model, convert_to_hf, load_hf_model

BATCH_SIZE = 2
PROMPT_LEN = 96
NEW_TOKENS = 64
RTOL = 1e-3
ATOL = 1e-2

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="varlen attention requires a CUDA device")


@pytest.fixture(scope="module")
def hf_model(tmp_path_factory) -> torch.nn.Module:
    """The converted debug model on the GPU, shared by the tests of this module."""
    titan_model = build_titan_model()
    dtype = next(titan_model.parameters()).dtype
    return load_hf_model(convert_to_hf(titan_model, tmp_path_factory.mktemp("hf")), dtype).cuda()


def random_tokens(batch_size: int, seq_len: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, VOCAB_SIZE, (batch_size, seq_len), generator=generator).cuda()


@torch.no_grad()
def test_generate_with_cache_matches_no_cache(hf_model):
    """Greedy generation through the KV cache picks the tokens a full uncached forward would pick."""
    prompt = random_tokens(BATCH_SIZE, PROMPT_LEN)
    # No padding: transformers >= 5.18 then drops the attention mask, so the model's default mask is used.
    out = hf_model.generate(
        prompt,
        generation_config=transformers.GenerationConfig(max_new_tokens=NEW_TOKENS, do_sample=False),
        use_cache=True,
        return_dict_in_generate=True,
        output_logits=True,
    )
    assert out.sequences.shape == (BATCH_SIZE, PROMPT_LEN + NEW_TOKENS)
    cached_logits = torch.stack(out.logits, dim=1)
    full_logits = hf_model(input_ids=out.sequences, use_cache=False).logits[:, PROMPT_LEN - 1 : -1]

    torch.testing.assert_close(cached_logits, full_logits, rtol=RTOL, atol=ATOL)

    generated = out.sequences[:, PROMPT_LEN:]
    torch.testing.assert_close(generated, full_logits.argmax(-1), rtol=0, atol=0)


@pytest.mark.parametrize("pass_mask", [False, True])
@torch.no_grad()
def test_chunked_prefill_matches_full_forward(hf_model, pass_mask):
    """Several new tokens on top of a non-empty cache give the logits of one full forward pass."""
    split = PROMPT_LEN // 2
    tokens = random_tokens(BATCH_SIZE, PROMPT_LEN + NEW_TOKENS)
    full_logits = hf_model(input_ids=tokens, use_cache=False).logits

    first = hf_model(input_ids=tokens[:, :split], use_cache=True)
    second = hf_model(
        input_ids=tokens[:, split:],
        past_key_values=first.past_key_values,
        attention_mask=torch.ones_like(tokens, dtype=torch.bool) if pass_mask else None,
        use_cache=True,
    )

    assert second.past_key_values.get_seq_length() == PROMPT_LEN + NEW_TOKENS
    torch.testing.assert_close(second.logits, full_logits[:, split:], rtol=RTOL, atol=ATOL)
