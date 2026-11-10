# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""The released SFT checkpoint, converted with the current HF modeling, served by HF transformers and vLLM."""

import json
import pathlib
import tempfile

import huggingface_hub
import pytest
import rich.console
import torch
import transformers
from hf_utils import ATTN_IMPLEMENTATIONS, HF_ASSETS_DIR, REPO_ROOT, load_convert_module, load_hf_model

REFERENCES = pathlib.Path(__file__).parent / "assets" / "domyn_edge_sft_completions.json"
# Questions with short answers, which greedy decoding must give exactly: free-form answers can be worded differently
# after a near tie between two implementations.
PROMPTS = [
    "Who are you? Answer with your name only.",
    "Come ti chiami? Rispondi solo con il tuo nome.",
    "What is the capital of France? Answer with one word.",
    "How many legs does a spider have? Answer with the number only.",
    "What is 17 * 23? Answer with the number only.",
    # Longer than the sliding window, so the windowed layers drop the start of the prompt that the global ones see.
    "Here is a list of item codes.\n"
    + "\n".join(f"Item {i}: code {(i * 7919) % 10000:04d}." for i in range(1, 201))
    + "\nWhat is the code of item 187? Answer with the code only.",
]
MAX_TOKENS = 1024
MAX_LOGPROB_DRIFT = {
    "eager": 0.25,
    "sdpa": 0.25,
    "flex_attention": 0.25,
    "torch_varlen": 0.0,
    "vLLM transformers": 0.25,
    "vLLM native": 0.25,
}

pytestmark = [
    pytest.mark.skipif(not torch.cuda.is_available(), reason="the deployed checkpoint is tested on a CUDA device"),
    pytest.mark.skipif(not REFERENCES.exists(), reason=f"no recorded completions in {REFERENCES}"),
]


def _build_sft_checkpoint(
    output_dir: pathlib.Path,
    revision: str | None = None,
    repo: str = "domyn/Domyn-Edge-SFT",
    flavor: str = "edgethinker",
    tokenizer_dir: pathlib.Path = REPO_ROOT / "assets" / "domyn-edge" / "tokenizers" / "posttraining",
) -> tuple[pathlib.Path, str]:
    """Write the SFT checkpoint as `convert.py` would today, with the released weights linked in.

    The released config and modeling predate the current conversion, so only the weights are taken from the hub `repo`,
    at `revision` (the latest cached one if None). `flavor` is the torchtitan flavor the checkpoint was trained with,
    which `convert.py` builds the config from. Returns the checkpoint directory and the hub commit hash it was built
    from.
    """
    snapshot = pathlib.Path(
        huggingface_hub.snapshot_download(
            repo,
            revision=revision,
            allow_patterns=["config.json", "*.safetensors", "*.index.json"],
            local_files_only=True,
        )
    )
    hub_config = json.loads((snapshot / "config.json").read_text())

    convert = load_convert_module()
    console = rich.console.Console(quiet=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    convert.save_tokenizer(console, tokenizer_dir, output_dir)
    convert.save_modeling(console, output_dir, assets_dir=HF_ASSETS_DIR)
    convert.save_config(
        console,
        tokenizer_dir,
        output_dir,
        "domynedge",
        flavor,
        "bfloat16",
        hub_config["max_position_embeddings"],
        hub_config["vocab_size"],
    )
    for weights in [*snapshot.glob("*.safetensors"), *snapshot.glob("*.index.json")]:
        (output_dir / weights.name).symlink_to(weights)

    config = json.loads((output_dir / "config.json").read_text())
    assert config["layer_types"] == [
        "full_attention" if left == -1 else "sliding_attention" for left, _ in hub_config["window_sizes"]
    ]
    assert config["head_sizes"] == hub_config["head_sizes"]
    assert config["position_layers"] == hub_config["position_layers"]

    return output_dir, snapshot.name


def _prompt_token_ids(tokenizer, prompt: str) -> list[int]:
    """A single user turn, formatted with the chat template for the assistant's answer."""
    messages = [{"role": "user", "content": prompt}]
    return list(tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=True, return_dict=False))


def _checked_references(tokenizer, checkpoint_dir: pathlib.Path) -> dict:
    """The recorded references, after checking they still apply to the checkpoint and cover the sliding window."""
    references = json.loads(REFERENCES.read_text())
    for reference in references["completions"]:
        assert _prompt_token_ids(tokenizer, reference["prompt"]) == reference["prompt_token_ids"], (
            f"the tokenizer or chat template changed since the references were recorded: {reference['prompt']!r}"
        )
    # At least one prompt must reach past the sliding window, for the windowed layers to matter.
    sliding_window = json.loads((checkpoint_dir / "config.json").read_text())["sliding_window"]
    longest_prompt = max(len(reference["prompt_token_ids"]) for reference in references["completions"])
    assert longest_prompt > sliding_window, f"the longest prompt ({longest_prompt} tokens) fits in the window"
    return references


def _assert_teacher_forcing(
    name: str, reference: dict, logprobs: list[dict[int, float]], tokenizer, top_k: int = 5
) -> None:
    """Check an implementation's next-token log-probabilities on the reference prefix."""
    reference_tokens = reference["completion_token_ids"]
    for offset, (reference_token, reference_top, implementation_logprobs) in enumerate(
        zip(reference_tokens, reference["completion_top_logprobs"], logprobs, strict=True)
    ):
        implementation_token = max(implementation_logprobs, key=implementation_logprobs.__getitem__)
        reference_top_tokens = [token for token, _ in reference_top]

        logprob_of_reference_token = implementation_logprobs[reference_token]
        impl_rank = 1 + sum(logprob > logprob_of_reference_token for logprob in implementation_logprobs.values())

        recorded_logprob = dict(map(tuple, reference_top)).get(reference_token)
        if recorded_logprob is not None:
            drift = abs(logprob_of_reference_token - recorded_logprob)
            assert drift <= MAX_LOGPROB_DRIFT[name], (
                f"{name}, {reference['prompt'][:60]!r}, token {offset}: log-probability {drift:.3g} from recorded"
            )

        reference_text = tokenizer.decode([reference_token])
        implementation_text = tokenizer.decode([implementation_token])
        failure = None
        if impl_rank > top_k:
            failure = f"reference {reference_text!r} is outside the {name} top {top_k} (rank {impl_rank})"
        if implementation_token not in reference_top_tokens:
            failure = f"{name} top {implementation_text!r} is outside the reference top {top_k}"
        if failure is not None:
            preceding = tokenizer.decode(reference_tokens[:offset])[-80:]
            pytest.fail(f"{name}, {reference['prompt'][:60]!r}, token {offset} after {preceding!r}: {failure}")


def _reference_answer(reference: dict) -> str:
    """The answer of a reference completion, after its reasoning."""
    assert "</think>" in reference["completion"], (
        f"{reference['prompt'][:60]!r}: the reference reasoning did not finish"
    )
    return reference["completion"].split("</think>")[-1].strip()


def _assert_answer(name: str, reference: dict, expected: str, text: str) -> None:
    """A greedy completion must finish the reasoning and give the reference answer."""
    assert "</think>" in text, f"{name}, {reference['prompt'][:60]!r}: the reasoning did not finish: {text[-200:]!r}"
    answer = text.split("</think>")[-1].strip()
    assert answer == expected, f"{name}, {reference['prompt'][:60]!r}: expected the answer {expected!r}, got {answer!r}"


@pytest.fixture(scope="module")
def sft_checkpoint_dir(tmp_path_factory) -> pathlib.Path:
    """The SFT checkpoint the references were recorded with: the same hub repo, at the same revision."""
    references = json.loads(REFERENCES.read_text())
    repo, revision = references["repo"], references["revision"]
    try:
        checkpoint_dir, _ = _build_sft_checkpoint(tmp_path_factory.mktemp("sft"), revision, repo)
    except (huggingface_hub.errors.LocalEntryNotFoundError, OSError) as error:
        pytest.skip(f"{repo} at revision {revision} is not available: {error}")
    return checkpoint_dir


@pytest.mark.parametrize("top_k", [5])
@pytest.mark.parametrize("attn_implementation", ATTN_IMPLEMENTATIONS)
@torch.no_grad()
def test_hf_sft_completions_match_references(sft_checkpoint_dir, attn_implementation, top_k):
    """HF transformers, with every attention implementation, has the predictions of the recorded completions.

    Teacher forcing with one uncached forward pass over each prompt and reference completion, against the reference
    log-probabilities `torch_varlen` recorded with the same pass. Then greedy decoding must give the reference answers.
    """
    tokenizer = transformers.AutoTokenizer.from_pretrained(sft_checkpoint_dir)
    references = _checked_references(tokenizer, sft_checkpoint_dir)
    model = load_hf_model(sft_checkpoint_dir, attn_implementation)

    for reference in references["completions"]:
        completion = reference["completion_token_ids"]
        sequence = torch.tensor([reference["prompt_token_ids"] + completion], device="cuda")
        # The logits before each completion token, the last one predicting past the end.
        logits = model(input_ids=sequence, use_cache=False, logits_to_keep=len(completion) + 1).logits
        logprobs = logits[0, :-1].float().log_softmax(-1)
        # This implementation's top candidates and its log-probability of the reference token at every position, as
        # vLLM returns them.
        candidate_tokens = torch.cat(
            [logprobs.topk(top_k).indices, torch.tensor(completion, device="cuda")[:, None]], dim=1
        )
        candidate_logprobs = logprobs.gather(1, candidate_tokens)
        candidates = [
            dict(zip(tokens, values, strict=True))
            for tokens, values in zip(candidate_tokens.tolist(), candidate_logprobs.tolist(), strict=True)
        ]
        _assert_teacher_forcing(attn_implementation, reference, candidates, tokenizer, top_k)

    for reference in references["completions"]:
        prompt = torch.tensor([reference["prompt_token_ids"]], device="cuda")
        # `load_hf_model` types the model as a plain `Module`, without the `GenerationMixin` it also is.
        output = model.generate(prompt, max_new_tokens=references["max_tokens"], do_sample=False)  # pyright: ignore[reportCallIssue]
        text = tokenizer.decode(output[0, prompt.shape[1] :].tolist(), skip_special_tokens=True)
        _assert_answer(attn_implementation, reference, _reference_answer(reference), text)


@pytest.mark.parametrize("top_k", [5])
@pytest.mark.parametrize("model_impl", ["transformers", "vllm"])
def test_vllm_sft_completions_match_references(sft_checkpoint_dir, monkeypatch, model_impl, top_k):
    """vLLM serves the converted SFT checkpoint with the predictions of the recorded HF completions.

    Teacher forcing: given each reference prefix, vLLM's next-token prediction is compared with the recorded one. Then
    vLLM's own greedy completions must give the reference answers.
    """
    vllm = pytest.importorskip("vllm")
    # CUDA is already initialized in this process, and a forked engine process could not initialize it again.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

    tokenizer = transformers.AutoTokenizer.from_pretrained(sft_checkpoint_dir)
    references = _checked_references(tokenizer, sft_checkpoint_dir)
    llm = vllm.LLM(
        model=str(sft_checkpoint_dir),
        trust_remote_code=True,
        dtype="bfloat16",
        model_impl=model_impl,
        max_model_len=4096,
        enforce_eager=True,
        seed=0,
        # Every position computed from scratch, for the prompt log-probabilities.
        enable_prefix_caching=False,
    )
    # As vLLM's own tests check: the Transformers backend runs our HF modeling, the native implementation does not.
    assert llm.llm_engine.model_config.using_transformers_backend() == (model_impl == "transformers")
    name = "vLLM transformers" if model_impl == "transformers" else "vLLM native"

    # Teacher forcing: the prompt and the reference completion as one prompt, with vLLM's log-probabilities of the
    # next token at every position.
    forced = llm.generate(
        [
            {"prompt_token_ids": reference["prompt_token_ids"] + reference["completion_token_ids"]}
            for reference in references["completions"]
        ],
        vllm.SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=top_k),
    )
    for reference, output in zip(references["completions"], forced, strict=True):
        # Each entry holds vLLM's top candidates, plus the actual token at that position.
        start = len(reference["prompt_token_ids"])
        candidates = [
            {candidate: logprob.logprob for candidate, logprob in output.prompt_logprobs[start + offset].items()}
            for offset in range(len(reference["completion_token_ids"]))
        ]
        _assert_teacher_forcing(name, reference, candidates, tokenizer, top_k)

    # vLLM's own greedy completions must give the reference answers.
    generated = llm.generate(
        [{"prompt_token_ids": reference["prompt_token_ids"]} for reference in references["completions"]],
        vllm.SamplingParams(temperature=0.0, max_tokens=references["max_tokens"]),
    )
    for reference, output in zip(references["completions"], generated, strict=True):
        _assert_answer(name, reference, _reference_answer(reference), output.outputs[0].text)


@torch.no_grad()
def _record_references(top_k: int = 5, repo: str = "domyn/Domyn-Edge-SFT") -> None:
    """Record the greedy completions of the SFT checkpoint `repo` with HF transformers and `torch_varlen`.

    Along with each completion token, its `top_k` most likely tokens are recorded. They come from one uncached pass over
    the prompt and the completion, the computation teacher forcing repeats, not from the cached decoding steps, which
    round differently in bfloat16.
    """
    with tempfile.TemporaryDirectory() as workdir:
        checkpoint_dir, revision = _build_sft_checkpoint(pathlib.Path(workdir) / "sft", repo=repo)
        tokenizer = transformers.AutoTokenizer.from_pretrained(checkpoint_dir)
        model = load_hf_model(checkpoint_dir, "torch_varlen")
        completions = []
        for prompt in PROMPTS:
            prompt_token_ids = _prompt_token_ids(tokenizer, prompt)
            # `load_hf_model` types the model as a plain `Module`, without the `GenerationMixin` it also is.
            output = model.generate(  # pyright: ignore[reportCallIssue]
                torch.tensor([prompt_token_ids], device="cuda"),
                max_new_tokens=MAX_TOKENS,
                do_sample=False,
            )
            completion_token_ids = output[0, len(prompt_token_ids) :].tolist()
            # HF's top candidates at every completion position, as (token, log-probability) pairs, from the uncached
            # pass `test_hf_sft_completions_match_references` makes.
            sequence = torch.tensor([prompt_token_ids + completion_token_ids], device="cuda")
            logits = model(input_ids=sequence, use_cache=False, logits_to_keep=len(completion_token_ids) + 1).logits
            values, tokens = logits[0, :-1].float().log_softmax(-1).topk(top_k)
            top_logprobs = [
                [[token, value] for token, value in zip(position_tokens, position_values, strict=True)]
                for position_tokens, position_values in zip(tokens.tolist(), values.tolist(), strict=True)
            ]
            completions.append(
                {
                    "prompt": prompt,
                    "prompt_token_ids": prompt_token_ids,
                    "completion_token_ids": completion_token_ids,
                    "completion": tokenizer.decode(completion_token_ids, skip_special_tokens=True),
                    "completion_top_logprobs": top_logprobs,
                }
            )
            print(f"{prompt[:60]!r} ({len(prompt_token_ids)} tokens) -> {completions[-1]['completion']!r}")
    REFERENCES.parent.mkdir(exist_ok=True)
    REFERENCES.write_text(
        json.dumps({"repo": repo, "revision": revision, "max_tokens": MAX_TOKENS, "completions": completions}, indent=2)
        + "\n"
    )
    print(f"wrote {REFERENCES} for {repo} at revision {revision}")


if __name__ == "__main__":
    _record_references()
