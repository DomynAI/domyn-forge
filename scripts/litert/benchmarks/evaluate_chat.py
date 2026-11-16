#!/usr/bin/env python3
"""Evaluate quick-chat generation on fixed IFEval or GSM8K shards."""

import argparse
import hashlib
import json
import random
import re
from contextlib import contextmanager
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

NUMERIC_ANSWER = re.compile(r"(?m)^\s*####[ \t]+(-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?)[ \t]*\.?[ \t]*$")
NUMERIC_VALUE = re.compile(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")


def make_evaluation(task, seed, gsm8k_prompt="solve"):
    """Return the prompt renderer and scorer for the selected task."""
    if task == "gsm8k":
        from decimal import Decimal

        def prompt(row):
            instruction = (
                "Give only the final numerical answer on a line starting with ####."
                if gsm8k_prompt == "final-only"
                else "Solve the problem. End with a line starting with #### followed by the final numerical answer."
            )
            return row["question"] + "\n" + instruction

        def score(row, response):
            expected = NUMERIC_ANSWER.findall(row["answer"])
            if len(expected) != 1:
                msg = f"Unexpected GSM8K reference format for row {row['key']}"
                raise ValueError(msg)
            answers = NUMERIC_ANSWER.findall(response)
            prediction = answers[0].replace(",", "") if len(answers) == 1 else None
            target = Decimal(expected[0].replace(",", ""))
            numeric = NUMERIC_VALUE.findall(response)
            return {
                "correct": prediction is not None and Decimal(prediction) == target,
                "prediction": prediction,
                "valid_format": prediction is not None,
                "last_number_correct": bool(numeric) and Decimal(numeric[-1].replace(",", "")) == target,
            }

        return prompt, score

    import langdetect
    from lm_eval.tasks.ifeval.utils import InputExample, test_instruction_following_strict

    langdetect.DetectorFactory.seed = seed

    def score(row, response):
        result = test_instruction_following_strict(InputExample(**row), response)
        return {"correct": result.follow_all_instructions, "instructions": result.follow_instruction_list}

    return lambda row: row["prompt"], score


def stream_visible(chat, prompt, max_output_tokens):
    """Send one message and return the visible reply, seconds to its first text, and total seconds."""
    start = perf_counter()
    chunks = []
    first_visible = None
    for message in chat.send_message_async(prompt, max_output_tokens=max_output_tokens):
        visible = "".join(item.text for item in message.contents if hasattr(item, "text"))
        if visible:
            if first_visible is None:
                first_visible = perf_counter() - start
            chunks.append(visible)
    return "".join(chunks), first_visible, perf_counter() - start


@contextmanager
def open_litert(args):
    """Run the embedded chat template and collect streamed visible output."""
    import litert_lm as lm

    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with lm.Engine(
        str(args.model.absolute()),
        backend=lm.Backend.CPU(thread_count=args.threads),
        max_num_tokens=args.context,
        cache_dir=str(args.cache_dir.resolve()),
        enable_benchmark=True,
    ) as engine:

        def generate(prompt):
            with engine.create_conversation(
                sampler_config=lm.SamplerConfig(top_k=1, seed=args.seed),
                extra_context={"enable_thinking": False},
            ) as chat:
                response, first_visible, duration = stream_visible(chat, prompt, args.output_tokens)
                info = chat.get_benchmark_info()
            return {
                "response": response,
                "prefill_tokens": info.last_prefill_token_count,
                "decode_tokens": info.last_decode_token_count,
                "first_visible_seconds": first_visible,
                "seconds": duration,
                "hit_output_cap": info.last_decode_token_count >= args.output_tokens,
            }

        yield generate


@contextmanager
def open_source(args):
    """Generate greedily with the original BF16 weights and the Domyn quick-chat template."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig

    torch.set_num_threads(args.threads)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=False)
    template = (Path(__file__).parents[1] / "export" / "quick_chat.jinja").read_text()
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        device_map={"": args.source_device},
    ).eval()
    settings = GenerationConfig(
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        max_new_tokens=args.output_tokens,
    )

    def generate(prompt):
        start = perf_counter()
        rendered = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], chat_template=template, tokenize=False, add_generation_prompt=True
        )
        ids = tokenizer.encode((tokenizer.bos_token or "") + rendered, add_special_tokens=False)
        if len(ids) + args.output_tokens > args.context:
            msg = f"Prompt plus output budget exceeds {args.context} tokens"
            raise ValueError(msg)
        inputs = torch.tensor([ids], device=args.source_device)
        with torch.inference_mode():
            output = model.generate(inputs, generation_config=settings, attention_mask=torch.ones_like(inputs))
        tokens = output[0, len(ids) :].tolist()
        response = tokenizer.decode(tokens, skip_special_tokens=True)
        return {
            "response": response,
            "prefill_tokens": len(ids),
            "decode_tokens": len(tokens),
            "seconds": perf_counter() - start,
            "hit_output_cap": len(tokens) == args.output_tokens and tokens[-1] != tokenizer.eos_token_id,
        }

    yield generate


def write_record(output, **record):
    """Write and flush one record so completed examples survive an interruption."""
    print(json.dumps(record, ensure_ascii=False, allow_nan=False), file=output, flush=True)


def select_rows(data, task, seed):
    """Give shards and full runs the same deterministic example order."""
    rows = [json.loads(line) for line in data.splitlines()]
    if task == "gsm8k":
        rows = [{"key": i, **row} for i, row in enumerate(rows)]
    rows.sort(key=lambda row: row["key"])
    if len({row["key"] for row in rows}) != len(rows):
        msg = "Dataset keys must be unique"
        raise ValueError(msg)
    random.Random(seed).shuffle(rows)
    return rows


def main():
    """Run one auditable shard, flushing every completed example."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=("ifeval", "gsm8k"))
    parser.add_argument("model", type=Path)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--start", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--output-tokens", type=int, required=True)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--source-device", choices=("cpu", "mps"))
    parser.add_argument("--gsm8k-prompt", choices=("solve", "final-only"), default="solve")
    args = parser.parse_args()
    if args.source_device is None and args.cache_dir is None:
        parser.error("--cache-dir is required for LiteRT")
    if not 0 < args.output_tokens < args.context or args.threads < 1:
        parser.error("Invalid token budget or thread count")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    data = args.dataset.read_bytes()
    rows = select_rows(data, args.task, args.seed)
    end = len(rows) if args.limit is None else args.start + args.limit
    if not 0 <= args.start < end <= len(rows):
        parser.error("Invalid shard range")
    run(args, rows[args.start : end], hashlib.sha256(data).hexdigest())


def run(args, rows, dataset_sha256):
    """Write provenance and each scored prediction to a fresh JSONL file."""
    prompt, score = make_evaluation(args.task, args.seed, args.gsm8k_prompt)
    backend = open_source if args.source_device else open_litert
    model_files = sorted(args.model.glob("*.safetensors")) if args.source_device else [args.model]
    if not model_files:
        msg = "Expected a local source checkpoint with safetensors shards"
        raise ValueError(msg)
    fingerprints = {}
    for path in model_files:
        with path.open("rb") as model_file:
            fingerprints[path.name] = hashlib.file_digest(model_file, "sha256").hexdigest()
    with args.output.open("x", encoding="utf-8") as output:
        write_record(
            output,
            kind="config",
            task=args.task,
            model=str(args.model.absolute()),
            model_sha256=fingerprints,
            dataset_sha256=dataset_sha256,
            runtime=version("torch" if args.source_device else "litert-lm"),
            transformers=version("transformers") if args.source_device else None,
            scorer=version("lm-eval") if args.task == "ifeval" else "gsm8k-final-marker",
            gsm8k_prompt=args.gsm8k_prompt if args.task == "gsm8k" else None,
            backend=args.source_device or "litert-cpu",
            threads=args.threads,
            context=args.context,
            output_tokens=args.output_tokens,
            seed=args.seed,
            thinking=False,
            keys=[row["key"] for row in rows],
        )
        correct = capped = formatted = last_number_correct = 0
        with backend(args) as generate:
            for i, row in enumerate(rows, start=args.start):
                result = generate(prompt(row))
                scores = score(row, result["response"])
                correct += scores["correct"]
                capped += result["hit_output_cap"]
                formatted += scores.get("valid_format", False)
                last_number_correct += scores.get("last_number_correct", False) and not result["hit_output_cap"]
                write_record(output, kind="example", index=i, key=row["key"], **result, **scores)
        summary = {"examples": len(rows), "correct": correct, "hit_output_cap": capped}
        if args.task == "gsm8k":
            summary.update(valid_format=formatted, last_number_correct_uncapped=last_number_correct)
        write_record(output, kind="summary", **summary)
        print(f"{args.task}: {correct}/{len(rows)} correct, {capped} reached the output cap")


if __name__ == "__main__":
    main()
