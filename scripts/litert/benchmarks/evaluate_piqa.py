#!/usr/bin/env python3
"""Score the same zero-shot PIQA examples with BF16 and LiteRT-LM on a desktop."""

import argparse
import hashlib
import json
import math
import os
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from scripts.litert.checks.compare_quality import score_litert, score_source


def prepare_examples(dataset, tokenizer, limit, context, start=0):
    """Build a shared prompt and tokenized continuation for each PIQA choice."""
    rows = dataset.decode("utf-8").splitlines()
    if start < 0 or limit < 1 or start + limit > len(rows):
        msg = f"Requested rows fall outside the {len(rows)} rows in the dataset"
        raise ValueError(msg)
    examples = []
    cases = []
    for index, line in enumerate(rows[start : start + limit], start=start):
        row = json.loads(line)
        choices = [row["sol1"], row["sol2"]]
        if row["label"] not in (0, 1) or any(not choice.strip() for choice in choices):
            msg = f"Invalid PIQA example at validation row {index}"
            raise ValueError(msg)
        prefix = (tokenizer.bos_token or "") + f"Question: {row['goal']}\nAnswer:"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        examples.append({"row": index, "label": row["label"], "choice_characters": list(map(len, choices))})
        for choice_index, choice in enumerate(choices):
            continuation = " " + choice
            target_ids = tokenizer.encode(continuation, add_special_tokens=False)
            if not prefix_ids or not target_ids or len(prefix_ids) + len(target_ids) > context:
                msg = f"Invalid token count for validation row {index}, choice {choice_index}"
                raise ValueError(msg)
            cases.append((f"{index}/{choice_index}", prefix, continuation, prefix_ids, target_ids))
    return examples, cases


def summarize(examples, scores, elapsed):
    """Calculate raw and character-normalized accuracy from continuation losses."""
    if len(scores) != 2 * len(examples) or any(not math.isfinite(loss) for loss in scores):
        msg = "Expected two finite continuation losses per example"
        raise ValueError(msg)
    predictions = []
    for index, row in enumerate(examples):
        first, second = scores[2 * index : 2 * index + 2]
        lengths = row["choice_characters"]
        predictions.append(
            {
                **row,
                "choice_nll": [first, second],
                "prediction": int(second < first),
                "prediction_normalized": int(second / lengths[1] < first / lengths[0]),
            }
        )
    return {
        "correct": sum(row["prediction"] == row["label"] for row in predictions),
        "correct_normalized": sum(row["prediction_normalized"] == row["label"] for row in predictions),
        "seconds_including_load": round(elapsed, 2),
        "predictions": predictions,
    }


def main():
    """Run the requested models sequentially and save their per-example scores."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Canonical checkpoint, also used for its tokenizer")
    parser.add_argument("dataset", type=Path, help="PIQA validation JSONL with goal, sol1, sol2 and label")
    parser.add_argument("--source", action="store_true", help="Score the BF16 checkpoint on CPU")
    parser.add_argument("--bundle", action="append", nargs=3, default=[], metavar=("NAME", "FILE", "CACHE_DIR"))
    parser.add_argument("--limit", type=int, default=50, help="Number of validation rows to score from --start")
    parser.add_argument("--start", type=int, default=0, help="Zero-based first validation row")
    parser.add_argument("--context", type=int, default=512, help="LiteRT-LM token capacity for scoring")
    parser.add_argument("--output", required=True, type=Path, help="New results JSON file")
    args = parser.parse_args()

    names = (["bf16"] if args.source else []) + [bundle[0] for bundle in args.bundle]
    if not names or len(names) != len(set(names)):
        parser.error("Select one or more models with unique names")
    if args.context < 1 or args.limit < 1 or args.start < 0:
        parser.error("Context and limit must be positive and the start row must be nonnegative")
    if not args.output.parent.is_dir():
        parser.error(f"Output directory does not exist: {args.output.parent}")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    checkpoint = args.checkpoint.resolve(strict=True)
    bundles = [(name, Path(path).resolve(strict=True), Path(cache).resolve()) for name, path, cache in args.bundle]
    os.environ.setdefault("USE_TF", "0")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    dataset = args.dataset.read_bytes()
    examples, cases = prepare_examples(dataset, tokenizer, args.limit, args.context, args.start)
    results = {}
    if args.source:
        print(f"Scoring bf16 on {len(examples)} examples", flush=True)
        start = perf_counter()
        scores = score_source(checkpoint, cases)
        results["bf16"] = {"model_path": str(checkpoint), **summarize(examples, scores, perf_counter() - start)}
        print_summary("bf16", results["bf16"], len(examples))
    for name, bundle, cache in bundles:
        print(f"Scoring {name} on {len(examples)} examples", flush=True)
        cache.mkdir(parents=True, exist_ok=True)
        start = perf_counter()
        scores = list(score_litert(bundle, cache, cases, args.context))
        results[name] = {"model_path": str(bundle), **summarize(examples, scores, perf_counter() - start)}
        print_summary(name, results[name], len(examples))

    packages = ["transformers"] + (["torch"] if args.source else []) + (["litert-lm"] if bundles else [])
    report = {
        "dataset_sha256": hashlib.sha256(dataset).hexdigest(),
        "examples": len(examples),
        "start": args.start,
        "context": args.context,
        "versions": {name: version(name) for name in packages},
        "models": results,
    }
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")


def print_summary(name, result, total):
    """Print the two accuracy counts and the elapsed desktop scoring time."""
    print(
        f"{name}: {result['correct']}/{total} raw, "
        f"{result['correct_normalized']}/{total} length-normalized, "
        f"{result['seconds_including_load']:.2f}s including load",
        flush=True,
    )


if __name__ == "__main__":
    main()
