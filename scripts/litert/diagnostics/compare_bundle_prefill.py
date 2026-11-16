#!/usr/bin/env python3
"""Compare continuation scores across prefill and sliding-window boundaries."""

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from pathlib import Path

from scripts.litert.checks.compare_quality import score_litert

BOUNDARIES = (127, 128, 129, 1023, 1024, 1025, 1026, 1027, 2000)
CONTEXT = 2048
TARGET = " The gardener returns tomorrow."


def make_cases(tokenizer, lengths):
    """Create deterministic prompts with exact requested token counts."""
    text = " The garden contains herbs, flowers, and carefully labeled seed beds." * max(lengths)
    bos = tokenizer.bos_token or ""
    bos_ids = tokenizer.encode(bos, add_special_tokens=False)
    tokens = tokenizer.encode(text, add_special_tokens=False, truncation=True, max_length=max(lengths))
    cases = []
    for length in lengths:
        prefix_ids = bos_ids + tokens[: length - len(bos_ids)]
        prefix = bos + tokenizer.decode(
            prefix_ids[len(bos_ids) :], skip_special_tokens=False, clean_up_tokenization_spaces=False
        )
        if tokenizer.encode(prefix, add_special_tokens=False) != prefix_ids or len(prefix_ids) != length:
            msg = f"Prompt did not round-trip at {length} tokens"
            raise ValueError(msg)
        target_ids = tokenizer.encode(TARGET, add_special_tokens=False)
        cases.append((str(length), prefix, TARGET, prefix_ids, target_ids))
    return cases


def main():
    """Score both bundles with the same tokenizer and fail on a score regression."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--baseline-cache", required=True, type=Path)
    parser.add_argument("--candidate-cache", required=True, type=Path)
    parser.add_argument("--max-score-delta", type=float, default=0.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if not math.isfinite(args.max_score_delta) or args.max_score_delta < 0:
        parser.error("Use a finite, nonnegative score tolerance")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True, trust_remote_code=False)
    cases = make_cases(tokenizer, BOUNDARIES)
    with args.output.open("x", encoding="utf-8") as out:
        baseline = list(score_litert(args.baseline.absolute(), args.baseline_cache.absolute(), cases, CONTEXT))
        candidate = list(score_litert(args.candidate.absolute(), args.candidate_cache.absolute(), cases, CONTEXT))
        rows = []
        for case, old, new in zip(cases, baseline, candidate, strict=True):
            _, prefix, target, prefix_ids, target_ids = case
            rows.append(
                {
                    "prefix_tokens": len(prefix_ids),
                    "target_tokens": len(target_ids),
                    "text_sha256": hashlib.sha256((prefix + target).encode()).hexdigest(),
                    "baseline_nll": old,
                    "candidate_nll": new,
                    "absolute_delta": abs(new - old),
                }
            )
        maximum = max(row["absolute_delta"] for row in rows)
        passed = all(math.isfinite(row["absolute_delta"]) for row in rows) and maximum <= args.max_score_delta
        result = {
            "runtime": version("litert-lm"),
            "context": CONTEXT,
            "max_score_delta": args.max_score_delta,
            "passed": passed,
            "cases": rows,
        }
        print(json.dumps(result, indent=2, allow_nan=False), file=out)
    print(f"Boundary scoring {'passed' if passed else 'failed'}, max NLL delta {maximum:.6g}")
    return int(not passed)


if __name__ == "__main__":
    raise SystemExit(main())
