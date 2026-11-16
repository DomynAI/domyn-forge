#!/usr/bin/env python3
"""Benchmark one LiteRT-LM bundle on CPU with explicit token budgets."""

import argparse
import hashlib
import json
import platform
import statistics
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path

from scripts.litert.benchmarks.runtime_stats import directory_bytes, peak_rss_bytes

THREADS = 2


def write_record(out, **record):
    """Save each measurement as it finishes."""
    print(json.dumps(record, allow_nan=False), file=out, flush=True)


def warm_medians(runs):
    """Return the median of every numeric field over the runs after the first (warm-up) run."""
    warm = runs[1:]
    if not warm:
        return {}
    fields = [key for key, value in warm[0].items() if isinstance(value, int | float) and not isinstance(value, bool)]
    return {key: statistics.median(run[key] for run in warm) for key in fields if key != "index"}


def main():
    """Record a first run and subsequent runs with the same engine, then their warm medians."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path, help="New JSONL file")
    parser.add_argument("--prefill", type=int, required=True, help="Prompt tokens per run")
    parser.add_argument("--decode", type=int, default=64, help="Generated tokens per run")
    parser.add_argument("--context", type=int, default=2048, help="Runtime token budget (max_num_tokens)")
    parser.add_argument("--runs", type=int, default=4, help="Runs in one process; the first is a warm-up")
    args = parser.parse_args()
    if min(args.prefill, args.decode, args.runs) < 1 or args.prefill + args.decode > args.context:
        parser.error("Use positive token and run counts that fit the runtime context")
    if args.output.exists():
        parser.error(f"Output already exists: {args.output}")
    # Hugging Face snapshot links have an extensionless blob as their target, so keep the link path.
    bundle = args.bundle.absolute()
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    with bundle.open("rb") as weights:
        checksum = hashlib.file_digest(weights, "sha256").hexdigest()
    import litert_lm as lm

    with args.output.open("x", encoding="utf-8") as out:
        write_record(
            out,
            kind="config",
            bundle=str(bundle),
            bundle_sha256=checksum,
            bundle_bytes=bundle.stat().st_size,
            platform=platform.platform(),
            runtime=version("litert-lm"),
            backend="cpu",
            threads=THREADS,
            context=args.context,
            prefill=args.prefill,
            decode=args.decode,
            cache_bytes_before=directory_bytes(args.cache_dir),
        )
        runs = []
        with lm.Benchmark(
            model_path=str(bundle),
            backend=lm.Backend.CPU(thread_count=THREADS),
            max_num_tokens=args.context,
            prefill_tokens=args.prefill,
            decode_tokens=args.decode,
            cache_dir=str(args.cache_dir.resolve()),
        ) as benchmark:
            for index in range(args.runs):
                runs.append({"index": index, **asdict(benchmark.run())})
                write_record(out, kind="run", **runs[-1])
        write_record(
            out,
            kind="summary",
            warm_runs=len(runs) - 1,
            warm_median=warm_medians(runs),
            peak_rss_bytes=peak_rss_bytes(),
            cache_bytes_after=directory_bytes(args.cache_dir),
        )


if __name__ == "__main__":
    main()
