#!/usr/bin/env python3
"""Summarize PIQA reports or evaluate_chat shards as paired comparisons between models.

Every comparison is made on the rows that all models completed. Differences are reported in accuracy points
with a paired percentile bootstrap over rows (20,000 resamples, seed 42), as in QUALIFICATION.md.
"""

import argparse
import glob
import json
import random
import statistics
from itertools import combinations
from pathlib import Path

RESAMPLES = 20_000
SEED = 42
# Settings that must be identical across all shards before their rows are compared.
CHAT_SETTINGS = ("task", "dataset_sha256", "context", "output_tokens", "seed", "scorer", "thinking", "gsm8k_prompt")
PIQA_SETTINGS = ("dataset_sha256", "start", "examples", "context")
# Fields that identify what one model's shards ran; they may differ between models.
MODEL_IDENTITY = ("model_sha256", "backend", "runtime", "transformers")


def paired_bootstrap(deltas, resamples=RESAMPLES, seed=SEED):
    """Return the mean per-row difference and its 95% percentile interval, both in points."""
    if not deltas:
        msg = "No paired rows to compare"
        raise ValueError(msg)
    rng = random.Random(seed)
    draws = sorted(100 * statistics.mean(rng.choices(deltas, k=len(deltas))) for _ in range(resamples))
    low, high = draws[int(0.025 * resamples)], draws[int(0.975 * resamples) - 1]
    return 100 * statistics.mean(deltas), [low, high]


def compare(models, metrics):
    """Compare every pair of models on shared rows; `models` maps a name to {row id: {metric: bool}}."""
    common = sorted(set.intersection(*(set(rows) for rows in models.values())))
    summary = {
        "matched_rows": len(common),
        "models": {
            name: {metric: sum(bool(rows[i][metric]) for i in common) for metric in metrics}
            for name, rows in models.items()
        },
        "paired": [],
    }
    for (reference, left), (candidate, right) in combinations(models.items(), 2):
        for metric in metrics:
            deltas = [int(bool(right[i][metric])) - int(bool(left[i][metric])) for i in common]
            delta, interval = paired_bootstrap(deltas)
            summary["paired"].append(
                {
                    "reference": reference,
                    "candidate": candidate,
                    "metric": metric,
                    "candidate_only": deltas.count(1),
                    "reference_only": deltas.count(-1),
                    "delta_points": delta,
                    "interval_95_points": interval,
                }
            )
    return summary


def read_chat_shards(pattern):
    """Read the config and example records of the evaluate_chat shards matching a glob pattern."""
    paths = sorted(glob.glob(pattern))
    if not paths:
        msg = f"No shards match {pattern}"
        raise ValueError(msg)
    configs, rows = [], {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            # A shard that is still running can end with a partially written line.
            if not line.endswith("}"):
                continue
            record = json.loads(line)
            if record["kind"] == "config":
                configs.append(record)
            elif record["kind"] == "example":
                if record["index"] in rows:
                    msg = f"Duplicate row {record['index']} in {path}"
                    raise ValueError(msg)
                rows[record["index"]] = record
    return configs, rows


def summarize_chat(models):
    """Compare evaluate_chat runs given as (name, glob pattern) pairs."""
    configs, runs, provenance = [], {}, {}
    for name, pattern in models:
        model_configs, runs[name] = read_chat_shards(pattern)
        configs.extend(model_configs)
        # All shards of one model must have run the same weights on the same backend.
        identities = {json.dumps([config.get(key) for key in MODEL_IDENTITY]) for config in model_configs}
        if len(identities) != 1:
            msg = f"Shards of {name} differ in {', '.join(MODEL_IDENTITY)}"
            raise ValueError(msg)
        provenance[name] = {key: model_configs[0].get(key) for key in MODEL_IDENTITY}
    for setting in CHAT_SETTINGS:
        if len({json.dumps(config.get(setting)) for config in configs}) != 1:
            msg = f"Shards differ in {setting}"
            raise ValueError(msg)
    task = configs[0]["task"]
    common = set.intersection(*(set(rows) for rows in runs.values()))
    for i in common:
        if len({rows[i]["key"] for rows in runs.values()}) != 1:
            msg = f"Row {i} has different dataset keys across models"
            raise ValueError(msg)
    metrics = ["correct", "hit_output_cap"]
    if task == "gsm8k":
        metrics.append("valid_format")
    return {
        "task": task,
        "settings": {setting: configs[0].get(setting) for setting in CHAT_SETTINGS},
        "provenance": provenance,
        "completed_rows": {name: len(rows) for name, rows in runs.items()},
        **compare(runs, metrics),
    }


def summarize_piqa(paths):
    """Compare all models in one or more evaluate_piqa reports scored on the same rows."""
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in paths]
    for report in reports:
        # Reports written before evaluate_piqa had --start always began at row 0.
        report.setdefault("start", 0)
    for setting in PIQA_SETTINGS:
        if len({report[setting] for report in reports}) != 1:
            msg = f"Reports differ in {setting}"
            raise ValueError(msg)
    models = {}
    for report in reports:
        for name, result in report["models"].items():
            if name in models:
                msg = f"Model {name} appears in more than one report"
                raise ValueError(msg)
            models[name] = {
                row["row"]: {
                    "raw": row["prediction"] == row["label"],
                    "length_normalized": row["prediction_normalized"] == row["label"],
                }
                for row in result["predictions"]
            }
    return {
        "settings": {setting: reports[0][setting] for setting in PIQA_SETTINGS},
        **compare(models, ["raw", "length_normalized"]),
    }


def main():
    """Print one JSON summary."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    chat = commands.add_parser("chat", help="Compare evaluate_chat JSONL shards")
    chat.add_argument(
        "--model",
        nargs=2,
        action="append",
        required=True,
        metavar=("NAME", "GLOB"),
        help="Model name and a quoted glob for its shards, for example 'runs/gsm8k-bf16-*.jsonl'",
    )
    piqa = commands.add_parser("piqa", help="Compare evaluate_piqa reports")
    piqa.add_argument("reports", nargs="+", type=Path)
    args = parser.parse_args()
    if args.command == "chat":
        if len({name for name, _ in args.model}) != len(args.model) or len(args.model) < 2:
            parser.error("Give at least two models with unique names")
        summary = summarize_chat(args.model)
    else:
        summary = summarize_piqa(args.reports)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
