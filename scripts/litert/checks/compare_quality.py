#!/usr/bin/env python3
"""Compare teacher-forced continuation loss between Domyn Edge HF and LiteRT-LM."""

import argparse
import gc
import json
import math
import os
from pathlib import Path


def read_cases(path, tokenizer):
    """Tokenize prefixes and continuations as separate runtime inputs."""
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        prefix = (tokenizer.bos_token or "") + row["prefix"]
        continuation = row["continuation"]
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        target_ids = tokenizer.encode(continuation, add_special_tokens=False)
        if not prefix_ids or not target_ids:
            msg = f"Empty token sequence in {row['id']}"
            raise ValueError(msg)
        yield row["id"], prefix, continuation, prefix_ids, target_ids


def score_source(model_path, cases):
    """Sum BF16 model loss over each teacher-forced continuation."""
    import torch
    from torch.nn.functional import cross_entropy
    from transformers import AutoModelForCausalLM

    torch.set_num_threads(2)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
    ).eval()
    scores = []
    with torch.inference_mode():
        for _, _, _, prefix_ids, target_ids in cases:
            logits = (
                model(torch.tensor([prefix_ids + target_ids]), use_cache=False, logits_to_keep=len(target_ids) + 1)
                .logits[0, :-1]
                .float()
            )
            scores.append(cross_entropy(logits, torch.tensor(target_ids), reduction="sum").item())
    del model
    gc.collect()
    return scores


def score_litert(bundle, cache, cases, context):
    """Score the same continuation IDs in raw LiteRT-LM sessions."""
    import litert_lm as lm

    with lm.Engine(
        str(bundle), backend=lm.Backend.CPU(thread_count=2), max_num_tokens=context, cache_dir=str(cache)
    ) as engine:
        for name, prefix, target, prefix_ids, target_ids in cases:
            if engine.tokenize(prefix) != prefix_ids or engine.tokenize(target) != target_ids:
                msg = f"Tokenizer mismatch in {name}"
                raise ValueError(msg)
            with engine.create_session(apply_prompt_template=False) as session:
                session.run_prefill([prefix])
                result = session.run_text_scoring([target], store_token_lengths=True)
            if result.token_lengths != [len(target_ids)] or len(result.scores) != 1:
                msg = f"Unexpected score shape in {name}"
                raise ValueError(msg)
            yield -float(result.scores[0])


def main():
    """Print per-case and aggregate loss, optionally enforcing a regression limit."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Canonical v5 checkpoint")
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--cases", type=Path, default=Path(__file__).with_name("quality_cases.jsonl"))
    parser.add_argument("--context", type=int, default=512)
    parser.add_argument("--max-nll-increase", type=float, help="Optional gate in nats per scored token")
    args = parser.parse_args()
    if args.max_nll_increase is not None and (not math.isfinite(args.max_nll_increase) or args.max_nll_increase < 0):
        parser.error("The NLL increase limit must be finite and nonnegative")
    checkpoint, bundle = args.checkpoint.resolve(strict=True), args.bundle.resolve(strict=True)
    cache = args.cache_dir.resolve()
    cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("HF_MODULES_CACHE", str(checkpoint.parent / ".hf-modules"))
    from transformers import AutoTokenizer

    # The tokenizer needs no remote code; asking for it would print an interactive prompt into the JSON output.
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True, trust_remote_code=False)
    cases = list(read_cases(args.cases, tokenizer))
    if not cases or any(len(row[3]) + len(row[4]) > args.context for row in cases):
        parser.error("Provide nonempty cases that fit the selected context")
    source_scores = score_source(checkpoint, cases)
    bundle_scores = list(score_litert(bundle, cache, cases, args.context))
    total = sum(len(row[4]) for row in cases)
    if not all(math.isfinite(score) for score in source_scores + bundle_scores):
        msg = "Nonfinite continuation loss"
        raise ValueError(msg)
    for row, src, out in zip(cases, source_scores, bundle_scores, strict=True):
        print(json.dumps({"case": row[0], "tokens": len(row[4]), "source_nll": src, "bundle_nll": out}))
    source_nll = sum(source_scores) / total
    bundle_nll = sum(bundle_scores) / total
    delta = bundle_nll - source_nll
    print(
        json.dumps(
            {
                "scored_tokens": total,
                "source_nll_per_token": source_nll,
                "bundle_nll_per_token": bundle_nll,
                "nll_delta": delta,
                "source_perplexity": math.exp(source_nll),
                "bundle_perplexity": math.exp(bundle_nll),
            }
        )
    )
    return int(args.max_nll_increase is not None and delta > args.max_nll_increase)


if __name__ == "__main__":
    raise SystemExit(main())
