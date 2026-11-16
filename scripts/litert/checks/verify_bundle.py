#!/usr/bin/env python3
"""Run the Domyn Edge quick chat smoke suite and emit timing as JSON lines."""

import argparse
import io
import json
import platform
import re
import shutil
from dataclasses import asdict
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import litert_lm as lm
from litert_lm_builder.litertlm_peek import read_litertlm_header

from scripts.litert.benchmarks.evaluate_chat import stream_visible
from scripts.litert.benchmarks.runtime_stats import directory_bytes, peak_rss_bytes

FLOOR_QUESTIONS = (
    ("addition", "What is 17 + 25?", r"(?:17\s*\+\s*25\s*(?:=|is)\s*)?42[.!]?"),
    ("capital", "What is the capital of Japan?", r"(?:The capital of Japan is )?Tokyo[.!]?"),
    ("opposite", 'What is the opposite of "hot"?', r'(?:The opposite of ["“]?hot["”]? is )?["“]?cold[.!]?["”]?[.!]?'),
    ("week", "How many days are in a week?", r"(?:There are )?(?:7|seven)(?: days(?: in a week)?)?[.!]?"),
    ("translation", 'How do you say "thank you" in French?', r'(?:Thank you in French is )?["“]?merci[.!]?["”]?[.!]?'),
    ("multiplication", "What is 8 times 7?", r"(?:8\s*(?:times|\u00d7|\*)\s*7\s*(?:=|is)\s*)?56[.!]?"),
    ("decimal", "Which is larger: 0.9 or 0.11?", r"0\.9(?: is (?:larger|greater)(?: than 0\.11)?)?[.!]?"),
    ("rhyme", 'Complete the rhyme: "Roses are red, violets are ___"', r"(?:Roses are red, violets are )?blue[.!]?"),
)


def report(**values):
    """Print one JSON line and flush it, so partial runs stay readable."""
    print(json.dumps(values, ensure_ascii=False), flush=True)


def run_turn(chat, name, prompt, expected):
    """Send one prompt, report timing, and check the visible reply against a regex."""
    report(case=name, status="running")
    answer, first, elapsed = stream_visible(chat, prompt, max_output_tokens=64)
    answer = answer.strip()
    words = re.findall(r"\w+", answer.lower())
    repeated = len(words) > 1 and any(words.count(word) > len(words) / 2 for word in set(words))
    degenerate = not words or repeated or "<think>" in answer
    passed = not degenerate and re.fullmatch(expected, answer, re.IGNORECASE) is not None
    report(
        case=name,
        passed=passed,
        degenerate=degenerate,
        answer=answer,
        first_text_seconds=first,
        generation_seconds=elapsed,
        **asdict(chat.get_benchmark_info()),
    )
    return passed, degenerate


def main():
    """Run the selected cases and return a nonzero status if any fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path, help="Cache directory or :memory for no disk cache")
    parser.add_argument("--backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--context", type=int, choices=(512, 1024, 2048, 4096), default=1024)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--case", choices=("all", "arithmetic", "recall", "long-prefill", "floor"), default="all")
    parser.add_argument(
        "--long-entries", type=int, default=18, help="Number of filler sentences in the long-prompt case."
    )
    parser.add_argument("--chat-template", type=Path, help="Override the template for baseline A/B checks.")
    args = parser.parse_args()
    if args.threads < 1 or args.long_entries < 1:
        parser.error("Thread and filler counts must be positive")

    bundle = args.bundle.resolve(strict=True)
    cache = None if str(args.cache_dir) == ":memory" else args.cache_dir.resolve()
    cache_before = 0
    if cache is not None:
        cache.mkdir(parents=True, exist_ok=True)
        cache_before = directory_bytes(cache)
        if shutil.disk_usage(cache).free < 4 * 1024**3 and cache_before == 0:
            parser.error("Allow at least 4 GiB free for the initial compilation cache and host headroom")
    header_info = io.StringIO()
    header = read_litertlm_header(str(bundle), header_info)
    report(
        model=str(bundle),
        bundle_bytes=bundle.stat().st_size,
        sections=header.SectionMetadata().ObjectsLength(),
        bundle_format=header_info.getvalue().strip(),
        runtime=version("litert-lm"),
        platform=platform.platform(),
        backend=args.backend,
        context=args.context,
        cache_bytes_before=cache_before,
    )

    backend = lm.Backend.CPU(thread_count=args.threads) if args.backend == "cpu" else lm.Backend.GPU()
    template = args.chat_template.read_text() if args.chat_template else None
    results = []
    start = perf_counter()
    with lm.Engine(
        str(bundle),
        backend=backend,
        max_num_tokens=args.context,
        cache_dir=str(cache or ":memory"),
        enable_benchmark=True,
    ) as engine:
        report(engine_init_seconds=perf_counter() - start)
        config = {
            "sampler_config": lm.SamplerConfig(top_k=1, seed=0 if args.case == "floor" else 42),
            "chat_template": template,
        }
        if args.case == "floor":
            config["extra_context"] = {"enable_thinking": False}
            for name, question, expected in FLOOR_QUESTIONS:
                with engine.create_conversation(**config) as chat:
                    results.append(run_turn(chat, name, f"{question} Answer briefly.", expected))
        if args.case in ("all", "arithmetic"):
            with engine.create_conversation(**config) as chat:
                results.append(run_turn(chat, "arithmetic", "What is 2 + 2? Reply only with the number.", r"4\.?"))
        if args.case in ("all", "recall"):
            with engine.create_conversation(**config) as chat:
                results.append(
                    run_turn(chat, "recall_setup", "Remember the code MINT-742. Reply only with OK.", r"OK[.!]?")
                )
                results.append(
                    run_turn(chat, "recall_arithmetic", "What is 17 + 25? Reply only with the number.", r"42\.?")
                )
                results.append(
                    run_turn(
                        chat,
                        "recall",
                        "What code did I ask you to remember? Reply with just the code.",
                        r"MINT-742[.!]?",
                    )
                )
        if args.case in ("all", "long-prefill"):
            with engine.create_conversation(**config) as chat:
                filler = "This is a filler sentence about garden plants.\n" * args.long_entries
                prompt = (
                    "The access code is LIME-381.\n" + filler + "What is the access code? Reply with just the code."
                )
                results.append(run_turn(chat, "long_prefill", prompt, r"LIME-381[.!]?"))
    cache_after = directory_bytes(cache) if cache is not None else 0
    correct = sum(correct for correct, _ in results)
    degenerate = sum(degenerate for _, degenerate in results)
    passed = degenerate == 0 and correct >= (6 if args.case == "floor" else len(results))
    report(
        cache_bytes_after=cache_after,
        peak_process_rss_bytes=peak_rss_bytes(),
        correct=correct,
        examples=len(results),
        degenerate=degenerate,
        passed=passed,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
