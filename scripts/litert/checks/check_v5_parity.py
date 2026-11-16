#!/usr/bin/env python3
"""Compare legacy and canonical checkpoint logits, including cached decode."""

import argparse
import gc
import json
import os
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch


def cpu_varlen(query, key, value, cu_seq_q, cu_seq_k, max_q, max_k, *, scale, window_size, enable_gqa):
    """Reference for the legacy flash kernel on CPU, where it is unavailable."""
    del max_q, max_k
    import torch
    from torch.nn.functional import scaled_dot_product_attention

    left, right = window_size
    results = []
    qs, ks = cu_seq_q.tolist(), cu_seq_k.tolist()
    for q0, q1, k0, k1 in zip(qs[:-1], qs[1:], ks[:-1], ks[1:], strict=True):
        q, k, v = (x.transpose(0, 1)[None] for x in (query[q0:q1], key[k0:k1], value[k0:k1]))
        q_pos = torch.arange(k1 - k0 - (q1 - q0), k1 - k0)[:, None]
        k_pos = torch.arange(k1 - k0)[None, :]
        visible = k_pos <= q_pos + right
        if left >= 0:
            visible &= k_pos >= q_pos - left
        result = scaled_dot_product_attention(q, k, v, visible, scale=scale, enable_gqa=enable_gqa)
        results.append(result[0].transpose(0, 1))
    return torch.cat(results)


def checkpoint_logits(path, tokens, device, *, legacy, dtype=None, force_math=False):
    """Collect prefill and cached decode logits while loading one checkpoint at a time."""
    import torch
    from torch.nn.attention import SDPBackend, sdpa_kernel, varlen
    from transformers import AutoModelForCausalLM

    implementation = "eager" if legacy else ("sdpa" if device == "cpu" else "torch_varlen")
    model = AutoModelForCausalLM.from_pretrained(
        path,
        trust_remote_code=True,
        local_files_only=True,
        dtype=dtype if dtype is not None else torch.bfloat16,
        attn_implementation=implementation,
        device_map={"": device},
    ).eval()
    kernel = patch.object(varlen, "varlen_attn", cpu_varlen) if legacy and device == "cpu" else nullcontext()
    backend = sdpa_kernel(SDPBackend.MATH) if force_math else nullcontext()
    with torch.inference_mode(), kernel, backend:
        full = model(tokens, use_cache=False, logits_to_keep=2).logits.float().cpu()
        prefill = model(tokens[:, :-1], use_cache=True, logits_to_keep=1)
        decoded = (
            model(tokens[:, -1:], past_key_values=prefill.past_key_values, use_cache=True, logits_to_keep=1)
            .logits.float()
            .cpu()
        )
    del model
    gc.collect()
    return full, decoded


def main():
    """Run the migration parity checks and fail if logits exceed tolerance."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy", type=Path)
    parser.add_argument("canonical", type=Path)
    parser.add_argument(
        "--tokens", type=int, default=24, help="Use a length past the sliding window to check its boundary"
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--atol", type=float, default=0.15)
    parser.add_argument("--rtol", type=float, default=0.04)
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--force-math", action="store_true", help="Use the CPU SDPA math kernel for both checkpoints")
    parser.add_argument(
        "--check",
        choices=("all", "migration", "canonical-cache"),
        default="all",
        help="Select the comparisons that determine the exit status",
    )
    args = parser.parse_args()
    if args.tokens < 3:
        parser.error("Use at least three tokens")
    if args.force_math and args.device != "cpu":
        parser.error("The shared SDPA math control requires CPU")
    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("HF_MODULES_CACHE", str(args.canonical.resolve().parent / ".hf-modules"))
    import torch

    torch.set_num_threads(2)
    config = json.loads((args.legacy / "config.json").read_text())
    tokens = torch.randint(config["vocab_size"], (1, args.tokens), generator=torch.Generator().manual_seed(42))
    tokens = tokens.to(args.device)
    options = {"dtype": getattr(torch, args.dtype), "force_math": args.force_math}
    comparisons = []
    if args.check != "canonical-cache":
        legacy_full, legacy_decode = checkpoint_logits(
            str(args.legacy.resolve()), tokens, args.device, legacy=True, **options
        )
    canonical_full, canonical_decode = checkpoint_logits(
        str(args.canonical.resolve()), tokens, args.device, legacy=False, **options
    )
    if args.check != "canonical-cache":
        comparisons.extend(
            [
                ("full_prefill", canonical_full, legacy_full, True),
                ("cached_decode", canonical_decode, legacy_decode, True),
                ("legacy_cache_vs_full", legacy_decode, legacy_full[:, -1:], args.check == "all"),
            ]
        )
    comparisons.append(("canonical_cache_vs_full", canonical_decode, canonical_full[:, -1:], args.check != "migration"))
    for name, actual, expected, gated in comparisons:
        delta = (actual - expected).abs()
        close = torch.isclose(actual, expected, atol=args.atol, rtol=args.rtol)
        print(
            json.dumps(
                {
                    "case": name,
                    "gated": gated,
                    "tokens": args.tokens,
                    "max_abs": delta.max().item(),
                    "mean_abs": delta.mean().item(),
                    "outside_tolerance": int((~close).sum()),
                    "compared_logits": close.numel(),
                    "same_top_token": torch.equal(actual.argmax(-1), expected.argmax(-1)),
                    "atol": args.atol,
                    "rtol": args.rtol,
                    "device": args.device,
                    "dtype": args.dtype,
                    "force_math": args.force_math,
                    "legacy_cpu_kernel_reference": args.device == "cpu" and args.check != "canonical-cache",
                }
            ),
            flush=True,
        )
    for _, actual, expected, gated in comparisons:
        if gated:
            torch.testing.assert_close(actual, expected, atol=args.atol, rtol=args.rtol)


if __name__ == "__main__":
    main()
