# Domyn Edge to LiteRT-LM export

This directory converts the public [`domyn/Domyn-Edge-SFT`](https://huggingface.co/domyn/Domyn-Edge-SFT) checkpoint
into a `.litertlm` bundle for on-device inference with [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM)
(for example in Google AI Edge Gallery). No training is involved: the bundle is a post-training quantization of the
released SFT weights.

The bundle is meant for CPU inference on edge devices: phones, tablets and embedded boards that run LiteRT-LM, fully
offline. It has been tested on an Android 16 phone (Snapdragon 8 Gen 2, 8 GB of RAM) on the CPU backend, in Google AI
Edge Gallery and with the native LiteRT-LM runner, and on a macOS arm64 desktop CPU. The GPU backend does not accept
it yet.

Results, findings and open release checks are in [QUALIFICATION.md](QUALIFICATION.md). Evaluation and benchmark
commands are in [BENCHMARKING.md](BENCHMARKING.md).

## Results at a glance

| | BF16 source | INT4 bundle | Qwen3-4B INT4 (peer) |
| --- | ---: | ---: | ---: |
| Size | 7.3 GiB (safetensors) | 2.35 GiB | 2.12 GiB |
| GSM8K exact match (full test split, 1,319 problems) | 89.2% | 85.1% | 84.2% |
| Continuation perplexity (20 cases) | 6.65 | 7.36 | — |
| PIQA rows 50–149, raw / length-normalized | 71 / 80 | 71 / 77 | — |
| Mac CPU decode, 2 threads, 128 input tokens | — | 15.8 tok/s | 22.2 tok/s |
| Android phone CPU decode, 2 threads, 128 / 768 input tokens | — | 4.1 / 5.7 tok/s | 6.8 / 7.8 tok/s |

See [QUALIFICATION.md](QUALIFICATION.md#highlights) for what stood out in the conversion: the 289 MiB
attention-mask constants removed by `--compact-prefill-mask`, the quantization cost, and the phone's memory limits.
The raw and summarized results behind this table are in [`evidence/`](evidence/).

## Overview

1. **Stage** (`export/stage_v5_checkpoint.py`). The published checkpoint uses the legacy `window_sizes` config and its own
   remote code. Staging writes a fresh directory with Forge's canonical modeling from
   [`assets/domyn-edge/hf/`](../../assets/domyn-edge/hf/), converts each flash-style `(1024, 0)` window into
   `layer_types` plus `sliding_window: 1025` (Transformers counts the current token, as in
   [`convert.py`](../huggingface/convert.py)), and symlinks the original safetensors without copying or changing them.
2. **Check the migration** (`checks/check_v5_parity.py`). Compares legacy and staged logits for full prefill and cached
   decode, past the sliding-window boundary.
3. **Export** (`export/release_export.py`). One LiteRT Torch call that traces, converts, quantizes and packages the bundle.
4. **Verify** (`checks/verify_bundle.py`, `checks/compare_quality.py`, `checks/inspect_decoder.py`). Chat smoke tests, teacher-forced loss
   against the BF16 source, and a decoder report that fails if any INT4 block scale is zero.

## Export settings

| Setting | Value |
| --- | --- |
| Prefill signatures | 128 and 1,024 tokens, plus decode |
| KV cache / runtime token budget | 2,048 |
| Decoder weights | INT4, blockwise (block 32), symmetric, OCTAV ([`int4_reasoning_recipe.json`](export/int4_reasoning_recipe.json)) |
| Embeddings | INT8, channelwise, in a separate embedder model |
| Activations | BF16 between operations; quantized on the fly inside the INT4 and INT8 matrix multiplications (dynamic-range quantization, `compute_precision: INTEGER`) |
| Chat template | [`quick_chat.jinja`](export/quick_chat.jinja): Domyn system identity, ChatML, reasoning block pre-closed (thinking off) |
| Stop token | The tokenizer's turn-end token, added to the bundle metadata |

`release_export.py` also changes two things inside the pinned exporter, neither of which changes the recipe or the
bundle layout:

- Quantization writes through LiteRT Quantizer's file-backed serializer, because the pinned exporter otherwise holds
  the whole quantized model in RAM.
- With `--compact-prefill-mask`, the prefill attention masks are rewritten before TFLite compilation, so the BF16 mask
  fill stays a scalar instead of becoming a large constant (`export/compact_mask_ir.py`, explained in
  [QUALIFICATION.md](QUALIFICATION.md#compact-prefill-mask)).

After packaging, the exporter reads the bundle metadata back and fails if the template, stop token, thinking flag or
output channels are not the expected ones.

## Toolchain

Export on native Linux x86-64 with the image from [`release.Dockerfile`](release.Dockerfile): Python 3.13 pinned by
digest, packages pinned by [`release-linux-amd64.lock`](release-linux-amd64.lock), and LiteRT Torch at commit
`13644e33d20cb18b48c7e5fbb2698fa5795740bf`. The latest LiteRT Torch release (0.10.0) does not have the
`ModelExtension` API used here, so the lock includes nightly LiteRT, LiteRT Quantizer and TensorFlow builds. The lock
pins versions but not wheel hashes.

A full export needs about 32 GiB of RAM and 50 GiB of free disk. `release_export.py` refuses to start if the output
volume has less than four times the checkpoint's weights plus 2 GiB free.

## Usage

From the repository root, inside the toolchain image, run every step above with one command:

```bash
hf download domyn/Domyn-Edge-SFT --revision bfdad8935eede2977b2293b9754597529b21e55c --local-dir /work/source
scripts/litert/release.sh /work/source /work/run 0.15
```

[`release.sh`](release.sh) is the same script the release workflow runs. It writes the staged checkpoint, the parity,
decoder, smoke-test and quality reports, `pip freeze`, the input hashes and `bundle/model.litertlm` into
`/work/run`, and stops at the first failing check. The last argument is the largest allowed increase in
continuation loss over the BF16 source, in nats per token; the qualified candidate measured 0.102. The scripts
refuse to overwrite existing outputs. Unless `HF_MODULES_CACHE` is set, the remote-code module cache goes next
to the staged checkpoint. `release_export.py` also
checks that the staged modeling files match this checkout byte for byte, so restage after changing
`assets/domyn-edge/hf/`.

`check_v5_parity.py` uses `--tokens 1027` to cross the 1,025-token window. `--force-math` runs both checkpoints
through the same SDPA math kernel on CPU (the legacy flash kernel is replaced by an SDPA reference), so the migration
check is not affected by kernel differences. `compare_quality.py` scores the 20 continuations in
[`quality_cases.jsonl`](checks/quality_cases.jsonl) by default; pass `--cases` for another JSONL file with `id`, `prefix`
and `continuation` fields.

### Resuming after a compiler failure

If tracing succeeds but TFLite compilation fails, for example from running out of memory, compile the saved V2
bytecode with a compatible LiteRT V2 compiler and finish the export from that decoder:

```bash
python -m scripts.litert.export.release_export /work/v5 /work/bundle-resumed \
    --compiled-decoder /path/to/model.tflite --quantized-embedder /path/to/embedder_quantized.tflite
```

Use the embedder from the same run, and the same prefill lengths and cache length. To apply the mask rewrite to a
saved trace, run `export/compact_mask_ir.py` on it before compiling (see [QUALIFICATION.md](QUALIFICATION.md#compact-prefill-mask)).

## Release workflow

[`domyn-edge-litert-release.yml`](../../.github/workflows/domyn-edge-litert-release.yml) is a manually triggered
workflow for a self-hosted `domyn-litert` Linux runner with Docker. It takes a full checkpoint commit SHA and an NLL
threshold (default 0.15), builds the toolchain image, downloads the checkpoint, then runs `release.sh` with
networking disabled.
It records the Forge revision, checkpoint revision, image ID, `pip freeze`, input hashes, the parity, smoke-test and
quality reports, and the bundle checksum in the run directory on the runner. GitHub Actions retains the small
qualification reports as a run artifact. The workflow has not yet run end to end.

Use the [Hugging Face model card draft](evidence/MODEL_CARD.md) for the derived `.litertlm` bundle. Finalize its
checksum and qualification results using the Linux output before upload.

## Layout

The folders follow the order of a release: the scripts in `export/` and `checks/` are what `release.sh` runs and
gates on; `benchmarks/` measures a finished bundle; `diagnostics/` holds the investigation tools behind the mask
rewrite and the bundle comparisons; `evidence/` holds the results quoted in [QUALIFICATION.md](QUALIFICATION.md).
Run every script as a module from the repository root, for example `python -m scripts.litert.checks.verify_bundle`.

| Path | Purpose |
| --- | --- |
| [`release.sh`](release.sh) | The full stage, check, export and qualify sequence, shared by local runs and the release workflow |
| [`release.Dockerfile`](release.Dockerfile), [`release-linux-amd64.lock`](release-linux-amd64.lock) | Pinned Linux toolchain image |
| **[`export/`](export/)** | **Build the bundle** |
| `stage_v5_checkpoint.py` | Stage the published checkpoint with Forge's canonical modeling |
| `release_export.py` | Export, quantize, package and check the bundle metadata |
| `compact_mask_ir.py` | Keep the prefill mask fill scalar (used by `--compact-prefill-mask`, or standalone on a trace) |
| `int4_reasoning_recipe.json`, `quick_chat.jinja` | Quantization recipe and the chat template embedded in the bundle |
| **[`checks/`](checks/)** | **Release gates** |
| `check_v5_parity.py` | Legacy vs. staged logits, and cached vs. full decode |
| `inspect_decoder.py` | Decoder signatures, largest constant buffers, and a check that no INT4 block scale is zero |
| `verify_bundle.py` | Chat smoke suite: arithmetic, multi-turn recall, long-prompt retrieval, the eight-question floor |
| `compare_quality.py`, `quality_cases.jsonl` | Teacher-forced continuation loss, BF16 source vs. bundle |
| **[`benchmarks/`](benchmarks/)** | **Measure a bundle** (see [BENCHMARKING.md](BENCHMARKING.md)) |
| `prepare_datasets.py` | Download the pinned PIQA, GSM8K and IFEval revisions and check their SHA-256 |
| `evaluate_piqa.py`, `evaluate_chat.py` | PIQA teacher-forced scoring; IFEval and GSM8K generation |
| `benchmark_cpu.py` | Fixed-token CPU latency with warm medians |
| `summarize_results.py` | Paired comparisons with bootstrap intervals, from PIQA reports or chat shards |
| `runtime_stats.py` | Runtime-cache size and peak RSS helpers shared by the benchmark and smoke-test scripts |
| **[`diagnostics/`](diagnostics/)** | **Investigation tools** |
| `reproduce_sdpa_mask.py` | Small fixture that reproduces the mask fill expansion and its fix |
| `compare_decoder_weights.py` | Check that two decoders have identical INT4 weights, scales and zero points |
| `compare_bundle_prefill.py` | Compare two bundles' scores around the prefill and sliding-window boundaries |
| **[`evidence/`](evidence/)** | GSM8K and PIQA summaries, quality, latency, smoke tests, Android trials, and the model card draft |

## Contributing

- Run `uv run pre-commit run --all-files` before opening a pull request. It runs ruff and the CPU unit tests,
  including [`tests/test_litert_scripts.py`](../../tests/test_litert_scripts.py), which cover checkpoint staging,
  GSM8K and PIQA scoring, shard ordering and the result summaries without needing LiteRT or a checkpoint.
- Put a new script in the folder for its role, and import shared code by its full module path
  (`from scripts.litert.benchmarks.runtime_stats import ...`).
- Every script writes JSON or JSON lines with the dataset hash, package versions and settings, and refuses to
  overwrite an existing output. Keep both properties in new scripts, so results stay comparable.
- When you report a result in [QUALIFICATION.md](QUALIFICATION.md), give the command, the bundle SHA-256 and the
  device, and compare on matched rows with `summarize_results`.
