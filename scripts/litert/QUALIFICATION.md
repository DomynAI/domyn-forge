# Domyn Edge LiteRT-LM qualification

This page records how the LiteRT-LM release candidate was built and measured, what was found, and what is still
needed before it is published. Commands are in [README.md](README.md) and [BENCHMARKING.md](BENCHMARKING.md).

The bundle targets CPU inference on edge devices: any phone, tablet or embedded board that runs LiteRT-LM, offline,
with no server. It was qualified on a desktop CPU (macOS arm64) and on an Android 16 phone with a Snapdragon 8 Gen 2
(arm64) and 8 GB of RAM, running offline on the phone's CPU. The GPU backend does not accept it yet (see [GPU](#gpu)).

Every number below comes from a file in [`evidence/`](evidence/) or can be recomputed from one with
`summarize_results`.

## Highlights

| | What stood out |
| --- | --- |
| 🧩 **No model-specific export code** | The canonical Transformers v5 modeling in this repository goes through LiteRT Torch's generic Hugging Face path. The only Domyn-specific code is the stop token and thinking flag in the bundle metadata. |
| ✂️ **289 MiB of mask constants removed** | LiteRT Torch's V2 lowering turned a scalar BF16 mask fill into two full-size FP32 constants. Rewriting the `select` in FP32 keeps the fill a 4-byte scalar: the bundle shrinks by 10.7% (2.63 → 2.35 GiB) with bit-identical INT4 weights. We found no public report of this converter issue. |
| 📉 **Quantization costs about 4 GSM8K points** | On the full GSM8K test split (1,319 problems), the bundle solves 85.1% against 89.2% for the BF16 source (−4.1 points, 95% interval −5.8 to −2.4), level with the Qwen3-4B LiteRT-LM INT4 bundle (84.2%). PIQA changes are within noise. |
| 📱 **Runs offline on an Android phone's CPU** | Tested on an Android 16 phone with a Snapdragon 8 Gen 2 (arm64) and 8 GB of RAM. In Google AI Edge Gallery, offline, it answered arithmetic and multi-turn recall correctly, and found a code in a 1,909-token prompt. Decode is 4–6 tok/s on two CPU threads, about 25–40% slower than Qwen3-4B. |
| ⚠️ **Long chats put pressure on phone memory** | Near 2,000 tokens, Gallery displayed 6.19 GB of device RAM in use and 0.2 tok/s decode. A 1,024-token limit is faster and uses less swap than 2,048, so it is the recommended phone default. |
| ✅ **No zero INT4 scales** | None of the 122,762,880 block scales is zero, which matters for GPU delegates. |

## Candidate

| | |
| --- | --- |
| Source | `domyn/Domyn-Edge-SFT` at `bfdad8935eede2977b2293b9754597529b21e55c` |
| Bundle | INT4 decoder (block 32), INT8 embedder, prefill 128 / 1,024, 2,048-token cache, `--compact-prefill-mask` |
| Size | 2,520,888,320 bytes (2.35 GiB); the same export without the mask rewrite ("v5") is 2.63 GiB |
| SHA-256 | `31425c2ce528f08cb24748add0da99dadf2106437843252467017464c670ea18` |
| Toolchain | LiteRT Torch `13644e33d20cb18b48c7e5fbb2698fa5795740bf`, TensorFlow Nightly 2.23.0.dev20261109, LiteRT Quantizer 0.12.0.dev20261110, LiteRT-LM 0.19.1 |

This candidate was built on a Mac, with the trace and the V2 compilation in separate processes
(see [Limitations](#limitations)). The pinned Linux workflow has not produced a bundle yet.

## Checkpoint migration

The published checkpoint uses a flash-attention window of 1,024 previous tokens. Transformers' `sliding_window`
counts the current token too, so the staged config uses 1,025. This is the convention
[`convert.py`](../huggingface/convert.py) already applies when it converts from Titan (checked bit-for-bit by
`tests/test_hf_conversion.py`). Staging is needed only because the published checkpoint predates that layout.

With `check_v5_parity --tokens 1027 --force-math`, the legacy and staged checkpoints gave exactly the same logits for
both full prefill and cached decode, across the window boundary. On both checkpoints, BF16 cached decode differed
from the full forward pass on 2 of 115,272 logits beyond `atol=0.15, rtol=0.04`, so this drift was already in the
original model and was not introduced by the migration. In FP32, the staged checkpoint's cached decode matched the
full forward pass within 0.0000887 (gate: `atol=rtol=0.001`).

The shared math kernel is used because, in a separate ad hoc test (not included here), PyTorch's BF16 flash kernel
gave results up to 0.00049 apart for the same visible keys passed as a full masked tensor or a cropped one. The math
kernel gave identical outputs.

## Compact prefill mask

`inspect_decoder` on the v5 bundle found two FP32 constants, together 289 MiB, each filled with a single value. They
are the attention-mask fill (`-9984`, i.e. -10000 rounded to BF16): LiteRT Torch's V2 lowering of a BF16 `select`
with a scalar fill turned the broadcast scalar into full-size constants in each prefill signature.

`export/compact_mask_ir.py` rewrites each such `select` in the prefill MLIR (query length of at least 128) to run in FP32
and cast the result back to BF16. Every BF16 value is exactly representable in FP32, so the output is unchanged, and
the fill stays a broadcast scalar through compilation. Decode signatures are not touched.

- On the `diagnostics/reproduce_sdpa_mask.py` fixture, the fill went from 33,636,352 bytes to 4 bytes, and the CPU output matched
  the reference exactly on all 8,409,088 values.
- On the full model, the bundle shrank by 302,661,632 bytes (10.7%). `compare_decoder_weights` found identical packed
  INT4 weights, scales and zero points in all three signatures. The embedder, tokenizer and metadata sections matched
  by SHA-256, and 770 populated decoder buffers matched by hash. The only change was the two fill constants, replaced
  by one 4-byte value.
- With `release_export --compact-prefill-mask`, the hook produced the same prefill IR as the standalone rewrite and
  left the decode IR identical.

We found no public LiteRT Torch or converter issue describing this expansion (searched November 2026).

To apply it to a saved trace instead (for example before resuming with `--compiled-decoder`):

```bash
python -m scripts.litert.export.compact_mask_ir /path/to/model_intermediates /path/to/rewritten \
    --signature prefill_128 --signature prefill_1024
```

then compile `/path/to/rewritten` with the converter flags in `compile_models` in `diagnostics/reproduce_sdpa_mask.py`.

## Results

Unless noted otherwise: macOS arm64, LiteRT-LM 0.19.1, CPU backend, two threads, greedy decoding.

### Quality

| Check | Canonical BF16 | v5 | Candidate |
| --- | ---: | ---: | ---: |
| `compare_quality`, 20 continuations (259 tokens), perplexity | 6.63 (6.65 in the candidate run) | 7.35 | 7.36 |
| PIQA rows 0–49, 512 context, raw / length-normalized | 38 / 39 | 35 / 39 | 35 / 39 |
| PIQA rows 50–149, 1,024 context, raw / length-normalized | 71 / 80 | 71 / 77 | 71 / 77 |

- The candidate's continuation loss is 0.102 nats per token above the BF16 source
  ([`evidence/mac-cpu-quality.jsonl`](evidence/mac-cpu-quality.jsonl)), the same as v5 within 0.01 perplexity. The
  release workflow's default gate is 0.15.
- The candidate made the same prediction as v5 on all 150 PIQA rows under both scoring rules
  ([`evidence/piqa-summary.json`](evidence/piqa-summary.json)). On rows 0–49, 17 of
  the 100 choice scores changed (by at most 1.19 nats of summed loss) without changing any prediction.
- Rows 50–149 were chosen before scoring. Candidate vs. BF16: 0 points raw (95% interval −4 to +4) and −3 points
  length-normalized (−8 to +2), from a paired percentile bootstrap over the per-row predictions (20,000 resamples,
  seed 42). The −3 points were already present in v5, so they come from the export itself, not from the mask rewrite.
  The interval does not rule out a real regression.
- `compare_bundle_prefill` at a 2,048-token budget: v5 and the candidate gave identical scores on all nine prompts
  around the 128- and 1,024-token prefill boundaries and the 1,025-token window (127 to 2,000 tokens).

### Chat generation (GSM8K, IFEval)

`evaluate_chat`, greedy, thinking off, 2,048-token budget, in shards of the same seeded order. BF16 is the staged
checkpoint on Apple MPS through Transformers with the same `quick_chat.jinja` template; the bundles run on LiteRT-LM
CPU with two threads. Qwen3-4B is the same peer bundle as in [CPU latency](#cpu-latency). Summaries come from
`summarize_results chat`, on the rows that all three models completed.

**GSM8K** (full test split, 1,319 problems, 512 output tokens, `solve` prompt;
[`evidence/gsm8k-summary.json`](evidence/gsm8k-summary.json)):

| | BF16 source | Candidate (INT4) | Qwen3-4B (INT4) |
| --- | ---: | ---: | ---: |
| Exact match | 1,176 (89.2%) | 1,122 (85.1%) | 1,110 (84.2%) |
| Final answer on a `####` line | 1,288 | 1,257 | 1,218 |
| Hit the 512-token cap | 26 | 43 | 14 |

- Candidate vs. BF16: −4.1 points (95% interval −5.8 to −2.4); 96 problems only BF16 solved, 42 only the candidate
  solved. Unlike PIQA, this interval excludes zero, so the export does cost reasoning accuracy.
- Candidate vs. Qwen3-4B: +0.9 points (−1.3 to +3.1), no measurable difference.
- The candidate hits the output cap more often than BF16 (43 vs. 26 problems, +1.3 points, interval +0.3 to +2.3),
  so part of the loss is answers that run out of tokens before the `####` line.
- The interim snapshot at 916 problems showed −5.9 points; the full split narrowed the gap.

**IFEval** (541 prompts, 1,024 output tokens): BF16 reaches 393/541 (72.6%) prompt-level strict accuracy and 670/834
(80.3%) instruction-level, with 99 replies hitting the cap. The bundle runs are partly done: 40 of 541 prompts for
the candidate and 321 for Qwen3-4B. They were stopped when the qualification machine ran short of memory, compute
time and disk space, after the GSM8K runs finished. The completed shards are kept, and finishing these runs from
where they stopped is the next evaluation step ([Next steps](#next-steps)). Until then, IFEval has no bundle
comparison.

### Chat smoke tests

The candidate passed `verify_bundle` (arithmetic, three-turn recall with arithmetic in the middle, and long-prompt
retrieval) at a 1,024-token budget. On desktop CPU it answered 7 of the 8 fixed floor questions, with no degenerate
answers. It reversed the answer in the decimal comparison. [Machine-readable desktop results](evidence/mac-cpu-smoke.json)
include the individual replies.
At a 2,048-token budget, it also retrieved the expected code from a 1,026-token prompt. The v5 bundle retrieved the
expected value from a 1,206-token prompt with 6.93 GiB peak process RSS.

### CPU latency

`benchmark_cpu`, warm cache, 1,024-token budget, 64 output tokens. Each value is the median of three runs after the
first run in the same process. The [Qwen3-4B LiteRT-LM bundle](https://huggingface.co/litert-community/Qwen3-4B)
(`qwen3_4b_mixed_int4.litertlm`, revision `65314f65d0630563a2bf26ee977499b55c5c0287`) is included as a peer of similar
size.

| Bundle | Time to first token, 128 / 768 input tokens | Decode, 128 / 768 input tokens |
| --- | ---: | ---: |
| v5 | 2.07 / 11.93 s | 15.38 / 15.64 tok/s |
| Candidate | 2.05 / 11.89 s | 15.84 / 15.84 tok/s |
| Qwen3-4B | 1.99 / 11.65 s | 22.15 / 22.56 tok/s |

Prefill speed is close to Qwen3-4B; decode is about 30% slower. Every run is in
[`evidence/mac-cpu-latency.json`](evidence/mac-cpu-latency.json); `benchmark_cpu` writes the same medians in its
summary record.

### Android phone (CPU, offline)

The test device was an Android 16 phone with a Snapdragon 8 Gen 2 (arm64) and 8 GB of RAM (about 7 GB available to Android), using the
CPU backend (XNNPACK) with two threads, the candidate bundle
(SHA-256 above) and the Qwen3-4B peer bundle. The fixed-token runs used Google's prebuilt Android arm64
`litert_lm_advanced_main`, storage object generation `1793238236151134`, SHA-256
`8e199c6b3fb438cb94943567a279f9e1f994d57c3e1f4667084fc8ae2d06a6eb`.

**Fixed-token benchmark** (`litert_lm_advanced_main`, 1,024-token budget, 32 output tokens, median of 4 iterations):

| Bundle | Input tokens | Time to first token | Prefill | Decode | Peak system RAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Candidate | 128 | 9.12 s | 14.4 tok/s | 4.07 tok/s | 3,276 MB |
| Qwen3-4B | 128 | 7.35 s | 17.8 tok/s | 6.83 tok/s | 3,794 MB |
| Candidate | 768 | 31.12 s | 24.8 tok/s | 5.71 tok/s | 2,990 MB |
| Qwen3-4B | 768 | 34.71 s | 22.2 tok/s | 7.75 tok/s | 2,962 MB |

The phone was not under controlled conditions: the 128-token runs were unplugged on a low battery at thermal status 2,
and the 768-token runs had battery saver on. A later profiling rerun at 128 tokens (USB power, thermal status 0, airplane
mode, median of 2) gave 6.34 s / 5.95 tok/s for the candidate and 6.97 s / 8.07 tok/s for Qwen3-4B. Treat these as
indicative; the ranking held in every set: Qwen3-4B decodes faster, prefill is similar.

**Google AI Edge Gallery** 1.0.23, imported bundle, airplane mode on, Wi-Fi off, CPU:

- Multi-turn chat: "7 times 8" → 56; "Remember this code: LILAC-27" → OK; "What was the code?" → LILAC-27. About
  5.5–6.7 tok/s decode and 7–9 s to first token. Gallery displayed 4.1 GB of device RAM in use.
- Long prompt with a 2,048-token limit: the correct code from a 1,909-token prompt, but 135 s to first token,
  0.2 tok/s decode and 6.19 GB (90%) of RAM.
- A Gallery import with a 4,096-token limit was killed by Android (signal 9) before it answered. That limit is above
  the bundle's 2,048-token cache, so it is not a supported setting; Gallery does not stop you from choosing it.
- The 2.35 GiB imported file produced a 2.07 GiB XNNPACK disk cache, so the two files occupy about 4.42 GiB.

**Context and memory** (native runner, 2,048-token limit unless noted):

- The fixed-token sweep completed all eight trials, using 768, 1,024, 1,536, and 1,900 input tokens with 16 output
  tokens per trial. At 1,900 input tokens, sampled process swap peaked at 3.54 GiB and decode was 0.18–0.47 tok/s.
- In a separate comparison of four paired 768-input, 16-output trials, the 1,024-token limit was faster in every
  pair. The trials alternated order with Battery Saver off. Thermal status ranged from 0 to 1.

| Runtime limit | First token (median) | Decode (median) | Runner-reported peak RAM (median) | Peak sampled swap (median) |
| ---: | ---: | ---: | ---: | ---: |
| 1,024 | 42.8 s | 4.28 tok/s | 3,204 MB | 76 MiB |
| 2,048 | 50.8 s | 2.30 tok/s | 3,544 MB | 565 MiB |

Use 1,024 total tokens as the default for phones in this class on CPU and offer 2,048 for longer prompts. The limit covers the prompt,
chat history, and generated answer. [Machine-readable Android results](evidence/android-cpu.json) include the individual trials.

### GPU

The GPU backend was not tried on the phone. The Mac WebGPU backend rejected the BF16 tensors in the attention composites, so the decoder could not be fully
delegated. GPU inference is not supported yet. Published LiteRT-LM GPU results use FP16 activations and an FP16 KV
cache ([ML Drift](https://developers.googleblog.com/en/ml-drift-next-gen-gpu-aiml-inference-at-the-edge/),
[Qwen3-4B-Thinking-2507](https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507)), not BF16.

### Quantization check

`inspect_decoder` found no zero scales among the 122,762,880 block scales of the 253 INT4 tensors, in either v5 or
the candidate. A zero scale comes from an all-zero weight block; the CPU rejects it but the GPU delegate does not, so
[Google's conversion guide](https://github.com/google-ai-edge/litert-samples/blob/main/models/conversion.md) asks for
this check after every blockwise INT4 export.

### Size compared with similar bundles

Community LiteRT-LM exports of dense Qwen3-4B derivatives with blockwise INT4 and INT8 embeddings are about 2.3 to
2.7 GB (decimal), for example [Qwen3-4B-Thinking-2507](https://huggingface.co/litert-community/Qwen3-4B-Thinking-2507)
and [FastContext-1.0-4B-SFT](https://huggingface.co/litert-community/FastContext-1.0-4B-SFT). The candidate
(2.52 GB) is in that range. v5 (2.82 GB) was above it because of the mask constants.

## What the results say about the recipe

The [Highlights](#highlights) list the measured outcomes. Three design choices explain most of them:

- **Dynamic-range quantization, not weight-only.** The recipe sets `compute_precision: INTEGER`, so the runtime
  quantizes activations on the fly inside every INT4 and INT8 matrix multiplication. This is the scheme Google's
  community INT4 bundles use, and it is what makes CPU decode fast, but it adds activation error on top of weight
  error. The GSM8K loss cannot yet be split between the two.
- **INT4 everywhere.** The recipe quantizes every fully connected layer to INT4, attention projections included.
  The Qwen3-4B peer is a mixed INT4 bundle, which keeps some layers at higher precision, and its own BF16 baseline
  was not measured, so the comparison shows parity in outcome but not in recipe.
- **BF16 activations.** Exporting in BF16 keeps the activation dtype of the source model, but it caused two of the
  three problems found: the mask fill expansion and the GPU delegate rejecting the attention composites. Published
  LiteRT-LM GPU bundles use FP16 or FP32 activations.

## Limitations

- **Toolchain.** The pinned stack contains nightly builds, and the lock has no wheel hashes. The LiteRT Torch source
  archive is pinned by commit only.
- **Build path.** The candidate was traced, compiled and packaged on a Mac in separate processes, not by the single
  `release_export` command on Linux that the workflow runs, and with LiteRT Quantizer 0.12.0.dev20261110 where the
  lock pins dev20261111. A workflow build may differ in bytes and must be scored again.
- **Modeling changes since the build.** The candidate was staged before `configuration_domynedge.py` gained the
  `window_sizes` property and the `max_window_layers` default for vLLM 0.33. Neither changes the modeling code or the
  weights, but the export has not been repeated with them.
- **Evaluation size.** The continuation and PIQA sets are small (20 continuations, 150 PIQA rows), and their
  confidence intervals allow a real regression. `quality_cases.jsonl` is a regression sample, not a benchmark.
  GSM8K covers the full test split. The IFEval bundle runs were stopped partway by memory, compute and disk limits,
  so IFEval has no bundle comparison yet.
- **Chat run provenance.** The chat run's protocol recorded the SHA-256 of `evaluate_chat.py` before it started, but
  the file was edited about two hours into the run, so later shards ran a different file. The output records have
  the same fields before and after the edit, but the change cannot be checked line by line. The release workflow now
  records the hash of every script it runs. Since the runs, the script has also moved to `benchmarks/`, which
  changed only the line that locates `quick_chat.jinja`, so its current hash differs from the recorded one.
- **BF16 reference device.** The chat BF16 reference ran on Apple MPS; the PIQA and continuation references ran on CPU.
- **Scope.** Text chat only, with thinking off and no tool calling. The source model's full chat template is not
  embedded. Context is limited to 2,048 tokens.
- **Devices.** One Android phone, on CPU. Other phones, tablets, embedded boards, iOS and GPU backends have not
  been tested. Power and thermal conditions varied across the tests. The paired
  runtime-limit trial recorded those conditions separately for each run.

## Next steps

1. Run the release workflow on the Linux runner, and compare its bundle with this candidate using
   `compare_decoder_weights` and `compare_bundle_prefill`.
2. Meet the [conversion guide](https://github.com/google-ai-edge/litert-samples/blob/main/models/conversion.md)'s
   quality gates on the final artifact. `verify_bundle` checks the fixed eight-question floor and a three-turn
   conversation. Task parity requires at least 100 examples and an independent FP32 reference. Resume the IFEval
   bundle runs from the stopped shards (candidate from prompt 40, Qwen3-4B from prompt 321) on a machine with enough
   memory and disk, compare them with `summarize_results chat`, and evaluate the full PIQA split when used for a
   release claim.
3. Repeat offline Android qualification for the final artifact if its hash changes, including the 1,024-token default.
4. Split the GSM8K loss between its causes with two exports that change one thing each: weight-only INT4
   (`compute_precision: FLOAT`) and INT8 attention projections. Compare them with `summarize_results chat` on the
   same 1,319 problems.
5. Export with FP32 or FP16 activations to make the attention composites GPU-compatible, starting from the
   `reproduce_sdpa_mask.py` fixture.
6. Report the mask fill expansion upstream to LiteRT Torch with `reproduce_sdpa_mask.py`, and drop the rewrite once
   it is fixed there. At the pinned commit, the BF16 `torch.where` fills are in
   `generative/export_hf/experimental/composites/sdpa.py` (sliding-window prefill and the general path); doing that
   `where` in FP32 at the source would make the IR rewrite unnecessary.
7. Move to a LiteRT Torch release with `ModelExtension` when there is one, and add wheel hashes to the lock.
8. Propose the recipe to [`google-ai-edge/litert-samples`](https://github.com/google-ai-edge/litert-samples/blob/main/models/conversion.md)
   and publish the bundle with a model card once the checks above pass.
