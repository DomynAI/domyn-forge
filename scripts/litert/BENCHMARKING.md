# Domyn Edge LiteRT-LM benchmarks

How to measure a bundle before deploying it to an edge device. Results are in [QUALIFICATION.md](QUALIFICATION.md).

Run everything from the repository root with `python -m`, in the toolchain image from [README.md](README.md#toolchain)
or an environment with the same Transformers, PyTorch and LiteRT-LM versions. IFEval also needs `lm-eval==0.4.13`,
`langdetect` and the NLTK `punkt_tab` data, and preparing the datasets needs `pyarrow`. Use a new output file and a
separate runtime cache directory for each bundle. Run latency measurements one at a time, with no other model
running.

## Datasets

Download the pinned dataset revisions once. The script writes `piqa-validation.jsonl`, `gsm8k-test.jsonl` and
`ifeval-input.jsonl`, and fails unless each file has the SHA-256 recorded in the published results:

```bash
export HF_HOME="$PWD/.cache/litert-eval/hf"
python -m scripts.litert.benchmarks.prepare_datasets .cache/litert-eval
```

| Dataset | Source | SHA-256 of the JSONL |
| --- | --- | --- |
| PIQA validation | [`baber/piqa`](https://huggingface.co/datasets/baber/piqa/tree/142f6d7367fd9877f0fb3b5734ea6a545f54cdd1) | `61533005…1f55` |
| GSM8K test | [`openai/gsm8k`](https://huggingface.co/datasets/openai/gsm8k/tree/740312add88f781978c0658806c59bc2815b9866/main) `main` | `752adc99…bafbac` |
| IFEval | [`google/IFEval`](https://huggingface.co/datasets/google/IFEval/tree/193ae952ec892316cf21fb5d8cbb704e811fe873) | `6a85310c…88f2` |

## PIQA

PIQA scores the two answer choices of each question with teacher forcing, on the same token IDs for the BF16 source
and each bundle. It reports raw accuracy and accuracy with each choice's loss divided by its length in characters,
as in the [lm-evaluation-harness task](https://github.com/EleutherAI/lm-evaluation-harness/blob/main/lm_eval/tasks/piqa/piqa.yaml).

Score the source (`--source`) and any number of bundles (`--bundle NAME FILE CACHE_DIR`):

```bash
python -m scripts.litert.benchmarks.evaluate_piqa STAGED_CHECKPOINT .cache/litert-eval/piqa-validation.jsonl \
    --source \
    --bundle v5 V5_BUNDLE V5_CACHE \
    --bundle candidate CANDIDATE_BUNDLE CANDIDATE_CACHE \
    --start 50 --limit 100 --context 1024 \
    --output piqa-50-149.json
```

The report holds the dataset SHA-256, package versions and every choice's loss. The source score does not depend
on the bundle, so it can be computed once per checkpoint and reused. Use `--start 0 --limit 1838` for the full split.

## Chat generation (IFEval, GSM8K)

`evaluate_chat` generates replies through the bundle's own chat template, with thinking off and greedy decoding, and
writes one JSON line per example, so an interrupted run keeps its results. Use the IFEval and GSM8K files from
[Datasets](#datasets).

```bash
python -m scripts.litert.benchmarks.evaluate_chat ifeval BUNDLE .cache/litert-eval/ifeval-input.jsonl \
    --cache-dir CACHE --output-tokens 1024 --output ifeval.jsonl
python -m scripts.litert.benchmarks.evaluate_chat gsm8k BUNDLE .cache/litert-eval/gsm8k-test.jsonl \
    --cache-dir CACHE --output-tokens 512 --output gsm8k.jsonl
```

For the BF16 reference, pass the staged checkpoint as the model and `--source-device cpu` (or `mps`) instead of
`--cache-dir`. It renders the same `quick_chat.jinja` template. Examples are shuffled with `--seed`, so `--start` and
`--limit` split the same order into disjoint shards. A full GSM8K run takes many hours per bundle on a laptop CPU, so
the published results were run in shards named `TASK-MODEL-START.jsonl`, for example:

```bash
for start in $(seq 0 64 1318); do
    python -m scripts.litert.benchmarks.evaluate_chat gsm8k BUNDLE .cache/litert-eval/gsm8k-test.jsonl \
        --cache-dir CACHE --output-tokens 512 --start "$start" \
        --limit "$(( 1319 - start < 64 ? 1319 - start : 64 ))" --output "runs/gsm8k-candidate-$start.jsonl"
done
```

IFEval reports strict accuracy. GSM8K asks for the final answer on a `####` line and reports exact match, how many
replies used that format, and how many had the right last number without hitting the output cap. Both report how
many replies hit the output cap. `--gsm8k-prompt final-only` asks for the answer alone, which is faster but a
different task.

## Comparing results

`summarize_results` compares models on the rows they all completed, so it also works while shards are still
running. It checks that every shard used the same dataset and settings and that all shards of one model ran the
same weights, records each model's SHA-256 and backend, and reports each pairwise difference in points with a paired
bootstrap 95% interval (20,000 resamples, seed 42):

```bash
python -m scripts.litert.benchmarks.summarize_results chat \
    --model bf16 'runs/gsm8k-bf16-*.jsonl' \
    --model candidate 'runs/gsm8k-candidate-*.jsonl' \
    --model qwen 'runs/gsm8k-qwen-*.jsonl'
python -m scripts.litert.benchmarks.summarize_results piqa piqa-source.json piqa-v5.json piqa-candidate.json
```

Quote the glob patterns so the script, not the shell, expands them. The published summaries are in
[`evidence/`](evidence/); add the summary of any new result there next to the QUALIFICATION.md entry that quotes it.

## Chat smoke tests

`verify_bundle` runs arithmetic, three-turn recall and long-prompt retrieval through the bundle's chat template. It
records time to first visible text, prefill and decode rates, peak process RSS and runtime cache size, and exits with
status 1 if a case fails.

```bash
python -m scripts.litert.checks.verify_bundle BUNDLE --cache-dir CACHE --context 1024
python -m scripts.litert.checks.verify_bundle BUNDLE --cache-dir CACHE --context 2048 --case long-prefill --long-entries 72
python -m scripts.litert.checks.verify_bundle BUNDLE --cache-dir :memory --context 1024 --case floor
```

The floor requires at least six correct answers with zero degenerate replies. The JSONL keeps every answer for
review. Run the standard suite a second time with the same disk cache to measure warm startup.

## CPU latency

`benchmark_cpu` runs LiteRT-LM's benchmark with fixed prefill and decode token counts and two CPU threads, and keeps
every run. The first run is a warm-up; the summary record holds the median of every metric over the later runs
(`warm_median`), which is what [QUALIFICATION.md](QUALIFICATION.md#cpu-latency) reports.

```bash
python -m scripts.litert.benchmarks.benchmark_cpu BUNDLE --cache-dir CACHE --context 1024 --prefill 128 --decode 64 \
    --output latency-128.jsonl
python -m scripts.litert.benchmarks.benchmark_cpu BUNDLE --cache-dir CACHE --context 1024 --prefill 768 --decode 64 \
    --output latency-768.jsonl
```

## Android

Use Android arm64 `litert_lm_advanced_main` from the LiteRT-LM source build described in Google's
[LiteRT-LM instructions](https://github.com/google-ai-edge/LiteRT-LM/blob/main/docs/getting-started/build-and-run.md)
or the corresponding [Google prebuilt runner](https://github.com/google-ai-edge/LiteRT-CLI#environment-variables).
The basic `litert_lm_main` runs a chat prompt and its output cannot be used for fixed-token timing. Enable USB
debugging and confirm that `adb devices` lists the phone. Set `runner_dir` to the directory containing the
advanced binary and its required shared libraries.

```bash
device_dir="/data/local/tmp/domyn-edge-$(date +%Y%m%d-%H%M%S)"
adb shell mkdir "$device_dir" &&
  adb push BUNDLE "$device_dir/model.litertlm" &&
  adb push "$runner_dir/litert_lm_advanced_main" "$runner_dir"/*.so "$device_dir/" &&
  adb shell chmod 755 "$device_dir/litert_lm_advanced_main" &&
  adb shell "sha256sum '$device_dir/model.litertlm'"
adb shell "LD_LIBRARY_PATH='$device_dir' '$device_dir/litert_lm_advanced_main' \
    --backend=cpu --model_path='$device_dir/model.litertlm' \
    --benchmark --async=false --max_num_tokens=1024 \
    --benchmark_prefill_tokens=768 --benchmark_decode_tokens=16 \
    --prefill_batch_sizes=128,1024 --num_cpu_threads=2 \
    --report_peak_memory_footprint" > "android-${device_dir##*/}-1024.log"
```

Compare the phone's SHA-256 with the host file before benchmarking. Check that the log reports the requested
context and actual token counts. For the runtime-limit comparison, change only `--max_num_tokens` to 2048 and
alternate which limit runs first in matched pairs. Keep all trial logs, including memory and thermal readings.
Import the bundle into Google AI Edge Gallery with **Default max tokens** set to 1,024 for the tested Android CPU setup.
A separate 2,048-token import supports longer prompts with the latency documented in the qualification report.
Verify offline multi-turn chat and record the device, OS, Gallery version, backend, context, and peak memory.
Record the runner's source commit or the prebuilt object generation and binary hash.

Phone numbers move a lot with the device's state. Before each set, plug the phone in, turn battery saver off, wait
for `adb shell dumpsys thermalservice` to report thermal status 0, and save `dumpsys battery` and
`dumpsys thermalservice` next to the results. The Android results in [QUALIFICATION.md](QUALIFICATION.md#android-phone-cpu-offline)
were not all taken this way.
