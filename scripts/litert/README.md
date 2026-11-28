# HuggingFace to LiteRT-LM Conversion

[`convert.py`](convert.py) converts a DomynEdge HuggingFace checkpoint, such as
[`domyn/Domyn-Edge-SFT`](https://huggingface.co/domyn/Domyn-Edge-SFT) or the output of
[`scripts/huggingface/convert.py`](../huggingface/convert.py), into a single `.litertlm` bundle for offline CPU
inference with [LiteRT-LM](https://github.com/google-ai-edge/LiteRT-LM), for example on an Android phone.

## Overview

The script stages the checkpoint, traces it with [LiteRT Torch](https://github.com/google-ai-edge/litert-torch),
quantizes it and packages it:

| Setting | Value |
| --- | --- |
| Signatures | Prefill of 128 and 1,024 tokens, and decode |
| KV cache | 2,048 tokens, the bundle's limit for prompt and answer together |
| Decoder weights | INT4, blockwise (block 32), symmetric (OCTAV) |
| Embeddings | INT8, channelwise, in a separate embedder model |
| Activations | BF16 |
| Chat template | ChatML with the Domyn Edge system prompt, reasoning off (`CHAT_TEMPLATE` in `convert.py`) |
| Stop token | The tokenizer's end-of-turn token |
| Start token | None: LiteRT-LM prepends the start token to every prompt, and the chat template already opens with `<|im_start|>`, the SFT tokenizer's BOS |

Older checkpoints, including the released `Domyn-Edge-SFT`, describe attention with flash-style `window_sizes`. The
script stages them with the canonical `layer_types` and `sliding_window`, and the modeling from
[`assets/domyn-edge/hf`](../../assets/domyn-edge/hf), so the bundle is traced from the same code as the HF model.

`convert.py` changes three things in how LiteRT Torch runs, without changing the recipe or the bundle layout:

- **Prefill attention masks.** The masked attention logits are filled in float32 instead of BF16, so the fill stays
  a scalar instead of a dense constant per prefill signature and layer (289 MiB on `Domyn-Edge-SFT`).
- **Quantization** streams the quantized model to disk instead of building it in memory.
- **TFLite compilation** runs in a separate process: TensorFlow and the MLIR library LiteRT Torch traces with both
  bundle LLVM, and loading both in one process aborts on macOS arm64.

After packaging, the script reads the bundle metadata back and fails if the chat template, the start and stop
tokens or the output channels are not the expected ones.

## Results

On the 20 continuations in [`tests/assets/litert_continuation_cases.jsonl`](../../tests/assets/litert_continuation_cases.jsonl)
(259 scored tokens), the INT4 bundle of `Domyn-Edge-SFT` (revision `bfdad89`) adds 0.169 nats per token to the loss
of the BF16 source: perplexity goes from 6.63 to 7.85.

The same bundle, next to the
[Qwen3-4B LiteRT-LM bundle](https://huggingface.co/litert-community/Qwen3-4B) (`qwen3_4b_mixed_int4.litertlm`,
revision `65314f6`), on a Galaxy S23 (Snapdragon 8 Gen 2, 8 GB RAM, Android 17), offline:

| | Domyn-Edge-SFT INT4 | Qwen3-4B INT4 |
| --- | ---: | ---: |
| Bundle size | 2.35 GiB | 2.12 GiB |
| Decode, 128 / 768 prompt tokens | 5.8 / 5.5 tokens/s | 7.9 / 7.9 tokens/s |
| Prefill, 128 / 768 prompt tokens | 24.1 / 21.7 tokens/s | 22.2 / 22.1 tokens/s |
| Time to first token, 128 / 768 prompt tokens | 5.5 / 35.6 s | 5.9 / 34.9 s |
| Peak memory, 128 / 768 prompt tokens | 3,354 / 3,320 MB | 2,871 / 2,963 MB |

Both bundles ran with LiteRT-LM's Android arm64 `litert_lm_advanced_main`, on the CPU backend with two threads, a
1,024-token context and 32 decoded tokens. Each figure is the median of four runs after a warm-up run, with the phone
charging and at thermal status 0 before every run. In Google AI Edge Gallery 1.0.23 on the same phone, offline, the
built-in benchmark (CPU, 256 prompt and 256 decoded tokens, mean of three runs) gave 3.6 and 4.0 tokens/s decode in two
sets for Domyn-Edge-SFT and 4.4 tokens/s for Qwen3-4B.

## Installation

The export needs LiteRT Torch's `ModelExtension` API, which is not in a release yet (the latest, 0.10.0, predates
it), so the toolchain uses LiteRT Torch from a pinned commit and nightly builds of LiteRT and TensorFlow. Install it
in a separate Python 3.13 environment. These are the versions the conversion and its tests were run with, on macOS
arm64:

```bash
python3.13 -m venv .venv-litert
source .venv-litert/bin/activate
pip install torch==2.14.0 transformers==5.22.2 click rich \
    tf_nightly==2.23.0.dev20261109 keras-nightly==3.16.0.dev2026111003 \
    ai-edge-litert-nightly==2.4.0.dev20261108 litert-converter==0.6.0.dev20261106 \
    litert-quantizer-nightly==0.12.0.dev20261110 litert-lm==0.19.1 \
    jax==0.12.0.1 jaxtyping==0.3.11 torchao==0.18.0 typer==0.28.0 tabulate==0.10.0 fire==0.7.1 \
    kagglehub==1.0.3 multipledispatch==1.0.0 sentencepiece==0.2.3 pillow==13.0.0
pip install --no-deps \
    https://github.com/google-ai-edge/litert-torch/archive/13644e33d20cb18b48c7e5fbb2698fa5795740bf.tar.gz
```

The second line of packages are LiteRT Torch's own requirements, which `--no-deps` does not install; pinning them
keeps pip from replacing the nightly builds above.

## Usage

```bash
hf download domyn/Domyn-Edge-SFT --revision bfdad8935eede2977b2293b9754597529b21e55c --local-dir Domyn-Edge-SFT
python scripts/litert/convert.py \
    --model-dir Domyn-Edge-SFT \
    --output-dir Domyn-Edge-SFT-litert \
    --assets-dir assets/domyn-edge/hf
```

The output directory holds `model.litertlm` and `hf/`, the staged checkpoint the bundle was exported from. Its
weights are symbolic links to `--model-dir`, and it loads with `transformers` (`trust_remote_code=True`) to compare
the two. The export of `Domyn-Edge-SFT` peaks at 19 to 21 GiB of RAM, and the script refuses to start with less than
four times the checkpoint's weights plus 2 GiB free on the output volume.

### Parameters

- `--model-dir`: Directory with the HF checkpoint (`config.json`, safetensors weights and tokenizer files)
- `--output-dir`: New directory for the bundle; the script refuses an existing one
- `--assets-dir`: Directory containing `modeling_domynedge.py` and `configuration_domynedge.py` (`assets/domyn-edge/hf`
  in this repo)
- `--cache-length`: KV cache length, the most prompt and generated tokens one conversation can hold (default: 2048)
- `--prefill-lengths`: Prefill signature lengths, at least 128; repeat the option for several (default: 128 and 1024)

## Running the bundle

With the `litert-lm` package installed above:

```python
import pathlib

import litert_lm

# The first load writes a weight cache of about the bundle's size here; later loads reuse it.
cache_dir = pathlib.Path("litert-cache")
cache_dir.mkdir(exist_ok=True)
with litert_lm.Engine(
    "Domyn-Edge-SFT-litert/model.litertlm",
    backend=litert_lm.Backend.CPU(thread_count=2),
    max_num_tokens=2048,
    cache_dir=str(cache_dir),
) as engine:
    with engine.create_conversation() as chat:
        reply = chat.send_message("What is the capital of France? Answer briefly.")
        print("".join(item.text for item in reply.contents if hasattr(item, "text")))
```

On Android, load the bundle in [Google AI Edge Gallery](https://github.com/google-ai-edge/gallery) or with the
LiteRT-LM runtime, on the CPU backend. Use the CPU backend on desktop too: the macOS WebGPU backend rejects the BF16
tensors in the bundle's attention.

## Tests

[`tests/test_litert_conversion.py`](../../tests/test_litert_conversion.py) checks the staging and the options on CPU,
checks that an unquantized bundle of a random 4-layer DomynEdge gives the HF model's continuation loss, and measures
the continuation loss the INT4 bundle of the released checkpoint adds over its BF16 source. See
[tests/README.md](../../tests/README.md).
