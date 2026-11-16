---
license: apache-2.0
base_model: domyn/Domyn-Edge-SFT
base_model_relation: quantized
pipeline_tag: text-generation
tags:
- litert-lm
- litertlm
- on-device
- int4
---

# Domyn Edge SFT LiteRT-LM

This card describes the release candidate built from
[Domyn Edge SFT at `bfdad8935eede2977b2293b9754597529b21e55c`](https://huggingface.co/domyn/Domyn-Edge-SFT/tree/bfdad8935eede2977b2293b9754597529b21e55c).
The conversion uses post-training quantization of the published weights. It uses blockwise INT4 decoder weights
(block size 32), an INT8 embedder, 128- and 1,024-token prefill signatures, and an exported 2,048-token cache.

## Candidate file

| Property | Value |
| --- | --- |
| File | `domyn-edge-sft-v5-compact-int4.litertlm` |
| Size | 2,520,888,320 bytes (2.35 GiB) |
| SHA-256 | `31425c2ce528f08cb24748add0da99dadf2106437843252467017464c670ea18` |
| Target | CPU inference on edge devices running LiteRT-LM (phones, tablets, embedded boards), offline |
| Tested mobile environment | Android 16 phone, Snapdragon 8 Gen 2 (arm64), 8 GB RAM, Google AI Edge Gallery 1.0.23, CPU |
| Tested desktop runtime | LiteRT-LM 0.19.1, macOS arm64, CPU |

## Run offline in Google AI Edge Gallery

Open **Menu → Models → Import model → Import model from local file**. Select the `.litertlm` file and set
**Default max tokens** to **1,024** for CPU on a phone with 8 GB of RAM. Open the imported model in **AI Chat** and select **CPU**.
The token budget includes the prompt, chat history, and answer. A separate import with **2,048** max tokens
supports longer prompts and can be substantially slower. The tested Gallery installation also wrote a 2.07 GiB
weight cache beside the model, for 4.42 GiB across those files.

## Qualification and limits

Offline Gallery testing verified short answers, multi-turn recall, and a 1,909-token retrieval prompt. Near the
2,048-token limit, that prompt took 134.9 seconds to first token. The bundle's cache holds 2,048 tokens, so do not
set a larger limit. In four native CPU pairs with 768 input and 16
output tokens, the medians at 1,024 and 2,048 total tokens were 42.8 and 50.8 seconds to first token, and 4.28 and
2.30 tokens per second during decode. These are measurements on one Android phone. The supported scope is text chat
with extended thinking off. GPU, iOS, tool calling, and other Android devices remain unqualified.

The desktop CPU smoke floor passed at 7/8, and the three-turn recall test passed. On the full GSM8K test split
(1,319 problems, thinking off), the bundle solves 85.1% against 89.2% for the BF16 source (−4.1 points, 95%
interval −5.8 to −2.4), level with the Qwen3-4B LiteRT-LM INT4 bundle (84.2%). The GPU backend is not supported:
the bundle's BF16 activations are rejected by the tested GPU delegate. The [qualification notes](../QUALIFICATION.md) record the
protocol, toolchain, supporting results, and outstanding release gates.

The source weights and attribution belong to the Domyn Edge authors. Distribute the bundle with the Apache-2.0
license and retain applicable attribution. The [source model card](https://huggingface.co/domyn/Domyn-Edge-SFT)
provides the original citation and model limitations.
