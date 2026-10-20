<div align="center">

<picture>
    <img src="static/domyn-forge-logo-light.svg#gh-light-mode-only" alt="Domyn Forge" height="100">
    <img src="static/domyn-forge-logo-dark.svg#gh-dark-mode-only" alt="Domyn Forge" height="100">
</picture>


**Everything you need to continue pretraining or fine-tune DomynEdge with TorchTitan.**

Domyn Forge collects the artifacts behind DomynEdge: the model definition, tokenizers, a pretokenized dataloader,
run configs and Hugging Face conversion, so everybody can keep training it.
It runs using stock [TorchTitan](https://github.com/pytorch/torchtitan), pinned as a dependency, and has been
tested on the [Leonardo](https://www.hpc.cineca.it/systems/hardware/leonardo/) HPC cluster (CINECA).

[![Python](https://img.shields.io/badge/python-3.13-brightgreen)](pyproject.toml)
[![TorchTitan](https://img.shields.io/badge/torchtitan-v0.3.0-orange)](https://github.com/pytorch/torchtitan/releases/tag/v0.3.0)
[![License - Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue)](LICENSE)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Pyright](https://microsoft.github.io/pyright/img/pyright_badge.svg)](https://github.com/microsoft/pyright)

<a href="#quick-start">Quick start</a> •
<a href="#continue-training-or-fine-tune">Continue training</a> •
<a href="#data-format">Data format</a> •
<a href="#export-to-hugging-face">Export</a> •
<a href="#development">Development</a>

</div>

> [!WARNING]
> Domyn Forge is at an early stage of development and has not been used during production, and TorchTitan, which it builds on, is evolving quickly.
> Interfaces, configs and defaults may change between versions, and some combinations of features are not tested yet.

---

## What's included

| | |
| --- | --- |
| 🧠 **Model** | DomynEdge for TorchTitan: hybrid sliding-window / global attention with per-layer head size, QK norm and tied embeddings. Flavors: `edgethinker` (full model) and `debugmodel` (small, for smoke tests). Varlen and flex attention backends. |
| 🔤 **Tokenizers** | The pretraining, midtraining and post-training tokenizers in [`assets/domyn-edge/tokenizers/`](assets/domyn-edge/tokenizers/). |
| 📦 **Data** | A map-style dataloader for pretokenized `*.arrow` shards: memory-mapped, shuffled, exactly resumable from checkpoints, with per-role loss masking for fine-tuning. |
| 🚀 **Training** | Example files to launch on one node with [`train.sh`](scripts/launch/train.sh) or on several with Slurm through [`train.sbatch`](scripts/launch/train.sbatch). |
| 🤗 **Export** | Conversion from TorchTitan checkpoints to Hugging Face format, with the modeling code, and a generation script to try the result. |
| ✅ **Tests** | CPU unit tests for the model, adapter and dataloader, plus GPU and 4-GPU integration tests ([tests/README.md](tests/README.md)). |

---

## Quick start

```bash
uv sync
# Tokenize and pack TorchTitan's c4_test sample into data/, which debug.py reads.
uv run scripts/data/pack_c4_test.py
NGPU=2 ./scripts/launch/train.sh scripts/launch/configs/domynedge/debug.py --training.steps 5
```

---

## Continue training or fine-tune

A run is a single Python file in [`scripts/launch/configs/`](scripts/launch/configs/) that defines `TRAIN_CONFIG`,
the full TorchTitan `Trainer.Config`, and calls `run(TRAIN_CONFIG)`. Start from
[`domynedge/edgethinker.py`](scripts/launch/configs/domynedge/edgethinker.py) and fill in:

1. **Tokenizer** — `hf_assets_path`, pointing at the tokenizer matching your stage in
   [`assets/domyn-edge/tokenizers/`](assets/domyn-edge/tokenizers/).
2. **Data** — `dataloader.dataset_path`, a directory of pretokenized shards (see [Data format](#data-format)).
   For fine-tuning, set `dataloader.loss_enabled_turns` to the roles to train on (default: `["assistant"]`).
   To use TorchTitan's default text dataloader instead, replace `dataloader` in the config with
   `HuggingFaceTextDataLoader.Config(...)` from `torchtitan.hf_datasets.text_datasets`; it tokenizes raw text on the
   fly, so it does not need pretokenized shards (and has no per-role loss masking).
3. **Starting weights** — either a TorchTitan (DCP) checkpoint, or a Hugging Face checkpoint loaded with
   `checkpoint.initial_load_path` and `checkpoint.initial_load_in_hf=True`.
   Remember that a Hugging Face export keeps only the tokenizer's rows of the embeddings (115269 for the bundled
   tokenizers), while `edgethinker` trains with 115328: build the model with the export's vocabulary size, e.g.
   `model_registry("edgethinker", vocab_size=115269)`, or the load fails with a size mismatch.

Then launch it on one node, overriding any field from the command line:

```bash
NGPU=8 ./scripts/launch/train.sh scripts/launch/configs/domynedge/edgethinker.py \
    --training.steps 100 \
    --parallelism.tensor_parallel_degree 2
```

or on several nodes with Slurm, submitting from the repository root (adjust the `#SBATCH` lines in
[`train.sbatch`](scripts/launch/train.sbatch) for your cluster):

```bash
sbatch --account=<account> --nodes=4 scripts/launch/train.sbatch \
    scripts/launch/configs/domynedge/edgethinker.py --training.steps 100
```

| Config | Model flavor | Purpose |
| --- | --- | --- |
| [`domynedge/debug.py`](scripts/launch/configs/domynedge/debug.py) | `debugmodel` | Small model for smoke tests |
| [`domynedge/edgethinker.py`](scripts/launch/configs/domynedge/edgethinker.py) | `edgethinker` | Template for a full EdgeThinker run |

---

## Data format

The dataloader reads every `*.arrow` shard in `dataset_path` (for example the output of `Dataset.save_to_disk`).
Each row is one packed training sample of at most `seq_len + 1` tokens:

| Column | Required | Content |
| --- | --- | --- |
| `input_ids` | yes | Token ids; several documents can be packed in one row. |
| `position_ids` | yes | Position of each token in its document, restarting at 0 at every document start. Attention never crosses a restart. |
| `token_type_ids` | no | Role of each token: `1` system, `2` user, `3` assistant, `4` tool. Only roles in `loss_enabled_turns` are trained on. |

Without `token_type_ids` every token counts as `assistant`, so pretraining data trains on every token with the
default settings. Shorter rows are right-padded, and padding never contributes to the loss.

---

## Export to Hugging Face

[`convert.py`](scripts/huggingface/convert.py) turns a TorchTitan checkpoint into a Hugging Face model with
custom modeling code. Point `--model-dir` at one checkpoint step, and use the tokenizer the run trained with:

```bash
uv run scripts/huggingface/convert.py \
    --model-dir /path/to/dump_folder/checkpoint/step-1000 \
    --tokenizer-dir assets/domyn-edge/tokenizers/pretraining \
    --output-dir hf-checkpoints/edgethinker \
    --assets-dir assets/domyn-edge/hf \
    --model-name domynedge \
    --model-flavor edgethinker \
    --dtype bfloat16 \
    --max-seq-len 32768
```

If the run overrode the flavor's vocabulary size (for example [`debug.py`](scripts/launch/configs/domynedge/debug.py),
which trains `debugmodel` with the full 115328-token vocabulary), pass it with `--model-vocab-size`.

Then try the converted model with [`generate.py`](scripts/huggingface/generate.py):

```bash
uv run scripts/huggingface/generate.py --input-dir hf-checkpoints/edgethinker --prompt "The capital of Italy is"
```

> [!CAUTION]
> With recent PyTorch nightlies, generation with the KV cache (the `transformers` default) degenerates after the
> first new token, because `varlen_attn` now aligns the causal mask of a single decoding query to the first cached
> key. Full forward passes are not affected. Until the modeling code is fixed, check generations with
> `model.generate(..., use_cache=False)`.

See [scripts/huggingface/README.md](scripts/huggingface/README.md) for all options.

---

## Development

```bash
uv sync
uv run pre-commit install
uv run pytest
```

The pre-commit hooks ([`.pre-commit-config.yaml`](.pre-commit-config.yaml)) run `ruff format`, `ruff check`,
`pyright` and `pytest` on every commit, plus whitespace, YAML/TOML, large-file and conventional-commit checks.
To run them on the whole tree:

```bash
uv run pre-commit run --all-files
```

GPU tests are skipped without CUDA, including in pre-commit and CI. See [tests/README.md](tests/README.md)
for what each test covers and how to run the GPU tests on Leonardo with Slurm.

TorchTitan is pinned in `[tool.uv.sources]` in [`pyproject.toml`](pyproject.toml).

---

## License

Domyn Forge is released under the [Apache License 2.0](LICENSE).
