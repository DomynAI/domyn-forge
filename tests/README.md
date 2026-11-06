# Tests

```bash
uv run pytest
```

Tests that need a GPU are skipped when CUDA is not available, so in CI
(the pre-commit `pytest` hook) only covers the CPU tests:

| File | CPU | GPU only |
| --- | --- | --- |
| [`test_domynedge.py`](test_domynedge.py) | Flavor configs, weight tying, state dict adapter, `update_from_config` checks, sharding configs | Forward: varlen vs flex parity (logits and gradients), causality, packed documents, sliding window |
| [`test_token_datasets.py`](test_token_datasets.py) | Pretokenized `*.arrow` dataset and dataloader: causal shift and padding, loss masking with and without `token_type_ids`, shuffling, exact resume, data-parallel splits, `drop_last`, `max_docs_per_batch` | — |
| [`test_hf_conversion.py`](test_hf_conversion.py) | Converted config: head sizes, RoPE, layer types and sliding window | DCP -> HF conversion gives the torchtitan residual stream and logits, with every attention implementation; packed documents |
| [`test_hf_generation.py`](test_hf_generation.py) | — | HF model behaviour on the debug model: chunked prefill with and without a mask, hidden states, `use_cache` and `return_dict` |
| [`test_deployed_checkpoint.py`](test_deployed_checkpoint.py) | — | The released `domyn/Domyn-Edge-SFT` weights, converted with the current modeling, reproduce the recorded reference completions with every HF attention implementation and with vLLM (see below) |
| [`test_dataloader_restore.py`](test_dataloader_restore.py) | — | **Integration, 4 GPUs.** DomynEdge training resumed from a checkpoint reproduces the batches, losses and validation batches of an uninterrupted run (varlen and flex) |

Integration tests are marked `integration` and are skipped unless 4 GPUs are visible. Select or exclude them with
`-m integration` / `-m "not integration"`.

## Deployed checkpoint

[`test_deployed_checkpoint.py`](test_deployed_checkpoint.py) needs `domyn/Domyn-Edge-SFT` in the local Hugging Face
cache, at the previously recorded completions in
[`assets/domyn_edge_sft_completions.json`](assets/domyn_edge_sft_completions.json).
vLLM uses a separate environment built from the `vllm` dependency group.

Create the vLLM environment once:

```bash
UV_PROJECT_ENVIRONMENT=.venv-vllm uv sync --group vllm --no-group torch-nightly
```

Then, on a GPU node:

```bash
# HF transformers, every attention implementation (vLLM is skipped: it is not installed here)
uv run pytest tests/test_deployed_checkpoint.py -k hf -rfs
# vLLM
UV_PROJECT_ENVIRONMENT=.venv-vllm uv run --no-sync pytest tests/test_deployed_checkpoint.py -k vllm -rfs
```

The references are greedy completions recorded with HF transformers and `torch_varlen`, the training kernel. Record
them again after changing the prompts, the tokenizer or the checkpoint:

```bash
uv run python tests/test_deployed_checkpoint.py
```

## Running the GPU tests on Leonardo

As an example, to run the GPU tests on [Leonardo HPC](https://docs.hpc.cineca.it/hpc/leonardo.html#leonardo-card),
you can use the following `srun` command:

```bash
srun --account=<account> --partition=boost_usr_prod --qos=boost_qos_dbg \
    --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --time=00:30:00 \
    bash -c 'module load gcc/12.2.0 cuda/12.6 && uv run --no-sync pytest -v -m "not integration" tests/'
```

The integration tests need a node with 4 GPUs:

```bash
srun --account=<account> --partition=boost_usr_prod --qos=boost_qos_dbg \
    --nodes=1 --ntasks=1 --gres=gpu:4 --cpus-per-task=32 --time=00:30:00 \
    bash -c 'module load gcc/12.2.0 cuda/12.6 && uv run --no-sync pytest -v -m integration tests/'
```
