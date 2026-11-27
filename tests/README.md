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

The references are greedy completions recorded with HF transformers and `torch_varlen`, the training kernel. They
store the GPU architecture they were recorded on (`sm_80` for the current ones). On that architecture `torch_varlen`
must reproduce the recorded log-probabilities exactly; on any other, every implementation gets the same tolerance. Record
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

## Running the tests on AMD GPUs (ROCm)

`torch.cuda.is_available()` is true on ROCm, so the GPU tests run there too. Build the environment with the ROCm
build of the same PyTorch nightly instead of the CUDA one:

```bash
uv sync --no-group torch-nightly --group torch-nightly-rocm
```

As an example, on one GCD of an MI250X node:

```bash
srun --account=<account> --partition=<partition> --nodes=1 --ntasks=1 --gpus-per-node=1 --cpus-per-task=7 \
    --mem=60G --time=00:30:00 uv run --no-sync pytest -v -m "not integration" tests/
```

Known differences on MI250X with `torch==2.16.0.dev20261019+rocm10.1`:

- On ROCm, `varlen_attn` aligns the causal mask to the top left when queries are shorter than keys, where flash
  attention on CUDA aligns it to the bottom right. HF generation with `torch_varlen` is therefore wrong after the
  prompt: `test_hf_generation.py::test_chunked_prefill_matches_full_forward` fails, and so does the greedy decoding
  step of `test_deployed_checkpoint.py` for `torch_varlen`. Training, where both have the same length, is not
  affected.
- In `test_deployed_checkpoint.py`, `flex_attention` fails to compile for the 256-wide heads: the default Triton
  configuration needs more shared memory than the 64 KiB of the MI250X.
- With `PYTORCH_ALLOC_CONF=expandable_segments:True`, the default of the launch scripts, the first large allocation
  can hang with the ROCm 10.1 runtime on an older `amdgpu` kernel driver (seen with 6.10.5). Run them with
  `PYTORCH_ALLOC_CONF=` to disable it.
