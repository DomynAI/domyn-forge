# TorchTitan to HuggingFace Model Conversion

This directory contains scripts to convert trained torchtitan models to the HuggingFace transformers format,
enabling easy model sharing, inference, and integration with the HuggingFace ecosystem.

## Overview

The conversion process transforms torchtitan model checkpoints (stored in PyTorch Distributed Checkpoint format)
into fully compatible HuggingFace models that can be used with standard transformers tools and libraries.

## Files

- `convert.py` - Main conversion script that converts torchtitan checkpoints to HuggingFace format
- `generate.py` - Text generation script for testing converted models
- `assets/domyn-edge/hf/modeling_domynedge.py` (repo root) - HuggingFace-compatible model implementation
- `assets/domyn-edge/hf/configuration_domynedge.py` (repo root) - HuggingFace-compatible configuration class

## Usage

### 1. Model Conversion

Convert a trained torchtitan model to HuggingFace format:

```bash
uv run convert.py \
    --model-dir [MODEL PATH] \
    --tokenizer-dir [TOKENIZER PATH] \
    --output-dir hf-checkpoints/test \
    --assets-dir ../../assets/domyn-edge/hf \
    --model-flavor edgethinker \
    --model-name domynedge \
    --max-seq-len 8192 \
    --dtype bfloat16 \
    --overwrite
```

#### Required Parameters

- `--model-dir`: Directory containing the torchtitan DCP checkpoint
- `--tokenizer-dir`: Directory containing tokenizer configuration and vocabulary files
- `--output-dir`: Directory where the converted HuggingFace model will be saved
- `--assets-dir`: Directory containing `modeling_domynedge.py` and `configuration_domynedge.py` (`assets/domyn-edge/hf` in this repo)
- `--model-name`: Name of the model specification.
- `--model-flavor`: Specific model variant/flavor. This should be `edgethinker` for the current domynedge model, but this option is left open for ablation purposes and future variants.
- `--max-seq-len`: Context length the checkpoint was trained with, written to `config.json` as `max_position_embeddings` (e.g. `8192` for stage-1 pretraining, `262144` for Domyn-Edge-Base).

#### Optional Parameters

- `--dtype`: Target data type for converted weights (`float16`, `bfloat16`, `float32`). Attention is always performed in 16-bit precision as required by flash attention.
- `--overwrite`: Overwrite output directory if it exists (default: False)
- `--vocab-size`: Number of vocabulary rows to export (default: the tokenizer's vocabulary size)

### 2. Text Generation

Test the converted model with text generation:

```bash
python generate.py \
    --input-dir hf-checkpoints/test \
    --prompt "The capital of Italy is" \
    --device cuda \
    --max-length 100 \
    --dtype bfloat16
```

#### Parameters

- `--input-dir`: Directory containing the converted HuggingFace model
- `--prompt`: Text prompt for generation
- `--device`: Device to run on (`cpu` or `cuda`, default: `cuda`)
- `--max-length`: Maximum length of generated text (default: 100)
- `--dtype`: Data type for model weights (default: `bfloat16`)

## Example Workflow

```bash
# 1. Convert trained model
uv run scripts/huggingface/convert.py \
    --model-dir /path/to/checkpoint/step-N/model \
    --tokenizer-dir assets/domyn-edge/tokenizers/pretraining \
    --output-dir hf-checkpoints/test \
    --assets-dir assets/domyn-edge/hf \
    --model-flavor edgethinker \
    --model-name domynedge \
    --max-seq-len 8192 \
    --dtype bfloat16 \
    --overwrite

# 2. Test the converted model
uv run scripts/huggingface/generate.py \
    --input-dir hf-checkpoints/test \
    --max-new-tokens 512 \
    --prompt "The capital of Italy is"
```

## Notes

### Determinism

Running the generation script with different batch compositions may yield different outputs.
For example, if you run the generation with `--prompt "The capital of Italy is"` then you are expecteed to get always the same output.
However, if you run the generation with `--prompt "The capital of Italy is" --prompt "The capital of Germany is"` then the output for the first prompt may differ from the previous case.
