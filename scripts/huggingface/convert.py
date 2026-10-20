# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Torchtitan to HuggingFace Model Conversion Script.

This script converts trained torchtitan models to the HuggingFace transformers format.
It handles conversion of model weights, tokenizer files, configuration, and generation
parameters to create a fully compatible HuggingFace model checkpoint.

The conversion process includes:
- Loading torchtitan model weights from DCP checkpoint format
- Converting state dict to HuggingFace format using model-specific adapters
- Saving weights in HuggingFace safetensors format
- Copying and configuring tokenizer files
- Generating model configuration and generation config files
- Including modeling and configuration classes for custom model architectures

Usage:
    python convert.py \
        --model-dir <path-to-dcp-checkpoint> \
        --tokenizer-dir <path-to-tokenizer> \
        --output-dir <output-directory> \
        --assets-dir <path-to-hf-modeling-files> \
        --model-name <model-name> \
        --model-flavor <model-flavor> \
        --dtype <target-dtype>

Example usage is provided at the end of this file.
"""

import importlib
import json
import math
import pathlib
import shutil
import warnings

import click
import rich
import rich.console
import rich.json
import rich.panel
import torch
import torch.distributed.checkpoint as dcp
import transformers


def embedding_init_std(embedding_config):
    """Measure the std the model config initializes embedding rows with.

    The configured init is applied to a probe rather than read structurally:
    `param_init["weight"]` is an arbitrary callable, and DomynEdge's is a
    truncated normal whose realized std sits a little below its nominal `std=`.
    Seeded so the exported config.json does not move between runs.
    """
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        probe = torch.empty(1024, embedding_config.embedding_dim)
        embedding_config.param_init["weight"](probe)
    return probe.std().item()


def resolve_vocab_size(vocab_size, tok_vocab, orig_vocab):
    """Decide how many vocabulary rows to export, and validate the choice.

    `vocab_size` is the training loss's own vocab when the caller knows it;
    None falls back to len(tokenizer), which is what this script always used to
    do. save_model and save_config both call this so the exported weights and
    the exported config.json can never disagree about the width.
    """
    keep = tok_vocab if vocab_size is None else int(vocab_size)
    if keep <= 0:
        msg = f"vocab_size must be positive, got {keep}."
        raise ValueError(msg)
    if orig_vocab is not None and keep > orig_vocab:
        msg = f"Requested vocab size ({keep}) exceeds checkpoint vocab size ({orig_vocab})."
        raise ValueError(msg)
    if keep > tok_vocab:
        msg = (
            f"Requested vocab size ({keep}) exceeds tokenizer vocab size ({tok_vocab}); "
            "the extra rows could never be produced by the tokenizer."
        )
        raise ValueError(msg)
    return keep


def warn_on_untrained_embedding_rows(console, embedding, tokenizer, vocab_size, init_std):
    """Warn if the kept embedding rows include a tail that never trained.

    Detection is exact rather than heuristic: an untrained row sits at
    init_std * sqrt(hidden_size), several times below any trained row.

    Args:
        console: Rich console used to print the warning.
        embedding: Token embedding weight, already truncated to vocab_size rows.
        tokenizer: Tokenizer used to decode the ids of the untrained rows.
        vocab_size: Number of exported vocabulary rows.
        init_std: std the model config initializes the embedding with, from
            embedding_init_std.
    """
    norms = embedding.float().norm(dim=-1)
    expected_init = init_std * math.sqrt(embedding.shape[1])
    threshold = expected_init * 1.5

    # Walk back from the cut, collecting only a contiguous run: an isolated
    # low-norm row further down is a rare token, not an untrained one.
    first_untrained = vocab_size
    while first_untrained > 0 and norms[first_untrained - 1] < threshold:
        first_untrained -= 1
    if first_untrained >= vocab_size:
        return

    ids = list(range(first_untrained, vocab_size))
    names = [repr(tokenizer.decode([i])) for i in ids]
    console.print(
        rich.panel.Panel.fit(
            f"[bold]{len(ids)} exported embedding rows never trained[/bold]\n\n"
            f"ids {ids[0]}-{ids[-1]}: {', '.join(names)}\n\n"
            f"Their norms are ~{norms[ids[0]]:.3f}, i.e. initialization "
            f"({init_std:.4f} * sqrt({embedding.shape[1]}) = {expected_init:.3f}); "
            f"trained rows here median {norms[:first_untrained].median():.3f}.\n"
            f"They will be sampled at inference. Re-run with "
            f"--vocab-size {first_untrained} to drop them.",
            title="[red]untrained vocabulary[/red]",
            border_style="red",
        )
    )


def save_model(
    console,
    tokenizer_dir,
    model_dir,
    output_dir,
    dtype,
    model_name,
    model_flavor,
    vocab_size=None,
    model_vocab_size=None,
):
    """Convert and save torchtitan model weights to HuggingFace format.

    Args:
        console: Rich console for progress output
        tokenizer_dir: Path to tokenizer directory
        model_dir: Path to DCP checkpoint directory
        output_dir: Path where HF model will be saved
        dtype: Target data type for weights (float16/bfloat16/float32)
        model_name: Name of model specification
        model_flavor: Specific model variant
        vocab_size: Number of vocabulary rows to keep.
        model_vocab_size: Vocabulary size the checkpoint was trained with, if it differs from the flavor default.

    Raises:
        ValueError: If model_flavor is not found
    """
    warnings.filterwarnings("ignore", category=UserWarning, module="torch.distributed")

    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)
    console.print("Loading model...", end="")
    model_spec = importlib.import_module(f"domyn_forge.models.{model_name}").model_registry(
        model_flavor, vocab_size=model_vocab_size
    )
    with torch.device("cpu"):
        model = model_spec.model.build()
    console.print("done.")

    sd_adapter = model_spec.state_dict_adapter(model_spec.model, None)
    model_state_dict = model.state_dict()
    console.print("Loading DCP checkpoint...", end="")
    torch.distributed.checkpoint.load(model_state_dict, checkpoint_id=model_dir)
    console.print("done.")

    # Dropping padded vocabs
    orig_vocab = model_state_dict["lm_head.weight"].shape[0]
    tok_vocab = len(tokenizer)
    keep_vocab = resolve_vocab_size(vocab_size, tok_vocab, orig_vocab)

    console.print("Dropping padded vocab tokens from state dict...")
    console.print(f"Original vocab size: {orig_vocab}")
    console.print(f"Tokenizer vocab size: {tok_vocab}")
    console.print(f"Keeping vocab size: {keep_vocab}")
    model_state_dict["lm_head.weight"] = model_state_dict["lm_head.weight"][:keep_vocab]
    model_state_dict["tok_embeddings.weight"] = model_state_dict["tok_embeddings.weight"][:keep_vocab]
    warn_on_untrained_embedding_rows(
        console,
        model_state_dict["tok_embeddings.weight"],
        tokenizer,
        keep_vocab,
        embedding_init_std(model_spec.model.tok_embeddings),
    )
    console.print("Converting state dict to HF format...", end="")
    hf_state_dict = sd_adapter.to_hf(model_state_dict)
    console.print("done.")

    hf_state_dict = {
        k: v.to({"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[dtype])
        for k, v in hf_state_dict.items()
    }

    console.print("Saving HF safetensors checkpoint...", end="")

    if any(p.is_dir() for p in output_dir.iterdir() if p.name != "sharded"):
        # Check for unexpected subdirectories in output_dir (other than 'sharded')
        # Sanity check to prevent accidental deletion of important data
        msg = f"Output directory {output_dir} contains unexpected subdirectories. Only 'sharded' is allowed."
        raise ValueError(msg)

    shutil.rmtree(output_dir, ignore_errors=True)
    storage_writer = dcp.HuggingFaceStorageWriter(
        path=output_dir,
        save_distributed=True,
        fqn_to_index_mapping=None,
        enable_consolidation=True,
        thread_count_consolidation=5,
    )
    dcp.save(hf_state_dict, storage_writer=storage_writer)
    console.print("done.")

    warnings.resetwarnings()


def save_tokenizer(console, tokenizer_dir, output_dir):
    """Copy tokenizer configuration and vocabulary files to output directory.

    Args:
        console: Rich console for progress output
        tokenizer_dir: Path to source tokenizer directory
        output_dir: Path where tokenizer files will be saved

    Raises:
        FileNotFoundError: If required tokenizer files are missing
    """
    console.print("Copying tokenizer files...", end="")
    shutil.copyfile(tokenizer_dir / "special_tokens_map.json", output_dir / "special_tokens_map.json")
    shutil.copyfile(tokenizer_dir / "tokenizer_config.json", output_dir / "tokenizer_config.json")
    shutil.copyfile(tokenizer_dir / "tokenizer.json", output_dir / "tokenizer.json")
    shutil.copyfile(tokenizer_dir / "generation_config.json", output_dir / "generation_config.json")
    if (tokenizer_dir / "chat_template.jinja").exists():
        shutil.copyfile(tokenizer_dir / "chat_template.jinja", output_dir / "chat_template.jinja")
    console.print("done.")


def save_modeling(console, output_dir, assets_dir):
    """Copy custom model implementation files to output directory.

    Copies the modeling and configuration Python modules that implement
    the custom DomynEdge architecture for HuggingFace compatibility.

    Args:
        console: Rich console for progress output
        output_dir: Path where modeling files will be saved
        assets_dir: Directory containing the modeling files
    """
    assets_dir = pathlib.Path(assets_dir)
    console.print("Saving modeling file...", end="")
    shutil.copyfile(assets_dir / "modeling_domynedge.py", output_dir / "modeling_domynedge.py")
    shutil.copyfile(assets_dir / "configuration_domynedge.py", output_dir / "configuration_domynedge.py")
    pathlib.Path(output_dir / "__init__.py").touch()
    console.print("done.")


def save_config(console, tokenizer_dir, output_dir, model_name, model_flavor, dtype, max_seq_len, vocab_size=None):
    """Generate and save HuggingFace model configuration file.

    Creates config.json with model architecture parameters extracted from
    torchtitan model arguments and HuggingFace metadata.

    Args:
        console: Rich console for progress output
        tokenizer_dir: Path to tokenizer directory
        output_dir: Path where config will be saved
        model_name: Name of model specification
        model_flavor: Specific model variant
        dtype: Target data type for model weights
        max_seq_len: Maximum sequence length the model config is built with
        vocab_size: Must match what save_model kept
    """
    console.print("Saving config file...", end="")

    model_config = (
        importlib.import_module(f"domyn_forge.models.{model_name}")
        .model_registry(model_flavor, max_seq_len=max_seq_len)
        .model
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_dir)

    pathlib.Path(output_dir / "config.json").write_text(
        json.dumps(
            {
                "architectures": ["DomynEdgeForCausalLM"],
                "attention_dropout": 0.0,
                "auto_map": {
                    "AutoConfig": "configuration_domynedge.DomynEdgeConfig",
                    "AutoModel": "modeling_domynedge.DomynEdgeModel",
                    "AutoModelForCausalLM": "modeling_domynedge.DomynEdgeForCausalLM",
                },
                "dtype": dtype,
                "hidden_act": "silu",
                "hidden_size": model_config.tok_embeddings.embedding_dim,
                "intermediate_size": model_config.layers[0].feed_forward.w1.out_features,
                "max_position_embeddings": model_config.layers[0].attention.rope.max_seq_len,
                "model_type": "domynedge",
                "num_attention_heads": model_config.layers[0].attention.q_heads,
                "num_hidden_layers": len(model_config.layers),
                "num_key_value_heads": model_config.layers[0].attention.kv_heads,
                "rms_norm_eps": model_config.layers[0].attention.q_norm.eps,
                "position_layers": ["rope" if layer.attention.use_rope else "nope" for layer in model_config.layers],
                "window_sizes": [(layer.attention.sliding_window_size, 0) for layer in model_config.layers],
                "head_sizes": [layer.attention.head_dim for layer in model_config.layers],
                "tie_word_embeddings": True,
                "rope_theta": model_config.layers[0].attention.rope.theta,
                "transformers_version": transformers.__version__,
                "use_cache": True,
                "vocab_size": resolve_vocab_size(vocab_size, len(tokenizer), None),
            },
            indent=4,
        )
    )
    console.print("done.")


@torch.inference_mode()
@click.command()
@click.option("--model-dir", "model_dir", type=click.Path(), required=True, help="Input directory with DCP weights.")
@click.option(
    "--tokenizer-dir", "tokenizer_dir", type=click.Path(), required=True, help="Input directory for tokenizer."
)
@click.option(
    "--output-dir", "output_dir", type=click.Path(), required=True, help="Output directory for HF checkpoint."
)
@click.option(
    "--assets-dir",
    "assets_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Input directory with the HF modeling and configuration files.",
)
@click.option(
    "--dtype",
    "dtype",
    type=click.Choice(["float16", "bfloat16", "float32"]),
    default="bfloat16",
    help="Export dtype for HF checkpoint (default: bfloat16).",
)
@click.option("--model-name", "model_name", type=str, required=True, help="Model name (default: llama3).")
@click.option("--model-flavor", "model_flavor", type=str, required=True, help="Model flavor (default: 8B).")
@click.option("--max-seq-len", "max_seq_len", type=int, required=True, help="Maximum sequence length.")
@click.option("--overwrite", "overwrite", is_flag=True, default=False, help="Overwrite output directory if it exists.")
@click.option("--vocab-size", "vocab_size", type=int, default=None, help="Vocabulary rows to export.")
@click.option(
    "--model-vocab-size",
    "model_vocab_size",
    type=int,
    default=None,
    help="Vocabulary size the checkpoint was trained with, if the run overrode the flavor default.",
)
def main(
    model_dir,
    tokenizer_dir,
    output_dir,
    assets_dir,
    dtype,
    model_name,
    model_flavor,
    max_seq_len,
    overwrite,
    vocab_size,
    model_vocab_size,
):
    """Main conversion function orchestrating the complete conversion process.

    Args:
        model_dir: Path to DCP checkpoint directory
        tokenizer_dir: Path to tokenizer directory
        output_dir: Path where HF model will be saved
        assets_dir: Path to the HF modeling and configuration files
        dtype: Target data type for weights
        model_name: Name of model specification
        model_flavor: Specific model variant
        max_seq_len: Maximum sequence length
        overwrite: Whether to overwrite existing output directory
        vocab_size: Vocabulary rows to export
        model_vocab_size: Vocabulary size the checkpoint was trained with, if it differs from the flavor default

    Raises:
        ValueError: If output directory contains invalid characters
    """
    console = rich.console.Console()
    console.log("Starting conversion from DCP to HF safetensors format with the following parameters:")
    console.log(
        rich.panel.Panel.fit(
            rich.json.JSON.from_data(
                {
                    "model_dir": model_dir,
                    "tokenizer_dir": tokenizer_dir,
                    "output_dir": output_dir,
                    "assets_dir": assets_dir,
                    "dtype": dtype,
                    "model_name": model_name,
                    "model_flavor": model_flavor,
                    "overwrite": overwrite,
                    "vocab_size": vocab_size,
                    "model_vocab_size": model_vocab_size,
                }
            )
        )
    )

    model_dir = pathlib.Path(model_dir)
    tokenizer_dir = pathlib.Path(tokenizer_dir)
    output_dir = pathlib.Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=overwrite)

    save_model(
        console, tokenizer_dir, model_dir, output_dir, dtype, model_name, model_flavor, vocab_size, model_vocab_size
    )
    save_tokenizer(console, tokenizer_dir, output_dir)
    save_modeling(console, output_dir, assets_dir)
    save_config(console, tokenizer_dir, output_dir, model_name, model_flavor, dtype, max_seq_len, vocab_size)

    console.log("Conversion completed successfully.")


if __name__ == "__main__":
    main()
