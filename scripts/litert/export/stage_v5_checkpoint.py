#!/usr/bin/env python3
"""Prepare a local Domyn Edge HF checkpoint with Forge's canonical modeling."""

import argparse
import json
import os
import shutil
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[3] / "assets" / "domyn-edge" / "hf"
AUTO_MAP = {
    "AutoConfig": "configuration_domynedge.DomynEdgeConfig",
    "AutoModel": "modeling_domynedge.DomynEdgeModel",
    "AutoModelForCausalLM": "modeling_domynedge.DomynEdgeForCausalLM",
}
MODELING_FILES = ("configuration_domynedge.py", "modeling_domynedge.py")
TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "chat_template.jinja",
    "generation_config.json",
)


def canonical_config(config):
    """Translate flash-style windows into the canonical HF layer configuration."""
    if config.get("model_type") != "domynedge":
        msg = "Expected a Domyn Edge HF checkpoint"
        raise ValueError(msg)
    config = dict(config, auto_map=AUTO_MAP)
    windows = config.pop("window_sizes", None)
    if windows is not None:
        if len(windows) != config["num_hidden_layers"]:
            msg = "Expected one window per layer"
            raise ValueError(msg)
        if any(right != 0 or left < -1 for left, right in windows):
            msg = "Expected causal flash-style attention windows"
            raise ValueError(msg)
        local_windows = {left for left, _ in windows if left >= 0}
        if len(local_windows) > 1:
            msg = "Canonical Domyn Edge requires a single local window size"
            raise ValueError(msg)
        config["layer_types"] = ["sliding_attention" if left >= 0 else "full_attention" for left, _ in windows]
        # The legacy flash window counts preceding keys. Transformers also
        # counts the current token, as in scripts/huggingface/convert.py.
        config["sliding_window"] = next(iter(local_windows)) + 1 if local_windows else None
    if len(config["layer_types"]) != config["num_hidden_layers"]:
        msg = "Expected one attention type per layer"
        raise ValueError(msg)
    if "rope_parameters" not in config and "rope_theta" in config:
        config["rope_parameters"] = {"rope_type": "default", "rope_theta": config.pop("rope_theta")}
    return config


def stage_checkpoint(source, output):
    """Write canonical model files to a fresh directory and link the source weights."""
    source = source.resolve(strict=True)
    output = output.resolve()
    config = canonical_config(json.loads((source / "config.json").read_text()))
    index = source / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        weight_files = [index.name, *shards]
    else:
        weight_files = ["model.safetensors"]
    for name in weight_files:
        if Path(name).name != name or not (source / name).is_file():
            msg = f"Missing or invalid checkpoint file: {name}"
            raise ValueError(msg)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (source / name).is_file():
            raise FileNotFoundError(source / name)

    output.mkdir(parents=True, exist_ok=False)
    (output / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    for name in MODELING_FILES:
        shutil.copyfile(ASSETS / name, output / name)
    for name in TOKENIZER_FILES:
        if (source / name).is_file():
            shutil.copyfile(source / name, output / name)
    for name in weight_files:
        # Keep the staging path usable inside a bind mount of the workspace.
        (output / name).symlink_to(os.path.relpath(source / name, output))
    return output


def main():
    """Stage a checkpoint from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    print(stage_checkpoint(args.source, args.output))


if __name__ == "__main__":
    main()
