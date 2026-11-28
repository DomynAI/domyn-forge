# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""HuggingFace to LiteRT-LM Model Conversion Script.

This script converts a DomynEdge HuggingFace checkpoint, such as `domyn/Domyn-Edge-SFT` or the output of
`scripts/huggingface/convert.py`, into a single `.litertlm` bundle for on-device CPU inference with
LiteRT-LM (https://github.com/google-ai-edge/LiteRT-LM).

The conversion process includes:
- Staging the checkpoint with this repository's modeling files and a canonical `config.json`
  (`layer_types` and `sliding_window` instead of the older flash-style `window_sizes`)
- Tracing the model with LiteRT Torch, with prefill signatures and a fixed-size KV cache
- Quantizing the decoder weights to INT4 (blockwise, 32) and the embedding table to INT8
- Packaging the decoder, the embedder, the tokenizer and a chat template into `model.litertlm`
- Reading the packaged metadata back to check the chat template and the start and stop tokens

The output directory holds `model.litertlm` and `hf/`, the staged checkpoint the bundle was exported from
(its weights are symbolic links to the source checkpoint), which can be loaded with transformers to compare
the two.

Usage:
    python convert.py \
        --model-dir <path-to-hf-checkpoint> \
        --output-dir <output-directory> \
        --assets-dir <path-to-hf-modeling-files>
"""

import contextlib
import io
import json
import os
import pathlib
import shutil
import subprocess
import sys
import typing
from unittest import mock

import click
import rich
import rich.console
import rich.json
import rich.panel

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

# Linear layers: INT4 weights in blocks of 32 (OCTAV). Embedding table: INT8 per channel.
QUANTIZATION_RECIPE = [
    {
        "regex": ".*",
        "operation": "FULLY_CONNECTED",
        "algorithm_key": "OCTAV",
        "op_config": {
            "weight_tensor_config": {"num_bits": 4, "granularity": "BLOCKWISE_32", "symmetric": True},
            "compute_precision": "INTEGER",
        },
    },
    {
        "regex": ".*",
        "operation": "EMBEDDING_LOOKUP",
        "algorithm_key": "min_max_uniform_quantize",
        "op_config": {
            "weight_tensor_config": {"num_bits": 8, "granularity": "CHANNELWISE", "symmetric": True},
            "compute_precision": "INTEGER",
        },
    },
]

# The chat template packaged in the bundle. It opens the assistant turn with an empty reasoning block, so the model
# answers directly, as with `enable_thinking=False` in the HF chat template.
CHAT_TEMPLATE = """\
{%- macro text(content) -%}
  {%- if content is string -%}
    {{- content -}}
  {%- else -%}
    {%- for item in content if item.type == 'text' -%}
      {{- item.text -}}
    {%- endfor -%}
  {%- endif -%}
{%- endmacro -%}
{{- '<|im_start|>system\\nYou are Domyn Edge, a model trained by Domyn.' -}}
{%- if messages and messages[0].role == 'system' -%}
  {%- set system_text = text(messages[0].content)|trim -%}
  {%- if system_text -%}{{- '\\n\\n' + system_text -}}{%- endif -%}
{%- endif -%}
{{- '<|im_end|>\\n' -}}
{%- for message in messages if message.role in ['user', 'assistant'] -%}
  {{- '<|im_start|>' + message.role + '\\n' -}}
  {{- text(message.content)|trim -}}
  {{- '<|im_end|>\\n' -}}
{%- endfor -%}
{%- if add_generation_prompt -%}
  {{- '<|im_start|>assistant\\n<think>\\n\\n</think>\\n\\n' -}}
{%- endif -%}
"""

# Prefill queries are at least this long. Decode can have a short query axis after grouped-query packing, and its
# attention mask is left as traced.
MIN_PREFILL_TOKENS = 128


def canonical_config(config: dict[str, typing.Any]) -> dict[str, typing.Any]:
    """Return the checkpoint config in the form the modeling in `assets/domyn-edge/hf` reads.

    Older checkpoints describe attention with flash-style `window_sizes` (`[left, right]` per layer, `-1` for global
    attention) and a top-level `rope_theta`. They are translated to `layer_types`, `sliding_window` and
    `rope_parameters`, as `scripts/huggingface/convert.py` writes them. A config already in that form is kept.

    Args:
        config: The checkpoint's `config.json`, as a dict.

    Returns:
        A new dict, with `auto_map` pointing at the staged modeling files.

    Raises:
        ValueError: If the config is not DomynEdge, or its attention windows cannot be expressed canonically.
    """
    if config.get("model_type") != "domynedge":
        msg = "Expected a DomynEdge HF checkpoint"
        raise ValueError(msg)
    config = {**config, "auto_map": AUTO_MAP}
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
            msg = "All sliding window layers must have the same window size"
            raise ValueError(msg)
        config["layer_types"] = ["sliding_attention" if left >= 0 else "full_attention" for left, _ in windows]
        # The flash window counts the preceding tokens; transformers' `sliding_window` counts the current one too.
        config["sliding_window"] = next(iter(local_windows)) + 1 if local_windows else None
    if len(config.get("layer_types") or []) != config["num_hidden_layers"]:
        msg = "Expected one attention type per layer"
        raise ValueError(msg)
    if "rope_parameters" not in config and "rope_theta" in config:
        config["rope_parameters"] = {"rope_type": "default", "rope_theta": config.pop("rope_theta")}
    return config


def stage_checkpoint(console, model_dir, staged_dir, assets_dir):
    """Write the canonical config, modeling and tokenizer files to `staged_dir`, and link the weights.

    Args:
        console: Rich console for progress output
        model_dir: Path to the source HF checkpoint
        staged_dir: Path of the staged checkpoint, which must not exist
        assets_dir: Directory containing the modeling files

    Raises:
        ValueError: If a weight file listed by the checkpoint is missing
        FileNotFoundError: If the tokenizer files are missing
    """
    console.print("Staging checkpoint...", end="")
    config = canonical_config(json.loads((model_dir / "config.json").read_text()))
    index = model_dir / "model.safetensors.index.json"
    if index.is_file():
        shards = sorted(set(json.loads(index.read_text())["weight_map"].values()))
        weight_files = [index.name, *shards]
    else:
        weight_files = ["model.safetensors"]
    for name in weight_files:
        if pathlib.Path(name).name != name or not (model_dir / name).is_file():
            msg = f"Missing or invalid checkpoint file: {name}"
            raise ValueError(msg)
    for name in ("tokenizer.json", "tokenizer_config.json"):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(model_dir / name)

    staged_dir.mkdir(parents=True)
    (staged_dir / "config.json").write_text(json.dumps(config, indent=4) + "\n")
    for name in MODELING_FILES:
        shutil.copyfile(assets_dir / name, staged_dir / name)
    for name in TOKENIZER_FILES:
        if (model_dir / name).is_file():
            shutil.copyfile(model_dir / name, staged_dir / name)
    for name in weight_files:
        # Relative links keep the staged checkpoint usable when the directories are mounted elsewhere.
        (staged_dir / name).symlink_to(os.path.relpath(model_dir / name, staged_dir))
    console.print("done.")


def promote_prefill_mask_selects(module):
    """Compute the attention mask fill of every prefill signature in float32, in place.

    The masked attention logits are traced as a bfloat16 `select` against a scalar fill broadcast to the full
    `[batch, heads, query, key]` shape. The converter keeps that broadcast as a dense constant, one per prefill
    signature and attention layer; on Domyn-Edge-SFT they add 289 MiB to the bundle. Converting the logits to float32
    around the select keeps the fill scalar, and the select result is converted back to bfloat16, so the attention
    output stays in the traced dtype.

    Args:
        module: The StableHLO MLIR module of one exported signature.

    Returns:
        The number of rewritten selects.
    """
    from litert_converter.mlir import ir

    with module.context, ir.Location.unknown():
        selects = [
            op
            for function in module.body.operations
            if function.operation.name == "func.func"
            for region in function.regions
            for block in region.blocks
            for op in block.operations
            if op.operation.name == "stablehlo.select"
        ]
        count = sum(_promote_select(op, ir) for op in selects)
        if count:
            module.operation.verify()
    return count


def _promote_select(op, ir):
    """Rewrite one rank-4 bfloat16 select with a broadcast scalar fill, if `op` is one."""
    condition, logits, fill = op.operands
    broadcast = fill.owner
    if not hasattr(broadcast, "operation") or broadcast.operation.name != "stablehlo.broadcast_in_dim":
        return False
    scalar = broadcast.operands[0]
    if str(scalar.type) != "tensor<bf16>":
        return False
    logits_type = ir.RankedTensorType(logits.type)
    if logits_type.element_type != ir.BF16Type.get() or logits_type.rank != 4:
        return False
    if logits_type.shape[-2] < MIN_PREFILL_TOKENS:
        return False
    fp32 = ir.F32Type.get()
    wide_logits_type = ir.RankedTensorType.get(logits_type.shape, fp32)
    with ir.InsertionPoint(op):
        wide_logits = ir.Operation.create("stablehlo.convert", results=[wide_logits_type], operands=[logits])
        wide_scalar = ir.Operation.create(
            "stablehlo.convert", results=[ir.RankedTensorType.get([], fp32)], operands=[scalar]
        )
        wide_fill = ir.Operation.create(
            "stablehlo.broadcast_in_dim",
            results=[wide_logits_type],
            operands=[wide_scalar.result],
            attributes={"broadcast_dimensions": broadcast.attributes["broadcast_dimensions"]},
        )
        selected = ir.Operation.create(
            "stablehlo.select",
            results=[wide_logits_type],
            operands=[condition, wide_logits.result, wide_fill.result],
        )
        restored = ir.Operation.create("stablehlo.convert", results=[logits_type], operands=[selected.result])
    op.result.replace_all_uses_with(restored.result)
    op.operation.erase()
    if not list(fill.uses):
        broadcast.erase()
    return True


@contextlib.contextmanager
def compact_prefill_masks(num_prefill_signatures):
    """Apply `promote_prefill_mask_selects` to every module LiteRT Torch lowers while the context is open.

    Raises:
        RuntimeError: If a prefill signature had no attention mask to rewrite.
    """
    from litert_torch.backend import export as lowering

    original = lowering.exported_program_to_mlir
    compacted = []

    def lower_and_compact(*args, **kwargs):
        result = original(*args, **kwargs)
        count = promote_prefill_mask_selects(result.module)
        if count:
            compacted.append(count)
        return result

    with mock.patch.object(lowering, "exported_program_to_mlir", lower_and_compact):
        yield
    if len(compacted) != num_prefill_signatures:
        msg = f"Expected a rewritten attention mask in each of the {num_prefill_signatures} prefill signatures"
        raise RuntimeError(msg)


# Builds LiteRT Torch's converter flags (`converter_v2._build_tfl_converter_flags` at the pinned commit) and runs the
# TFLite compilation of the traced signatures, in a process that loads TensorFlow and nothing else.
_COMPILE_TFLITE = """\
import json, sys
from tensorflow.compiler.mlir.lite import converter_flags_pb2
from tensorflow.lite.python import convert

options, input_dir, output_path = json.loads(sys.argv[1]), sys.argv[2], sys.argv[3]
flags = converter_flags_pb2.ConverterFlags()
flags.model_origin_framework = converter_flags_pb2.ConverterFlags.PYTORCH
flags.enable_composite_direct_lowering = True
flags.enable_x64 = options["enable_x64"]
flags.strict_qdq_mode = True
flags.fold_fp16_resource_casts = options["fold_fp16_resource_casts"]
if options["debug_dir"]:
    flags.debug_dir = options["debug_dir"]
for name, value in options["extra_flags"].items():
    if not hasattr(flags, name):
        sys.exit(f"Unknown TFLite converter flag: {name}")
    setattr(flags, name, value)
convert.convert_mlir_bytecode(flags, input_dir, output_path)
"""
_COMPILE_OPTION_DEFAULTS = {
    "enable_debug": False,
    "enable_timing": False,
    "print_ir_before": None,
    "print_ir_after": None,
    "elide_elementsattrs_if_larger": None,
}


def converter_options(
    *, enable_x64=False, fold_fp16_resource_casts=True, debug_dir=None, _litert_converter_flags=None, **others
):
    """Stand in for `_build_tfl_converter_flags`, keeping the options `_COMPILE_TFLITE` turns into flags.

    Raises:
        RuntimeError: If LiteRT Torch asks for a flag `_COMPILE_TFLITE` does not set.
    """
    extra_flags = dict(_litert_converter_flags or {})
    unexpected = {name: value for name, value in others.items() if _COMPILE_OPTION_DEFAULTS.get(name, ...) != value}
    unexpected |= {name: value for name, value in extra_flags.items() if isinstance(value, dict)}
    if unexpected:
        msg = f"Unexpected TFLite converter options: {unexpected}"
        raise RuntimeError(msg)
    return {
        "enable_x64": enable_x64,
        "fold_fp16_resource_casts": fold_fp16_resource_casts,
        "debug_dir": debug_dir,
        "extra_flags": extra_flags,
    }


def compile_tflite_in_subprocess(options, input_dir, output_path):
    """Stand in for `_convert_mlir_bytecode`, compiling the traced signatures in a separate process.

    TensorFlow and the MLIR library LiteRT Torch traces with each bundle their own LLVM, and loading both in one
    process aborts (`Option 'info-output-file' registered more than once`) on macOS arm64.
    """
    subprocess.run(
        [sys.executable, "-c", _COMPILE_TFLITE, json.dumps(options), str(input_dir), str(output_path)], check=True
    )


def quantize_to_disk(model_path, recipe):
    """Quantize a TFLite model, streaming the result to disk.

    It replaces LiteRT Torch's `maybe_quantize_model`, which builds the whole quantized model in memory before
    writing it.
    """
    from litert_quantizer.quantizer import Quantizer

    if not recipe:
        return model_path
    path = pathlib.Path(model_path)
    output = path.with_name(f"{path.stem}_quantized.tflite")
    Quantizer(path, recipe).quantize(serialize_to_path=output)
    return str(output)


def read_bundle_metadata(bundle):
    """Return the `LlmMetadata` packaged in a `.litertlm` bundle."""
    from litert_lm_builder import litertlm_header_schema_py_generated as bundle_schema
    from litert_lm_builder.litertlm_peek import read_litertlm_header
    from litert_lm_builder.runtime.proto.llm_metadata_pb2 import LlmMetadata

    header = read_litertlm_header(str(bundle), io.StringIO()).SectionMetadata()
    sections = [
        header.Objects(i)
        for i in range(header.ObjectsLength())
        if header.Objects(i).DataType() == bundle_schema.AnySectionDataType.LlmMetadataProto
    ]
    if len(sections) != 1:
        msg = f"Expected exactly one LlmMetadata section in {bundle}, found {len(sections)}"
        raise RuntimeError(msg)
    with bundle.open("rb") as stream:
        stream.seek(sections[0].BeginOffset())
        return LlmMetadata.FromString(stream.read(sections[0].EndOffset() - sections[0].BeginOffset()))


def check_bundle_metadata(bundle, eos_token_id):
    """Fail if the packaged chat template, start and stop tokens or output channels are not the expected ones.

    Raises:
        RuntimeError: Listing every unexpected setting.
    """
    metadata = read_bundle_metadata(bundle)
    problems = []
    if metadata.jinja_prompt_template != CHAT_TEMPLATE:
        problems.append("the packaged chat template differs from CHAT_TEMPLATE")
    if metadata.supports_thinking:
        problems.append("supports_thinking is set")
    if metadata.HasField("start_token"):
        problems.append(f"the runtime would prepend a start token to the template: {metadata.start_token}")
    if not any(list(stop.token_ids.ids) == [eos_token_id] for stop in metadata.stop_tokens):
        problems.append(f"the end-of-turn token {eos_token_id} is not a stop token")
    # LiteRT Torch adds a `thought` channel whenever the template contains <think>. It routes stray reasoning away
    # from the visible reply. Any other channel means the toolchain changed.
    channels = [(channel.channel_name, channel.start, channel.end) for channel in metadata.channels]
    if channels not in ([], [("thought", "<think>", "</think>")]):
        problems.append(f"unexpected output channels: {channels}")
    if problems:
        raise RuntimeError("Bundle metadata check failed: " + "; ".join(problems))


def save_bundle(console, staged_dir, output_dir, cache_length, prefill_lengths):
    """Export the staged checkpoint to `output_dir/model.litertlm` and check its metadata.

    Args:
        console: Rich console for progress output
        staged_dir: Path to the staged checkpoint
        output_dir: Path where the bundle will be saved
        cache_length: KV cache length, which is also the bundle's maximum number of tokens
        prefill_lengths: Lengths of the prefill signatures

    Returns:
        The path of the bundle.
    """
    # The staged checkpoint is local and the remote code is copied from this repository: nothing is downloaded.
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", str(output_dir / ".hf-modules"))
    from litert_torch._convert import converter_v2
    from litert_torch.generative.export_hf import model_ext
    from litert_torch.generative.export_hf.core import export_lib
    from litert_torch.generative.export_hf.export import export

    recipe = staged_dir / "litert_quantization_recipe.json"
    recipe.write_text(json.dumps(QUANTIZATION_RECIPE, indent=4) + "\n")
    template = staged_dir / "litert_chat_template.jinja"
    template.write_text(CHAT_TEMPLATE)

    class DomynEdgeExtension(model_ext.ModelExtension):
        model_types = ("domynedge",)
        is_causal_text_model = True
        eos_token_id = None

        def build_llm_metadata(self, source_model_artifacts, export_config, exported_model_artifacts, llm_metadata):
            del export_config, exported_model_artifacts
            eos = source_model_artifacts.tokenizer.eos_token_id
            self.eos_token_id = eos
            if not any(list(stop.token_ids.ids) == [eos] for stop in llm_metadata.stop_tokens):
                llm_metadata.stop_tokens.add().token_ids.ids.append(eos)
            llm_metadata.supports_thinking = False
            # LiteRT-LM prepends the start token to every prompt, chat or not. CHAT_TEMPLATE already opens with
            # <|im_start|>, the SFT tokenizer's BOS, so the model would read <|im_start|><|im_start|>system.
            llm_metadata.ClearField("start_token")
            return llm_metadata

    extension = DomynEdgeExtension()
    console.print("Exporting to LiteRT-LM (this takes a while)...")
    with (
        compact_prefill_masks(len(prefill_lengths)),
        mock.patch.object(export_lib, "maybe_quantize_model", quantize_to_disk),
        mock.patch.object(converter_v2, "_build_tfl_converter_flags", converter_options),
        mock.patch.object(converter_v2, "_convert_mlir_bytecode", compile_tflite_in_subprocess),
    ):
        export(
            model=str(staged_dir),
            output_dir=str(output_dir),
            task="text_generation",
            model_extension=extension,
            trust_remote_code=True,
            model_dtype="bfloat16",
            use_v2=True,
            delete_in_memory_params=True,
            prefill_lengths=prefill_lengths,
            cache_length=cache_length,
            llm_metadata_max_num_tokens_override=cache_length,
            externalize_embedder=True,
            single_token_embedder=True,
            # The tests empty QUANTIZATION_RECIPE to measure the conversion without the quantization error.
            quantization_recipe=str(recipe) if QUANTIZATION_RECIPE else None,
            use_jinja_template=True,
            jinja_chat_template_override=str(template),
            bundle_litert_lm=True,
        )
    bundle = output_dir / "model.litertlm"
    if extension.eos_token_id is None:
        msg = "The tokenizer has no end-of-turn token"
        raise RuntimeError(msg)
    console.print("Checking bundle metadata...", end="")
    check_bundle_metadata(bundle, extension.eos_token_id)
    console.print("done.")
    return bundle


def check_inputs(model_dir, output_dir, cache_length, prefill_lengths):
    """Validate the options before the export starts.

    Raises:
        click.BadParameter: On an option the export cannot use.
    """
    if output_dir.exists():
        msg = f"{output_dir} already exists; choose a new output directory"
        raise click.BadParameter(msg, param_hint="--output-dir")
    if min(prefill_lengths) < MIN_PREFILL_TOKENS or max(prefill_lengths) > cache_length:
        msg = f"Prefill lengths must be between {MIN_PREFILL_TOKENS} and the cache length ({cache_length})"
        raise click.BadParameter(msg, param_hint="--prefill-lengths")
    max_positions = json.loads((model_dir / "config.json").read_text())["max_position_embeddings"]
    if cache_length > max_positions:
        msg = f"The cache length exceeds the checkpoint's context limit ({max_positions})"
        raise click.BadParameter(msg, param_hint="--cache-length")
    # The export keeps the float32 decoder, its quantized copy and the bundle on disk at the same time.
    existing_parent = next(path for path in output_dir.parents if path.is_dir())
    weight_bytes = sum(path.stat().st_size for path in model_dir.glob("*.safetensors"))
    required = 4 * weight_bytes + 2 * 1024**3
    if shutil.disk_usage(existing_parent).free < required:
        msg = f"Allow at least {required / 1024**3:.1f} GiB free on the output volume"
        raise click.BadParameter(msg, param_hint="--output-dir")


@click.command()
@click.option(
    "--model-dir",
    "model_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Input directory with the HF checkpoint.",
)
@click.option(
    "--output-dir", "output_dir", type=click.Path(), required=True, help="New output directory for the bundle."
)
@click.option(
    "--assets-dir",
    "assets_dir",
    type=click.Path(exists=True, file_okay=False),
    required=True,
    help="Input directory with the HF modeling and configuration files.",
)
@click.option(
    "--cache-length",
    "cache_length",
    type=int,
    default=2048,
    help="KV cache length, the maximum number of prompt and generated tokens (default: 2048).",
)
@click.option(
    "--prefill-lengths",
    "prefill_lengths",
    type=int,
    multiple=True,
    default=(128, 1024),
    help="Prefill signature lengths, repeat the option for several (default: 128 and 1024).",
)
def main(model_dir, output_dir, assets_dir, cache_length, prefill_lengths):
    """Convert a DomynEdge HF checkpoint to an INT4 LiteRT-LM bundle.

    Args:
        model_dir: Path to the HF checkpoint directory
        output_dir: Path where the bundle and the staged checkpoint will be saved
        assets_dir: Path to the HF modeling and configuration files
        cache_length: KV cache length
        prefill_lengths: Prefill signature lengths
    """
    console = rich.console.Console()
    console.log("Starting conversion from HF to LiteRT-LM with the following parameters:")
    console.log(
        rich.panel.Panel.fit(
            rich.json.JSON.from_data(
                {
                    "model_dir": model_dir,
                    "output_dir": output_dir,
                    "assets_dir": assets_dir,
                    "cache_length": cache_length,
                    "prefill_lengths": prefill_lengths,
                }
            )
        )
    )

    model_dir = pathlib.Path(model_dir).resolve()
    output_dir = pathlib.Path(output_dir).resolve()
    assets_dir = pathlib.Path(assets_dir).resolve()
    prefill_lengths = sorted(set(prefill_lengths))
    check_inputs(model_dir, output_dir, cache_length, prefill_lengths)

    staged_dir = output_dir / "hf"
    stage_checkpoint(console, model_dir, staged_dir, assets_dir)
    bundle = save_bundle(console, staged_dir, output_dir, cache_length, prefill_lengths)

    console.log(f"Conversion completed successfully: {bundle} ({bundle.stat().st_size / 1024**3:.2f} GiB).")


if __name__ == "__main__":
    main()
