#!/usr/bin/env python3
"""Export a staged Domyn Edge v5 checkpoint directly to a LiteRT-LM bundle."""

import argparse
import io
import json
import os
import shutil
from contextlib import ExitStack, contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from scripts.litert.export.compact_mask_ir import MIN_QUERY_TOKENS
from scripts.litert.export.stage_v5_checkpoint import ASSETS, AUTO_MAP, MODELING_FILES

RECIPE = Path(__file__).with_name("int4_reasoning_recipe.json")
TEMPLATE = Path(__file__).with_name("quick_chat.jinja")


def _quantize_to_disk(model_path, recipe, output=None):
    """Use the public disk serializer with the pinned exporter's quantization stage."""
    from litert_quantizer.quantizer import Quantizer

    if not recipe:
        return model_path
    path = Path(model_path)
    output = output or path.with_name(f"{path.stem}_quantized.tflite")
    if output.exists():
        raise FileExistsError(output)
    Quantizer(path, recipe).quantize(serialize_to_path=output)
    return str(output)


def _check_output(parser, output, checkpoint, compiled):
    if output.exists():
        parser.error(f"Choose a fresh output directory: {output}")
    existing_parent = next(path for path in output.parents if path.is_dir())
    weight_bytes = sum(path.stat().st_size for path in checkpoint.glob("*.safetensors"))
    if not weight_bytes:
        parser.error("No staged safetensors weights found")
    # A resumed export reads the existing decoder and only adds the quantized
    # decoder, the packaged bundle, and their working files.
    if compiled is not None:
        decoder, embedder = compiled
        # The two quantized copies are smaller than the FP32 decoder with
        # this INT4 recipe. Reserve one decoder's size plus both embeds.
        minimum = decoder.stat().st_size + 2 * embedder.stat().st_size + 2 * 1024**3
    else:
        # A fresh build also keeps the V2 parameters and float TFLite.
        minimum = 4 * weight_bytes + 2 * 1024**3
    if shutil.disk_usage(existing_parent).free < minimum:
        parser.error(f"Allow at least {minimum / 1024**3:.1f} GiB free on the output volume")


def check_bundle_metadata(output, eos_token_id):
    """Read the packaged LlmMetadata back and fail on any unexpected prompt setting."""
    from litert_lm_builder import litertlm_header_schema_py_generated as bundle_schema
    from litert_lm_builder.litertlm_peek import read_litertlm_header
    from litert_lm_builder.runtime.proto.llm_metadata_pb2 import LlmMetadata

    bundles = sorted(output.glob("*.litertlm"))
    if len(bundles) != 1:
        msg = f"Expected one .litertlm bundle in {output}, found {len(bundles)}"
        raise RuntimeError(msg)
    header = read_litertlm_header(str(bundles[0]), io.StringIO()).SectionMetadata()
    sections = [
        header.Objects(i)
        for i in range(header.ObjectsLength())
        if header.Objects(i).DataType() == bundle_schema.AnySectionDataType.LlmMetadataProto
    ]
    if len(sections) != 1:
        msg = "Expected exactly one LlmMetadata section"
        raise RuntimeError(msg)
    with bundles[0].open("rb") as stream:
        stream.seek(sections[0].BeginOffset())
        metadata = LlmMetadata.FromString(stream.read(sections[0].EndOffset() - sections[0].BeginOffset()))

    problems = []
    if metadata.jinja_prompt_template != TEMPLATE.read_text():
        problems.append("packaged chat template differs from quick_chat.jinja")
    if metadata.supports_thinking:
        problems.append("supports_thinking is set")
    if eos_token_id is not None and not any(
        list(stop.token_ids.ids) == [eos_token_id] for stop in metadata.stop_tokens
    ):
        problems.append(f"turn-end token {eos_token_id} is not a stop token")
    # quick_chat.jinja pre-closes the reasoning block, so the model should not
    # emit <think>. The builder still adds a `thought` channel whenever the
    # template contains <think>; it routes any stray reasoning away from the
    # visible reply. Accept that channel only, so a toolchain change is noticed.
    channels = [(c.channel_name, c.start, c.end) for c in metadata.channels]
    if channels not in ([], [("thought", "<think>", "</think>")]):
        problems.append(f"unexpected output channels: {channels}")
    if problems:
        raise RuntimeError("Bundle metadata check failed: " + "; ".join(problems))
    print(json.dumps({"bundle": str(bundles[0]), "metadata_check": "passed", "channels": channels}))


def _parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path, help="Output of stage_v5_checkpoint")
    parser.add_argument("output", type=Path)
    parser.add_argument("--cache-length", type=int, default=2048)
    parser.add_argument("--prefill-lengths", type=int, nargs="+", default=[128, 1024])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--compact-prefill-mask",
        action="store_true",
        help="Preserve scalar BF16 prefill mask fills in the V2 converter",
    )
    parser.add_argument(
        "--compiled-decoder",
        type=Path,
        help="Resume from a V2 TFLite decoder compiled with the same checkpoint and export settings",
    )
    parser.add_argument("--quantized-embedder", type=Path, help="Reuse the matching quantized embedder when resuming")
    args = parser.parse_args()
    if args.compact_prefill_mask and args.compiled_decoder is not None:
        parser.error("The compact prefill mask requires a fresh BF16 trace")
    if args.compact_prefill_mask and min(args.prefill_lengths) < MIN_QUERY_TOKENS:
        parser.error(f"The compact prefill mask requires prefill lengths of at least {MIN_QUERY_TOKENS}")
    return parser, args


def _compiled_inputs(parser, args):
    if args.compiled_decoder is None and args.quantized_embedder is None:
        return None
    if args.compiled_decoder is None or args.quantized_embedder is None:
        parser.error("Supply both --compiled-decoder and --quantized-embedder")
    paths = (args.compiled_decoder.resolve(strict=True), args.quantized_embedder.resolve(strict=True))
    for path in paths:
        with path.open("rb") as stream:
            if stream.read(8)[4:] != b"TFL3":
                parser.error(f"Invalid TFLite header in {path}")
    return paths


def _check_checkpoint(parser, args, checkpoint):
    config = json.loads((checkpoint / "config.json").read_text())
    if config.get("model_type") != "domynedge" or "window_sizes" in config or "layer_types" not in config:
        parser.error("Stage the Domyn Edge checkpoint with stage_v5_checkpoint first")
    if config.get("auto_map") != AUTO_MAP or any(
        (checkpoint / name).read_bytes() != (ASSETS / name).read_bytes() for name in MODELING_FILES
    ):
        parser.error("The staged checkpoint must use this checkout's canonical modeling")
    if args.cache_length < max(args.prefill_lengths) or min(args.prefill_lengths) < 1:
        parser.error("Prefill lengths must be positive and at most the cache length")
    if args.cache_length > config["max_position_embeddings"]:
        parser.error("Cache length exceeds the source checkpoint's context limit")


@contextmanager
def _compact_prefill_masks(args):
    if not args.compact_prefill_mask or args.dry_run:
        yield
        return

    from litert_torch.backend import export as lowering

    from scripts.litert.export.compact_mask_ir import compact_prefill_module

    original = lowering.exported_program_to_mlir
    compacted = []

    def compact_lowering(*positional, **keyword):
        result = original(*positional, **keyword)
        count = compact_prefill_module(result.module)
        if count:
            compacted.append(count)
            print(json.dumps({"compact_prefill_mask_selects": count}), flush=True)
        return result

    with patch.object(lowering, "exported_program_to_mlir", compact_lowering):
        yield
    if len(compacted) != len(set(args.prefill_lengths)):
        msg = "Expected a compacted attention mask in every prefill signature"
        raise RuntimeError(msg)


def main():
    """Run or resolve the one-call LiteRT export."""
    parser, args = _parse_args()
    compiled = _compiled_inputs(parser, args)
    checkpoint = args.checkpoint.resolve(strict=True)
    _check_checkpoint(parser, args, checkpoint)
    output = args.output.resolve()
    if not args.dry_run:
        _check_output(parser, output, checkpoint, compiled)

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("USE_FLAX", "0")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_MODULES_CACHE", str(checkpoint.parent / ".hf-modules"))
    os.environ.setdefault("HF_HOME", str(checkpoint.parent / ".hf-cache"))

    from litert_torch.generative.export_hf import model_ext
    from litert_torch.generative.export_hf.core import export_lib
    from litert_torch.generative.export_hf.export import export

    class DomynEdgeV5Extension(model_ext.ModelExtension):
        model_types = ("domynedge",)
        is_causal_text_model = True

        eos_token_id = None

        def build_llm_metadata(self, source, _config, _artifacts, metadata):
            eos = source.tokenizer.eos_token_id
            self.eos_token_id = eos
            if eos is not None and not any(list(stop.token_ids.ids) == [eos] for stop in metadata.stop_tokens):
                metadata.stop_tokens.add().token_ids.ids.append(eos)
            metadata.supports_thinking = False
            return metadata

    extension = DomynEdgeV5Extension()
    # This pinned LiteRT Torch revision otherwise assembles the entire quantized
    # output in RAM. Keep the converter, recipe, and bundle builder unchanged.
    with ExitStack() as stack, _compact_prefill_masks(args):
        stack.enter_context(patch.object(export_lib, "maybe_quantize_model", _quantize_to_disk))
        if compiled is not None:
            decoder, embedder = compiled

            def reuse_embedder(_source, _config, artifacts):
                return replace(artifacts, embedder_model_path=str(embedder))

            def reuse_decoder(_source, _config, artifacts):
                target = _quantize_to_disk(decoder, str(RECIPE), output / "model_quantized.tflite")
                return replace(artifacts, prefill_decode_model_path=target)

            stack.enter_context(patch.object(export_lib, "export_embedder_model", reuse_embedder))
            stack.enter_context(patch.object(export_lib, "export_text_prefill_decode_model", reuse_decoder))
        result = export(
            model=str(checkpoint),
            output_dir=str(output),
            task="text_generation",
            model_extension=extension,
            trust_remote_code=True,
            model_dtype="bfloat16",
            use_v2=True,
            delete_in_memory_params=True,
            prefill_lengths=sorted(set(args.prefill_lengths)),
            cache_length=args.cache_length,
            llm_metadata_max_num_tokens_override=args.cache_length,
            externalize_embedder=True,
            single_token_embedder=True,
            quantization_recipe=str(RECIPE),
            use_jinja_template=True,
            jinja_chat_template_override=str(TEMPLATE),
            bundle_litert_lm=True,
            keep_temporary_files=True,
            dry_run=args.dry_run,
        )
    if args.dry_run:
        print(result)
    else:
        check_bundle_metadata(output, extension.eos_token_id)


if __name__ == "__main__":
    main()
