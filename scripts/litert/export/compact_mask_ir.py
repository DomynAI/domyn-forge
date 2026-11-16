#!/usr/bin/env python3
"""Preserve scalar BF16 attention fills through LiteRT V2 composite lowering."""

import argparse
import json
from pathlib import Path

# Prefill queries are at least this long. Decode can have a short query axis after grouped-query packing.
MIN_QUERY_TOKENS = 128


def compact_prefill_module(module):
    """Rewrite the long-query selects in an in-memory V2 decoder trace."""
    from litert_converter.mlir import ir

    with module.context, ir.Location.unknown():
        count = promote_selects(module, ir)
        if count:
            module.operation.verify()
    return count


def promote_selects(module, ir):
    """Promote scalar-fill selects before broadcasting, then restore BF16."""
    selects = []
    for function in module.body.operations:
        if function.operation.name != "func.func":
            continue
        for region in function.regions:
            for block in region.blocks:
                selects.extend(op for op in block.operations if op.operation.name == "stablehlo.select")
    return sum(_promote_select(op, ir) for op in selects)


def _promote_select(op, ir):
    """Rewrite one rank-four BF16 select with a scalar fill."""
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
    if logits_type.shape[-2] < MIN_QUERY_TOKENS:
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


def main():
    """Copy compiled inputs into a fresh directory with rewritten prefill bytecode."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--signature", action="append", required=True)
    args = parser.parse_args()
    source = args.input.resolve(strict=True)
    requested = {f"{signature}.mlirbc" for signature in args.signature}
    inputs = sorted(source.glob("*.mlirbc"))
    if not requested <= {path.name for path in inputs}:
        parser.error("Each requested signature must be in the input directory")
    inputs.extend((source / name).resolve(strict=True) for name in ("params.bin", "weights_metadata.json"))

    from litert_converter.mlir import ir
    from litert_torch.backend.export_utils import create_ir_context

    args.output.mkdir()
    with create_ir_context(), ir.Location.unknown():
        for path in inputs:
            output = args.output / path.name
            if path.name not in requested:
                output.symlink_to(path)
                continue
            module = ir.Module.parse(path.read_bytes())
            count = promote_selects(module, ir)
            if not count:
                msg = f"No scalar BF16 fill selects found in {path.name}"
                raise ValueError(msg)
            module.operation.verify()
            with output.open("xb") as stream:
                module.operation.write_bytecode(file=stream)
            print(json.dumps({"signature": path.stem, "scalar_fill_selects": count}))


if __name__ == "__main__":
    main()
