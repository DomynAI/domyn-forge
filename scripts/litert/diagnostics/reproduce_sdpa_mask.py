#!/usr/bin/env python3
"""Reproduce the LiteRT V2 BF16 fill expansion using a masked attention fixture."""

import argparse
import json
from pathlib import Path

CONTEXT = 2053
PREFILL = 512
KV_HEADS = 8


def trace(root):
    """Trace both composite bodies without loading TensorFlow into this process."""
    import torch
    from litert_torch._convert import converter_v2
    from litert_torch._convert.signature import Signature
    from litert_torch.backend import composite

    class MaskedLogits(torch.nn.Module):
        def __init__(self, select_in_fp32):
            super().__init__()
            self.select_in_fp32 = select_in_fp32

        def forward(self, keep, logits):
            builder = composite.StableHLOCompositeBuilder(name="odml.sdpa_mask_fixture")
            keep, logits = builder.mark_inputs(keep, logits.to(torch.bfloat16))
            select_input = logits.float() if self.select_in_fp32 else logits
            # -9984 is -10000 rounded to BF16, as in the original SDPA mask.
            fill = torch.tensor(-9984.0, dtype=select_input.dtype)
            masked = torch.where(keep, select_input, fill).to(logits.dtype)
            return builder.mark_outputs(masked).float()

    root.mkdir()
    keep = torch.ones((1, 1, PREFILL, CONTEXT), dtype=torch.bool)
    logits = torch.zeros((1, KV_HEADS, PREFILL, CONTEXT), dtype=torch.float32)
    for name, select_in_fp32 in (("baseline", False), ("fp32-select", True)):
        output = root / name
        output.mkdir()
        model = MaskedLogits(select_in_fp32).eval()
        signature = Signature("serving_default", model, (keep, logits), {})
        converter_v2.export_to_dir([signature], str(output))


def compile_models(root):
    """Compile saved MLIR in a process that does not import LiteRT Torch."""
    from tensorflow.compiler.mlir.lite import converter_flags_pb2 as proto
    from tensorflow.lite.python import convert

    for name in ("baseline", "fp32-select"):
        output = root / f"{name}.tflite"
        if output.exists():
            raise FileExistsError(output)
        flags = proto.ConverterFlags(
            model_origin_framework=proto.ConverterFlags.PYTORCH,
            enable_composite_direct_lowering=True,
            strict_qdq_mode=True,
            fold_fp16_resource_casts=True,
        )
        convert.convert_mlir_bytecode(flags, str(root / name), str(output))


def verify(root):
    """Check CPU outputs against the BF16 reference and count fill bytes."""
    import numpy as np
    from ai_edge_litert import schema_py_generated as schema
    from ai_edge_litert.interpreter import Interpreter

    rng = np.random.default_rng(42)
    logits = rng.integers(-128, 129, (1, KV_HEADS, PREFILL, CONTEXT), dtype=np.int16).astype(np.float32) / 64
    keep = rng.integers(0, 2, (1, 1, PREFILL, CONTEXT), dtype=np.int8).astype(bool)
    expected = np.where(keep, logits, np.float32(-9984.0))
    for name in ("baseline", "fp32-select"):
        path = root / f"{name}.tflite"
        model = schema.Model.GetRootAs(path.read_bytes())
        fill_sizes = []
        for index in range(model.SubgraphsLength()):
            graph = model.Subgraphs(index)
            for op_index in range(graph.OperatorsLength()):
                op = graph.Operators(op_index)
                if model.OperatorCodes(op.OpcodeIndex()).BuiltinCode() == schema.BuiltinOperator.SELECT_V2:
                    fill = model.Buffers(graph.Tensors(op.Inputs(2)).Buffer())
                    fill_sizes.append(fill.Size() or fill.DataLength())
        runner = Interpreter(model_path=str(path)).get_signature_runner()
        inputs = {
            key: keep if info["dtype"] == np.bool_ else logits for key, info in runner.get_input_details().items()
        }
        np.testing.assert_array_equal(next(iter(runner(**inputs).values())), expected)
        print(json.dumps({"variant": name, "file_bytes": path.stat().st_size, "fill_bytes": fill_sizes, "exact": True}))


def main():
    """Run one phase of the reproduction."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("trace", "compile", "verify"))
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    phases = {"trace": trace, "compile": compile_models, "verify": verify}
    phases[args.phase](args.output)


if __name__ == "__main__":
    main()
