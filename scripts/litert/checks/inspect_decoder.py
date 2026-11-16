#!/usr/bin/env python3
"""Report the signatures, largest constant buffers and zero INT4 scales in a LiteRT-LM decoder."""

import argparse
import io
import json
import mmap
from collections import defaultdict
from pathlib import Path

import numpy as np
from ai_edge_litert import schema_py_generated as tfl
from litert_lm_builder.litertlm_peek import get_model_type, read_litertlm_header

SCALE_DTYPES = {tfl.TensorType.FLOAT32: np.float32, tfl.TensorType.FLOAT16: np.float16}


def buffer_array(data, begin, buffer, dtype):
    """View a buffer's payload, stored either inline or after the FlatBuffer, without copying it."""
    if not buffer.Size():
        return buffer.DataAsNumpy().view(dtype)
    offset = begin + buffer.Offset()
    if not begin < offset <= offset + buffer.Size() <= len(data):
        msg = "External buffer falls outside the bundle"
        raise ValueError(msg)
    return np.frombuffer(data, dtype, count=buffer.Size() // np.dtype(dtype).itemsize, offset=offset)


def inspect_decoder(path: Path, largest: int) -> dict:
    """Read decoder metadata without loading the tensor payloads into RAM."""
    header = read_litertlm_header(str(path), io.StringIO()).SectionMetadata()
    section = next(
        (
            header.Objects(i)
            for i in range(header.ObjectsLength())
            if get_model_type(header.Objects(i)) == "tf_lite_prefill_decode"
        ),
        None,
    )
    if section is None:
        msg = "The bundle has no prefill/decode section"
        raise ValueError(msg)
    begin, end = section.BeginOffset(), section.EndOffset()
    with path.open("rb") as stream, mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if not 0 < begin < end <= len(data) or data[begin + 4 : begin + 8] != b"TFL3":
            msg = "Invalid TFLite decoder section"
            raise ValueError(msg)
        model = tfl.Model.GetRootAs(data, begin)
        sizes = [model.Buffers(i).Size() or model.Buffers(i).DataLength() for i in range(model.BuffersLength())]
        biggest = sorted(range(len(sizes)), key=sizes.__getitem__, reverse=True)[:largest]
        references = defaultdict(list)
        counts = defaultdict(int)
        scales = {}
        for i in range(model.SubgraphsLength()):
            graph = model.Subgraphs(i)
            for j in range(graph.TensorsLength()):
                tensor = graph.Tensors(j)
                buffer_id = tensor.Buffer()
                quantization = tensor.Quantization()
                if (
                    tensor.Type() == tfl.TensorType.INT4
                    and quantization is not None
                    and quantization.DetailsType() == tfl.QuantizationDetails.BlockwiseQuantization
                ):
                    table = quantization.Details()
                    block = tfl.BlockwiseQuantization()
                    block.Init(table.Bytes, table.Pos)
                    scale_tensor = graph.Tensors(block.Scales())
                    scales[scale_tensor.Buffer()] = SCALE_DTYPES[scale_tensor.Type()]
                if buffer_id in biggest:
                    counts[buffer_id] += 1
                    if len(references[buffer_id]) < 2:
                        references[buffer_id].append(
                            {
                                "subgraph": graph.Name().decode() if graph.Name() else str(i),
                                "shape": [tensor.Shape(d) for d in range(tensor.ShapeLength())],
                                "type": tensor.Type(),
                            }
                        )
        # A zero scale comes from an all-zero weight block: the CPU rejects it, the GPU delegate does not.
        zero_scales = sum(
            int((buffer_array(data, begin, model.Buffers(i), dtype) == 0).sum()) for i, dtype in scales.items()
        )
        return {
            "bundle_bytes": len(data),
            "decoder_bytes": end - begin,
            "signatures": sorted(
                model.SignatureDefs(i).SignatureKey().decode() for i in range(model.SignatureDefsLength())
            ),
            "int4_scale_buffers": len(scales),
            "zero_int4_scales": zero_scales,
            "largest_buffers": [
                {"buffer": i, "bytes": sizes[i], "tensor_references": counts[i], "examples": references[i]}
                for i in biggest
            ],
        }


def main():
    """Print a JSON report for one packaged decoder and fail on zero INT4 scales."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--largest", type=int, default=8)
    args = parser.parse_args()
    if args.largest < 0:
        parser.error("--largest must be nonnegative")
    report = inspect_decoder(args.bundle, args.largest)
    print(json.dumps(report, indent=2))
    return int(report["zero_int4_scales"] > 0)


if __name__ == "__main__":
    raise SystemExit(main())
