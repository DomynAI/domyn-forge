#!/usr/bin/env python3
"""Require identical packed INT4 weights and blockwise quantization parameters."""

import argparse
import hashlib
import json
import mmap
from collections import Counter
from pathlib import Path

from ai_edge_litert import schema_py_generated as tfl


def buffer_digest(data, buffer):
    """Hash a TFLite buffer, including tensors stored after the FlatBuffer."""
    if buffer.Size():
        start, end = buffer.Offset(), buffer.Offset() + buffer.Size()
        if not 0 < start < end <= len(data):
            msg = "External buffer falls outside the TFLite file"
            raise ValueError(msg)
        with memoryview(data)[start:end] as payload:
            return hashlib.sha256(payload).hexdigest()
    if not buffer.DataLength():
        msg = "Expected a populated quantization buffer"
        raise ValueError(msg)
    return hashlib.sha256(buffer.DataAsNumpy()).hexdigest()


def weight_digests(path):
    """Read weight and scale hashes keyed by decoder signature."""
    with path.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as data:
        if data[4:8] != b"TFL3":
            msg = f"Expected a TFLite decoder in {path}"
            raise ValueError(msg)
        model = tfl.Model.GetRootAs(data)
        digests = {}
        buffers = {}

        def digest(index):
            if index not in buffers:
                buffers[index] = buffer_digest(data, model.Buffers(index))
            return buffers[index]

        for index in range(model.SignatureDefsLength()):
            signature = model.SignatureDefs(index)
            graph = model.Subgraphs(signature.SubgraphIndex())
            weights = []
            for i in range(graph.TensorsLength()):
                tensor = graph.Tensors(i)
                if tensor.Type() != tfl.TensorType.INT4:
                    continue
                quant = tensor.Quantization()
                if quant is None or quant.DetailsType() != tfl.QuantizationDetails.BlockwiseQuantization:
                    msg = "Expected blockwise quantization for every INT4 tensor"
                    raise ValueError(msg)
                table = quant.Details()
                block = tfl.BlockwiseQuantization()
                block.Init(table.Bytes, table.Pos)
                weights.append(
                    (
                        tuple(tensor.Shape(j) for j in range(tensor.ShapeLength())),
                        block.BlockSize(),
                        tuple(block.BlockShape(j) for j in range(block.BlockShapeLength())),
                        digest(tensor.Buffer()),
                        digest(graph.Tensors(block.Scales()).Buffer()),
                        digest(graph.Tensors(block.ZeroPoints()).Buffer()) if block.ZeroPoints() >= 0 else None,
                    )
                )
            digests[signature.SignatureKey().decode()] = Counter(weights)
        return digests


def main():
    """Compare payloads across all prefill and decode signatures."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()
    baseline = weight_digests(args.baseline)
    candidate = weight_digests(args.candidate)
    if not baseline or any(not counts for counts in baseline.values()):
        parser.error("The baseline must contain INT4 weights in every signature")
    mismatches = sorted(key for key in baseline.keys() | candidate.keys() if baseline.get(key) != candidate.get(key))
    print(json.dumps({"int4_tensors": {k: sum(v.values()) for k, v in baseline.items()}, "mismatches": mismatches}))
    return int(bool(mismatches))


if __name__ == "__main__":
    raise SystemExit(main())
