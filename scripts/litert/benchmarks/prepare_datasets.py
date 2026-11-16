#!/usr/bin/env python3
"""Download the pinned PIQA, GSM8K and IFEval revisions and write the JSONL files used in QUALIFICATION.md.

Each file is checked against the SHA-256 that the published results recorded, so a mismatch fails here instead of
producing results that cannot be compared.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

DATASETS = {
    "piqa": {
        "repo": "baber/piqa",
        "revision": "142f6d7367fd9877f0fb3b5734ea6a545f54cdd1",
        "file": "piqa_validation.parquet",
        "columns": ["goal", "sol1", "sol2", "label"],
        "output": "piqa-validation.jsonl",
        "sha256": "61533005e22f175534909b1ec8eacb6da03c233933558d1bafb15787453b1f55",
    },
    "gsm8k": {
        "repo": "openai/gsm8k",
        "revision": "740312add88f781978c0658806c59bc2815b9866",
        "file": "main/test-00000-of-00001.parquet",
        "columns": ["question", "answer"],
        "output": "gsm8k-test.jsonl",
        "sha256": "752adc99f23132275139f1f7be57126b08735a55daf0c56859ee6df70abafbac",
    },
    "ifeval": {
        "repo": "google/IFEval",
        "revision": "193ae952ec892316cf21fb5d8cbb704e811fe873",
        "file": "ifeval_input_data.jsonl",
        "columns": None,  # Already JSONL; used as published.
        "output": "ifeval-input.jsonl",
        "sha256": "6a85310ca8ce15eff755aa08a3a4ff931c7e273e7515ebb3c492ea85fd8288f2",
    },
}


def parquet_to_jsonl(path, columns):
    """Return the selected columns as JSON lines, in the original row order."""
    import pyarrow.parquet as pq

    rows = pq.read_table(path, columns=columns).to_pylist()
    return "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows).encode()


def prepare(name, output_dir):
    """Download one dataset, write its JSONL file and return the path after checking its SHA-256."""
    from huggingface_hub import hf_hub_download

    spec = DATASETS[name]
    output = output_dir / spec["output"]
    if output.exists():
        msg = f"Output already exists: {output}"
        raise FileExistsError(msg)
    path = hf_hub_download(spec["repo"], spec["file"], repo_type="dataset", revision=spec["revision"])
    data = Path(path).read_bytes() if spec["columns"] is None else parquet_to_jsonl(path, spec["columns"])
    digest = hashlib.sha256(data).hexdigest()
    if digest != spec["sha256"]:
        msg = f"{name}: SHA-256 {digest} does not match the published {spec['sha256']}"
        raise ValueError(msg)
    partial = output.with_suffix(".partial")
    partial.write_bytes(data)
    shutil.move(partial, output)
    return output


def main():
    """Prepare the requested datasets."""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("output_dir", type=Path, help="Directory for the JSONL files")
    parser.add_argument("--dataset", choices=sorted(DATASETS), action="append", help="Default: all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in args.dataset or sorted(DATASETS):
        output = prepare(name, args.output_dir)
        print(json.dumps({"dataset": name, "path": str(output), "sha256": DATASETS[name]["sha256"]}))


if __name__ == "__main__":
    main()
