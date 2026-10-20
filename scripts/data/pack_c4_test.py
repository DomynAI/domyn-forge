# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Build a small pretokenized, packed dataset from TorchTitan's ``c4_test`` sample, for smoke tests.

Downloads ``tests/assets/c4_test/data.json`` (2,000 C4 documents) from TorchTitan at the pinned tag,
tokenizes every document with BOS and EOS, packs the token stream into rows of ``seq_len + 1`` tokens with
``position_ids`` restarting at each document start, and saves the rows as ``*.arrow`` shards readable by
``PretokenizedDataLoader``. Run it once from the repository root:

    uv run scripts/data/pack_c4_test.py
"""

import pathlib

import click
from datasets import Dataset, load_dataset
from torchtitan.components.tokenizer import HuggingFaceTokenizer

C4_TEST_URL = "https://raw.githubusercontent.com/pytorch/torchtitan/v0.3.0/tests/assets/c4_test/data.json"


@click.command()
@click.option("--tokenizer-dir", default="assets/domyn-edge/tokenizers/pretraining", show_default=True)
@click.option("--output-dir", default="data/c4_test_packed", show_default=True)
@click.option("--seq-len", default=2048, show_default=True, help="Training sequence length; rows hold seq_len + 1.")
@click.option("--num-shards", default=4, show_default=True)
def main(tokenizer_dir: str, output_dir: str, seq_len: int, num_shards: int) -> None:
    """Download, tokenize and pack c4_test into ``output_dir``."""
    texts: list[str] = load_dataset("json", data_files=C4_TEST_URL, split="train")["text"]

    tokenizer = HuggingFaceTokenizer(tokenizer_path=tokenizer_dir)
    tokens: list[int] = []
    positions: list[int] = []
    for text in texts:
        document = tokenizer.encode(text, add_bos=True, add_eos=True)
        tokens += document
        positions += range(len(document))

    # Cut the stream into rows; a document split across rows restarts its positions at the next row.
    row_len = seq_len + 1
    input_ids, position_ids = [], []
    for start in range(0, len(tokens) - row_len + 1, row_len):
        row_positions = positions[start : start + row_len]
        # Shift the leading piece of a document carried over from the previous row so it starts at 0.
        first_start = next((i for i, p in enumerate(row_positions) if p == 0), row_len)
        input_ids.append(tokens[start : start + row_len])
        position_ids.append([p - row_positions[0] if i < first_start else p for i, p in enumerate(row_positions)])

    Dataset.from_dict({"input_ids": input_ids, "position_ids": position_ids}).save_to_disk(
        pathlib.Path(output_dir), num_shards=num_shards
    )
    click.echo(f"{len(texts)} documents, {len(tokens)} tokens -> {len(input_ids)} rows of {row_len} in {output_dir}")


if __name__ == "__main__":
    main()
