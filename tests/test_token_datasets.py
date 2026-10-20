# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pretokenized ``*.arrow`` dataset and its resumable, shuffling dataloader.

Covers the causal shift and padding, loss masking (pretraining rows without ``token_type_ids`` vs fine-tuning rows
with them), input validation, shuffling, exact resume, data-parallel splits and ``max_docs_per_batch``.
"""

import math
import pathlib

import pytest
import torch
from datasets import Dataset
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer

from domyn_forge.datasets import PretokenizedDataLoader, PretokenizedDataset
from domyn_forge.datasets.token_datasets import count_rows, role2id, steps_per_epoch

NUM_SAMPLES = 20
BOS_ID = 1
EOS_ID = 2


class DummyTokenizer(BaseTokenizer):
    """Character tokenizer; the dataset only reads ``eos_id``."""

    def __init__(self):
        super().__init__()
        self.eos_id = EOS_ID
        self.bos_id = BOS_ID

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:  # noqa: FBT002
        tokens = [ord(c) for c in text]
        if add_bos:
            tokens.insert(0, BOS_ID)
        if add_eos:
            tokens.append(EOS_ID)
        return tokens

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(t) for t in token_ids if t > EOS_ID)

    def get_vocab_size(self) -> int:
        return 256


def save_shards(path: pathlib.Path, rows: dict[str, list], num_shards: int = 3) -> pathlib.Path:
    Dataset.from_dict(rows).save_to_disk(path, num_shards=num_shards)
    return path


@pytest.fixture
def id_dataset(tmp_path) -> pathlib.Path:
    """Shards where row ``i`` is ``input_ids=[i, eos]``: after the causal shift, ``input == [i]`` identifies the row."""
    return save_shards(
        tmp_path / "ids",
        {
            "input_ids": [[i, EOS_ID] for i in range(NUM_SAMPLES)],
            "position_ids": [[0, 1] for _ in range(NUM_SAMPLES)],
        },
    )


def make_dataset(path: pathlib.Path, seq_len: int = 1, **kwargs) -> PretokenizedDataset:
    return PretokenizedDataset(dataset_path=str(path), tokenizer=DummyTokenizer(), seq_len=seq_len, **kwargs)


def make_loader(
    path: pathlib.Path,
    *,
    shuffle: bool = True,
    shuffle_seed: int = 42,
    infinite: bool = False,
    dp_world_size: int = 1,
    dp_rank: int = 0,
    drop_last: bool = True,
    local_batch_size: int = 1,
    seq_len: int = 1,
    **config_kwargs,
) -> PretokenizedDataLoader:
    loader = PretokenizedDataLoader.Config(
        dataset_path=str(path),
        shuffle=shuffle,
        shuffle_seed=shuffle_seed,
        infinite=infinite,
        drop_last=drop_last,
        num_workers=0,
        **config_kwargs,
    ).build(
        dp_world_size=dp_world_size,
        dp_rank=dp_rank,
        tokenizer=DummyTokenizer(),
        seq_len=seq_len,
        local_batch_size=local_batch_size,
    )
    assert isinstance(loader, PretokenizedDataLoader)
    return loader


def take_ids(loader: PretokenizedDataLoader, n: int) -> list[int]:
    """Row id (first input token) of the first sample of each of the first ``n`` batches."""
    iterator = iter(loader)
    return [int(next(iterator)[0]["input"][0, 0]) for _ in range(n)]


def epoch_ids(loader: PretokenizedDataLoader) -> list[int]:
    """Row id of every sample of a finite (``infinite=False``) pass, so batch size and ``drop_last`` show."""
    return [int(t) for batch, _ in loader for t in batch["input"][:, 0].tolist()]


def per_rank_epoch_ids(path: pathlib.Path, dp_world_size: int, *, drop_last: bool) -> list[list[int]]:
    """Row ids each data-parallel rank sees in one shared epoch.

    Every rank gets the same count, which keeps ranks in lockstep: ``floor(N / world)`` when ``drop_last``
    trims the remainder, ``ceil(N / world)`` when the sampler pads the tail by repeating rows.
    """
    per_rank = NUM_SAMPLES // dp_world_size if drop_last else math.ceil(NUM_SAMPLES / dp_world_size)
    result = []
    for rank in range(dp_world_size):
        ids = epoch_ids(make_loader(path, dp_world_size=dp_world_size, dp_rank=rank, drop_last=drop_last))
        assert len(ids) == per_rank
        result.append(ids)
    return result


# Dataset


def test_len(id_dataset):
    assert len(make_dataset(id_dataset)) == NUM_SAMPLES


def test_getitem_shifts_for_causal_lm(id_dataset):
    dataset = make_dataset(id_dataset)
    for i in range(NUM_SAMPLES):
        inputs, labels = dataset[i]
        assert inputs["input"].tolist() == [i]
        assert inputs["positions"].tolist() == [0]
        assert labels.tolist() == [EOS_ID]


@pytest.mark.parametrize("seq_len", [1, 2, 4, 8])
def test_short_samples_are_right_padded(id_dataset, seq_len):
    """Inputs pad with eos, labels with ``IGNORE_INDEX``, and positions with 1.

    Positions are not padded with 0, because a 0 marks a document start and would add fake document boundaries
    for varlen attention; 1 merges the padding into the last document.
    """
    inputs, labels = make_dataset(id_dataset, seq_len=seq_len)[0]
    pad_len = seq_len - 1

    assert inputs["input"].tolist() == [0] + [EOS_ID] * pad_len
    assert labels.tolist() == [EOS_ID] + [IGNORE_INDEX] * pad_len
    assert inputs["positions"].tolist() == [0] + [1] * pad_len


def test_parallel_load_matches_serial(id_dataset):
    serial = make_dataset(id_dataset)
    parallel = make_dataset(id_dataset, num_initial_load_workers=2)

    assert len(serial) == len(parallel)
    for i in range(len(serial)):
        (serial_inputs, serial_labels), (parallel_inputs, parallel_labels) = serial[i], parallel[i]
        assert torch.equal(serial_inputs["input"], parallel_inputs["input"])
        assert torch.equal(serial_labels, parallel_labels)


def test_count_rows_and_steps_per_epoch(id_dataset):
    assert count_rows(str(id_dataset)) == NUM_SAMPLES
    assert steps_per_epoch(str(id_dataset), dp_world_size=2, local_batch_size=3) == NUM_SAMPLES // 2 // 3


def test_missing_path_raises():
    with pytest.raises(ValueError, match="dataset_path must be provided"):
        PretokenizedDataset(dataset_path=None, tokenizer=DummyTokenizer(), seq_len=1)


def test_nonexistent_path_raises(tmp_path):
    with pytest.raises(ValueError, match="does not exist"):
        make_dataset(tmp_path / "missing")


def test_directory_without_shards_raises(tmp_path):
    with pytest.raises(ValueError, match=r"No \.arrow shards"):
        make_dataset(tmp_path)


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        ({"input_ids": [[5, 6, 7]]}, "missing required keys"),
        ({"input_ids": [[5, 6, 7]], "position_ids": [[0, 1]]}, "position_ids"),
        ({"input_ids": [[5, 6, 7]], "position_ids": [[0, 1, 2]], "token_type_ids": [[3, 3]]}, "token_type_ids"),
        ({"input_ids": [[5, 6, 7, 8]], "position_ids": [[0, 1, 2, 3]]}, "seq_len \\+ 1"),
    ],
)
def test_malformed_rows_raise(tmp_path, rows, match):
    dataset = make_dataset(save_shards(tmp_path / "bad", rows, num_shards=1), seq_len=2)
    with pytest.raises(ValueError, match=match):
        dataset[0]


# Loss masking


def test_rows_without_token_type_ids_train_on_every_token(tmp_path):
    """Pretraining rows carry no ``token_type_ids``: every real token is a label, two packed documents included."""
    path = save_shards(
        tmp_path / "pretraining",
        {"input_ids": [[10, 11, EOS_ID, 20, 21, EOS_ID]], "position_ids": [[0, 1, 2, 0, 1, 2]]},
        num_shards=1,
    )
    inputs, labels = make_dataset(path, seq_len=7)[0]

    assert inputs["input"].tolist() == [10, 11, EOS_ID, 20, 21, EOS_ID, EOS_ID]
    assert inputs["positions"].tolist() == [0, 1, 2, 0, 1, 1, 1]
    assert labels.tolist() == [11, EOS_ID, 20, 21, EOS_ID, IGNORE_INDEX, IGNORE_INDEX]


CHAT_ROW = {
    "input_ids": [[10, 11, 12, 13, 14, 15]],
    "position_ids": [[0, 1, 2, 3, 4, 5]],
    "token_type_ids": [
        [
            role2id["system"],
            role2id["user"],
            role2id["user"],
            role2id["assistant"],
            role2id["assistant"],
            role2id["tool"],
        ]
    ],
}


@pytest.mark.parametrize(
    ("loss_enabled_turns", "expected_labels"),
    [
        # Labels are input_ids[1:], masked by the role of the token they predict (token_type_ids[1:]).
        (("assistant",), [IGNORE_INDEX, IGNORE_INDEX, 13, 14, IGNORE_INDEX]),
        (("user", "assistant"), [11, 12, 13, 14, IGNORE_INDEX]),
        (("system", "user", "assistant", "tool"), [11, 12, 13, 14, 15]),
        (("tool",), [IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, IGNORE_INDEX, 15]),
    ],
)
def test_token_type_ids_mask_the_loss_by_role(tmp_path, loss_enabled_turns, expected_labels):
    path = save_shards(tmp_path / "chat", CHAT_ROW, num_shards=1)
    inputs, labels = make_dataset(path, seq_len=5, loss_enabled_turns=loss_enabled_turns)[0]

    assert inputs["input"].tolist() == [10, 11, 12, 13, 14]
    assert labels.tolist() == expected_labels


def test_unknown_loss_enabled_turn_raises(id_dataset):
    with pytest.raises(KeyError):
        make_dataset(id_dataset, loss_enabled_turns=("narrator",))


# Dataloader: order, shuffling and resume


def test_no_shuffle_preserves_order(id_dataset):
    assert take_ids(make_loader(id_dataset, shuffle=False), NUM_SAMPLES) == list(range(NUM_SAMPLES))


def test_shuffle_permutes_order(id_dataset):
    order = take_ids(make_loader(id_dataset), NUM_SAMPLES)
    assert sorted(order) == list(range(NUM_SAMPLES))
    assert order != list(range(NUM_SAMPLES))


def test_shuffle_is_reproducible_per_seed(id_dataset):
    first = take_ids(make_loader(id_dataset, shuffle_seed=42), NUM_SAMPLES)
    assert take_ids(make_loader(id_dataset, shuffle_seed=42), NUM_SAMPLES) == first
    assert take_ids(make_loader(id_dataset, shuffle_seed=99), NUM_SAMPLES) != first


def test_epoch_advances_on_reloop(id_dataset):
    ids = take_ids(make_loader(id_dataset, infinite=True), 2 * NUM_SAMPLES)
    first_epoch, second_epoch = ids[:NUM_SAMPLES], ids[NUM_SAMPLES:]

    assert sorted(first_epoch) == sorted(second_epoch) == list(range(NUM_SAMPLES))
    assert first_epoch != second_epoch


@pytest.mark.parametrize(
    ("steps_before_save", "save_workers", "resume_workers"),
    [
        (6, 0, 0),  # within the first epoch
        (25, 0, 0),  # in the second epoch: the sampler epoch must be checkpointed too
        (25, 0, 2),  # saved by a serially loaded loader, resumed by a parallel one
        (25, 2, 0),
    ],
)
def test_exact_resume(id_dataset, steps_before_save, save_workers, resume_workers):
    reference = take_ids(make_loader(id_dataset, infinite=True), steps_before_save + 5)

    saved = make_loader(id_dataset, infinite=True, num_initial_load_workers=save_workers)
    iterator = iter(saved)
    for _ in range(steps_before_save):
        next(iterator)
    state = saved.state_dict()

    resumed = make_loader(id_dataset, infinite=True, num_initial_load_workers=resume_workers)
    resumed.load_state_dict(state)

    assert take_ids(resumed, 5) == reference[steps_before_save:]


# Dataloader: data parallel and drop_last


@pytest.mark.parametrize("dp_world_size", [2, 4])
def test_ranks_split_an_epoch_without_duplicates(id_dataset, dp_world_size):
    ids = [i for rank_ids in per_rank_epoch_ids(id_dataset, dp_world_size, drop_last=True) for i in rank_ids]
    assert sorted(ids) == list(range(NUM_SAMPLES))


def test_drop_last_trims_uneven_ranks(id_dataset):
    """With 20 rows over 3 ranks, ``drop_last=True`` gives each rank 6 rows and drops the other 2."""
    ids = [i for rank_ids in per_rank_epoch_ids(id_dataset, 3, drop_last=True) for i in rank_ids]

    assert len(ids) == len(set(ids)) == 18


def test_no_drop_last_pads_uneven_ranks(id_dataset):
    """With 20 rows over 3 ranks, ``drop_last=False`` gives each rank 7 rows, repeating one to cover all 20."""
    ids = [i for rank_ids in per_rank_epoch_ids(id_dataset, 3, drop_last=False) for i in rank_ids]

    assert len(ids) == 21
    assert set(ids) == set(range(NUM_SAMPLES))


def test_drop_last_controls_trailing_partial_batch(id_dataset):
    """Batches of 3 over 20 rows: 6 full batches and a partial batch of 2, kept only without ``drop_last``."""
    kept = epoch_ids(make_loader(id_dataset, shuffle=False, drop_last=False, local_batch_size=3))
    dropped = epoch_ids(make_loader(id_dataset, shuffle=False, drop_last=True, local_batch_size=3))

    assert kept == list(range(NUM_SAMPLES))
    assert dropped == list(range(18))


# Dataloader: config checks and max_docs_per_batch


def test_dataset_name_is_rejected(id_dataset):
    with pytest.raises(ValueError, match="dataset argument should be empty"):
        make_loader(id_dataset, dataset="c4")


def test_max_docs_per_batch_requires_infinite(id_dataset):
    with pytest.raises(ValueError, match="max_docs_per_batch is set but infinite=False"):
        make_loader(id_dataset, infinite=False, max_docs_per_batch=1)


def test_max_docs_per_batch_skips_batches_with_too_many_documents(tmp_path):
    # Even rows pack one document, odd rows three. Documents are counted on the inputs, after the causal shift
    # drops the last position, so the resets must fall in the first ``seq_len`` positions.
    one_doc, three_docs = [0, 1, 2, 3, 4], [0, 1, 0, 0, 1]
    path = save_shards(
        tmp_path / "docs",
        {
            "input_ids": [[i, 5, 6, 7, EOS_ID] for i in range(NUM_SAMPLES)],
            "position_ids": [one_doc if i % 2 == 0 else three_docs for i in range(NUM_SAMPLES)],
        },
    )
    loader = make_loader(path, shuffle=False, infinite=True, seq_len=4, max_docs_per_batch=2)

    assert take_ids(loader, NUM_SAMPLES // 2) == list(range(0, NUM_SAMPLES, 2))
