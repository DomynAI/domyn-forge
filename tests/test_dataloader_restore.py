# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""Integration test: training resumed from a checkpoint sees exactly the same batches and losses.

Each case trains the DomynEdge debug model for 20 steps on 4 GPUs (2 replicate x 2 shard data parallel) with
checkpoints at steps 10 and 20, deletes the step-20 checkpoint, and trains again: the second run resumes from step
10 and must reproduce the batches, losses and validation batches of the first.

The pretokenized shards are synthetic: packed documents of random tokens, each ending with EOS, with positions
restarting at 0.
"""

import dataclasses
import os
import pathlib
import shutil
import socket

import numpy as np
import pytest
import torch
import torchtitan.components.checkpoint
import torchtitan.components.loss
import torchtitan.components.optimizer
import torchtitan.components.validate
import torchtitan.config
import torchtitan.tools.logging
import torchtitan.trainer
from datasets import Dataset
from torch.multiprocessing.spawn import spawn
from torchtitan.components.loss import IGNORE_INDEX

from domyn_forge.datasets import PretokenizedDataLoader
from domyn_forge.models.domynedge import model_registry

NUM_GPUS = 4

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(torch.cuda.device_count() < NUM_GPUS, reason=f"needs {NUM_GPUS} GPUs"),
]

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TOKENIZER_DIR = REPO_ROOT / "assets" / "domyn-edge" / "tokenizers" / "pretraining"
# Padding uses the tokenizer's EOS id, so the debug model needs the full EdgeThinker vocabulary.
VOCAB_SIZE = 115328
EOS_ID = 115257
SEQ_LEN = 128
NUM_ROWS = 256
NUM_SHARDS = 10


def write_packed_shards(path: pathlib.Path, seed: int = 0) -> pathlib.Path:
    """Write ``NUM_ROWS`` rows of ``SEQ_LEN + 1`` tokens packing random documents that end with EOS.

    One row in eight stops short of ``SEQ_LEN + 1``, so right padding is exercised too.
    """
    rng = np.random.default_rng(seed)
    input_ids, position_ids = [], []
    for row in range(NUM_ROWS):
        row_len = SEQ_LEN + 1 if row % 8 else int(rng.integers(SEQ_LEN // 2, SEQ_LEN))
        tokens, positions = [], []
        while len(tokens) < row_len:
            doc_len = min(int(rng.integers(8, SEQ_LEN)), row_len - len(tokens))
            tokens += [*rng.integers(3, 2048, doc_len - 1).tolist(), EOS_ID]
            positions += list(range(doc_len))
        input_ids.append(tokens)
        position_ids.append(positions)
    Dataset.from_dict({"input_ids": input_ids, "position_ids": position_ids}).save_to_disk(path, num_shards=NUM_SHARDS)
    return path


def record_batch(inputs: torch.Tensor, labels: torch.Tensor, extra_kwargs: dict) -> dict[str, torch.Tensor]:
    return {
        "inputs": inputs.detach().cpu().clone(),
        "labels": labels.detach().cpu().clone(),
        **{k: v.detach().cpu().clone() for k, v in extra_kwargs.items() if isinstance(v, torch.Tensor)},
    }


class RecordingTrainer(torchtitan.trainer.Trainer):
    """Trainer that records every training batch and loss."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(torchtitan.trainer.Trainer.Config):
        pass

    def __init__(self, *args, **kwargs):
        self.registered_batches = []
        self.registered_losses = []
        super().__init__(*args, **kwargs)

    def post_dataloading_process(self, *args, **kwargs):
        inputs, labels, extra_kwargs = super().post_dataloading_process(*args, **kwargs)
        self.registered_batches.append(record_batch(inputs, labels, extra_kwargs))
        return inputs, labels, extra_kwargs

    def forward_backward_step(self, *args, **kwargs):
        loss = super().forward_backward_step(*args, **kwargs)
        self.registered_losses.append(loss.detach().cpu().clone())
        return loss


class RecordingValidator(torchtitan.components.validate.Validator):
    """Validator that records the batches of every validation run."""

    @dataclasses.dataclass(kw_only=True, slots=True)
    class Config(torchtitan.components.validate.Validator.Config):
        pass

    def __init__(self, *args, **kwargs):
        self.registered_batches = []
        self.current_batches = []
        super().__init__(*args, **kwargs)

    def post_dataloading_process(self, *args, **kwargs):
        inputs, labels, extra_kwargs = super().post_dataloading_process(*args, **kwargs)
        self.current_batches.append(record_batch(inputs, labels, extra_kwargs))
        return inputs, labels, extra_kwargs

    def validate(self, *args, **kwargs):
        self.current_batches = []
        result = super().validate(*args, **kwargs)
        self.registered_batches.append(self.current_batches)
        return result


def find_free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def train_worker(rank: int, world_size: int, port: int, config: RecordingTrainer.Config) -> None:
    os.environ.update(
        MASTER_ADDR="localhost",
        MASTER_PORT=str(port),
        LOCAL_RANK=str(rank),
        RANK=str(rank),
        WORLD_SIZE=str(world_size),
    )
    torchtitan.tools.logging.init_logger()
    trainer = config.build()
    assert isinstance(trainer, RecordingTrainer)
    try:
        trainer.train()
        assert isinstance(trainer.validator, RecordingValidator)
        dump = pathlib.Path(config.dump_folder)
        torch.save(trainer.registered_batches, dump / f"batches_{rank}.pt")
        torch.save(trainer.registered_losses, dump / f"losses_{rank}.pt")
        torch.save(trainer.validator.registered_batches, dump / f"valid_batches_{rank}.pt")
    finally:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def train(config: RecordingTrainer.Config) -> tuple[list, list, list]:
    """Train on ``NUM_GPUS`` processes and return rank 0's batches, losses and validation batches."""
    spawn(train_worker, args=(NUM_GPUS, find_free_port(), config), nprocs=NUM_GPUS, join=True)
    dump = pathlib.Path(config.dump_folder)
    batches, losses, valid_batches = (
        torch.load(dump / f"{name}_0.pt", weights_only=False) for name in ("batches", "losses", "valid_batches")
    )
    return batches, losses, valid_batches


@pytest.mark.parametrize(
    ("num_workers", "num_initial_load_workers", "prefetch_factor", "attn_backend"),
    [
        (1, 0, 2, "varlen"),
        (1, 1, 2, "varlen"),
        (1, 0, 8, "flex"),
    ],
)
def test_resume_reproduces_batches_and_losses(
    tmp_path, num_workers, num_initial_load_workers, prefetch_factor, attn_backend
):
    dataset_path = str(write_packed_shards(tmp_path / "data"))
    dump_folder = tmp_path / "run"

    config = RecordingTrainer.Config(
        dump_folder=dump_folder.as_posix(),
        hf_assets_path=TOKENIZER_DIR.as_posix(),
        model_spec=model_registry("debugmodel", attn_backend=attn_backend, vocab_size=VOCAB_SIZE),
        dataloader=PretokenizedDataLoader.Config(
            dataset_path=dataset_path,
            infinite=True,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            shuffle=True,
            shuffle_seed=42,
            num_initial_load_workers=num_initial_load_workers,
        ),
        optimizer=torchtitan.components.optimizer.default_adamw(lr=8e-4),
        # CUDA graphs record one training step and replay it, which needs every step to do the same thing. DomynEdge
        # rebuilds its attention masks from each batch's document boundaries: with varlen the number of documents
        # changes the tensor shapes and the replay fails, and with flex the replay seems to reuse the recorded batch's
        # mask.
        training=torchtitan.config.TrainingConfig(
            local_batch_size=4, seq_len=SEQ_LEN, steps=20, disable_cuda_graphs=True
        ),
        loss=torchtitan.components.loss.CrossEntropyLoss.Config(),
        checkpoint=torchtitan.components.checkpoint.CheckpointManager.Config(interval=10, enable=True),
        validator=RecordingValidator.Config(
            enable=True,
            steps=2,
            freq=10,
            dataloader=PretokenizedDataLoader.Config(
                dataset_path=dataset_path,
                infinite=False,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                shuffle=False,
            ),
        ),
        parallelism=torchtitan.config.ParallelismConfig(
            data_parallel_replicate_degree=2,
            data_parallel_shard_degree=2,
            spmd_backend="partial_dtensor",
        ),
        debug=torchtitan.config.DebugConfig(deterministic=True, seed=42),
    )

    # Train steps 1-20 with checkpoints at 10 and 20, then drop step 20 so the second run resumes from step 10.
    batches_before, losses_before, valid_batches_before = train(config)
    shutil.rmtree(dump_folder / "checkpoint" / "step-20")
    batches_after, losses_after, valid_batches_after = train(config)

    # The resumed run covers steps 11-20, which must match the last 10 steps of the first run.
    assert len(batches_after) == len(losses_after) == 10
    for before, after in zip(batches_before[-10:], batches_after, strict=True):
        for key in before:
            assert torch.equal(before[key], after[key]), f"{key} differs after resuming"
    for before, after in zip(losses_before[-10:], losses_after, strict=True):
        torch.testing.assert_close(before, after, atol=1e-4, rtol=0)

    # Validation is a fresh, unshuffled pass each time, so every run sees the same batches.
    reference = valid_batches_before[0]
    for run in [*valid_batches_before, *valid_batches_after]:
        for batch, reference_batch in zip(run, reference, strict=True):
            for key in batch:
                assert torch.equal(batch[key], reference_batch[key]), f"validation {key} differs between runs"

    for batch in batches_after:
        inputs, labels, positions = batch["inputs"], batch["labels"], batch["positions"]
        assert torch.all(positions < SEQ_LEN)
        assert torch.all(positions[:, 0] == 0)
        # A document restart must follow an EOS; within a document, positions count up by one.
        restarts = positions[:, 1:] == 0
        assert torch.all(inputs[:, :-1][restarts] == EOS_ID)
        real = labels[:, 1:] != IGNORE_INDEX
        assert torch.all((positions[:, 1:] == positions[:, :-1] + 1)[real & ~restarts])
        # Padding is filled with EOS and never trained on.
        assert torch.all(inputs[labels == IGNORE_INDEX] == EOS_ID)
