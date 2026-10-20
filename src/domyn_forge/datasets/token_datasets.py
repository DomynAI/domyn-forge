# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import glob
import multiprocessing as mp
from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pyarrow as pa
import torch
from datasets import Dataset as ArrowDataset
from datasets import concatenate_datasets
from torch.nn.functional import pad
from torch.utils.data import Dataset, DistributedSampler
from torchtitan.components.dataloader import ParallelAwareDataloader
from torchtitan.components.loss import IGNORE_INDEX
from torchtitan.components.tokenizer import BaseTokenizer
from torchtitan.tools.logging import logger

role2id = {
    "system": 1,
    "user": 2,
    "assistant": 3,
    "tool": 4,
}


def _pad_sample(
    sample: dict | pa.Table,
    seq_len: int,
    eos_id: int,
    loss_enabled_turns: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    """Shift a pretokenized sample for causal LM and right-pad it to ``seq_len``.

    Turns a stored ``PretokenizedDataset`` row into a training batch.
    ``sample`` must contain ``input_ids`` / ``position_ids`` of length
    ``<= seq_len + 1`` (shorter samples are right-padded to ``seq_len`` after the
    causal shift).
    """
    # If a dataset is in `with_format("arrow")`, indexing returns a single-row pyarrow.Table.
    # Convert to a plain dict of python lists so the logic below operates on per-token sequences.
    if isinstance(sample, pa.Table):
        sample = {k: v[0] for k, v in sample.to_pydict().items()}

    field_names = list(sample.keys())
    if "input_ids" not in field_names or "position_ids" not in field_names:
        msg = f"Sample is missing required keys: {field_names}"
        raise ValueError(msg)
    if len(sample["input_ids"]) != len(sample["position_ids"]):
        msg = (
            "Length mismatch between input_ids and position_ids: "
            f"{len(sample['input_ids'])} vs {len(sample['position_ids'])}"
        )
        raise ValueError(msg)
    if len(sample["input_ids"]) != len(sample.get("token_type_ids", sample["input_ids"])):
        msg = (
            "Length mismatch between input_ids and token_type_ids: "
            f"{len(sample['input_ids'])} vs {len(sample.get('token_type_ids', sample['input_ids']))}"
        )
        raise ValueError(msg)
    if len(sample["input_ids"]) > seq_len + 1:
        msg = f"Expected input_ids length <= seq_len + 1, got {len(sample['input_ids'])}"
        raise ValueError(msg)

    # Shift input_ids and positions for causal language modeling, then pad to seq_len.
    input_ids = torch.tensor(sample["input_ids"][:-1], dtype=torch.long)
    positions = torch.tensor(sample["position_ids"][:-1], dtype=torch.long)
    labels = torch.tensor(sample["input_ids"][1:], dtype=torch.long)
    token_type_ids = torch.tensor(
        sample.get("token_type_ids", [role2id["assistant"]] * len(sample["input_ids"]))[1:],  # fmt off
        dtype=torch.long,
    )

    # pad has the form (padding_left, padding_right); (0, pad_len) pads on the right.
    pad_len = seq_len - len(input_ids)
    input_ids = pad(input_ids, (0, pad_len), value=eos_id)
    positions = torch.cat([positions, torch.full((pad_len,), 1, dtype=torch.long)])
    labels = pad(labels, (0, pad_len), value=IGNORE_INDEX)
    # token type ids can be padded with any number.
    # it used to determine which tokens contribute to the loss.
    # padding tokens are already masked out by the labels,
    # so it does not matter what value is used for padding.
    token_type_ids = pad(token_type_ids, (0, pad_len), value=IGNORE_INDEX)

    return {
        # This can happen for document padded data used for ring-attention.
        # It is safe to always replace IGNORE_INDEX with eos_id even if it
        # does not normally occur in the input_ids.
        "input": torch.where(input_ids == IGNORE_INDEX, torch.tensor(eos_id, dtype=torch.long), input_ids),
        "positions": positions,
    }, torch.where(
        torch.isin(token_type_ids, loss_enabled_turns),
        labels,
        torch.full_like(labels, IGNORE_INDEX),
    )


def count_rows(dataset_path: str) -> int:
    """Return the number of rows ``PretokenizedDataset`` would load from ``dataset_path``.

    Shards are memory-mapped, so this reads only the Arrow metadata, not the tokens.
    """
    files = sorted(glob.glob(f"{dataset_path}/*.arrow"))
    if not files:
        msg = f"No .arrow shards found in {dataset_path}"
        raise ValueError(msg)
    return sum(len(ArrowDataset.from_file(f)) for f in files)


def steps_per_epoch(dataset_path: str, dp_world_size: int, local_batch_size: int) -> int:
    """Return the training steps in one pass over ``dataset_path`` with ``drop_last=True``.

    Matches ``PretokenizedDataLoader``: the ``DistributedSampler`` gives each
    rank ``rows // dp_world_size`` rows, which are then cut into full batches.
    """
    return count_rows(dataset_path) // dp_world_size // local_batch_size


class PretokenizedDataset(Dataset):
    """Map-style pretokenized dataset backed by memory-mapped Arrow.

    Reads every ``*.arrow`` shard in ``dataset_path`` via ``Dataset.from_file``
    and concatenates them, so rows stay random-access and memory-mapped (not all
    held in RAM) and no ``save_to_disk`` metadata is required. Random access is
    what lets a sampler shuffle indices and lets ``StatefulDataLoader`` checkpoint
    the exact stream position, so a shuffled run resumes on the exact next sample.

    Each row must hold ``input_ids`` / ``position_ids`` lists of length
    ``seq_len + 1``.
    """

    def __init__(
        self,
        dataset_path: str | None,
        tokenizer: BaseTokenizer,
        seq_len: int,
        num_initial_load_workers: int = 0,
        loss_enabled_turns: Sequence[str] = ("assistant",),
    ) -> None:
        """Load and concatenate every ``*.arrow`` shard found in ``dataset_path``.

        Args:
            dataset_path: Directory containing the pretokenized ``*.arrow`` shards.
            tokenizer: Tokenizer whose ``eos_id`` is used to pad inputs.
            seq_len: Training sequence length samples are shifted and padded to.
            num_initial_load_workers: If > 0, load shards in parallel with a
                ``spawn`` multiprocessing pool of this size; otherwise load serially.
            loss_enabled_turns: Turn roles whose tokens contribute to the loss.

        Raises:
            ValueError: If ``dataset_path`` is None, does not exist, or contains no
                ``*.arrow`` shards.
        """
        if dataset_path is None:
            msg = "dataset_path must be provided for PretokenizedDataset."
            raise ValueError(msg)
        if not Path(dataset_path).exists():
            msg = f"Dataset path {dataset_path} does not exist."
            raise ValueError(msg)

        logger.info(f"Loading map-style dataset from {dataset_path}")

        # sorted() so the index->row mapping is identical across ranks and resumes,
        # which the DistributedSampler relies on for a correct, resumable shuffle.
        files = sorted(glob.glob(f"{dataset_path}/*.arrow"))
        if not files:
            msg = f"No .arrow shards found in {dataset_path}"
            raise ValueError(msg)

        if num_initial_load_workers > 0:
            logger.info(f"Loading with {num_initial_load_workers} workers in parallel of {len(files)} files")
            # Use spawn to avoid the warning "use of fork() may lead to deadlocks in the child." since
            # fork (default context)    copies the parent process's entire memory space into the child.
            # Fast to start, but the child inherits everything — including CUDA state, locks, threads.
            with mp.get_context("spawn").Pool(num_initial_load_workers) as pool:
                datasets = list(pool.imap(ArrowDataset.from_file, files))
        else:
            datasets = [ArrowDataset.from_file(f) for f in files]

        data = concatenate_datasets(datasets)

        # Ensure rows come back as plain dicts of per-token lists (what _pad_sample
        # expects), not a format-specific representation such as pyarrow.Table.
        self._data = data.with_format(None)
        self.seq_len = seq_len

        assert tokenizer.eos_id is not None
        self._eos_id: int = tokenizer.eos_id  # eos_id is used for padding, so it must be present

        self._loss_enabled_turns = torch.tensor(list(map(role2id.__getitem__, loss_enabled_turns)), dtype=torch.long)

    def __len__(self) -> int:
        """Return the number of rows across all loaded shards."""
        return len(self._data)

    def __getitem__(self, idx: int):
        """Return the shifted, right-padded ``(inputs, labels)`` pair for row ``idx``."""
        return _pad_sample(self._data[idx], self.seq_len, self._eos_id, self._loss_enabled_turns)


class PretokenizedDataLoader(ParallelAwareDataloader):
    """Resumable, shuffling dataloader for pretokenized samples.

    Shuffling happens at the dataloader level via a ``DistributedSampler`` whose order is a deterministic function
    of ``(shuffle_seed, epoch)``; ``StatefulDataLoader`` checkpoints how far into that order we are, so a reloaded run
    resumes on the exact next sample.

    Use with a directory of pretokenized ``*.arrow`` shards (see ``PretokenizedDataset``).
    """

    @dataclass(kw_only=True, slots=True)
    class Config(ParallelAwareDataloader.Config):
        """Configuration for ``PretokenizedDataLoader``."""

        infinite: bool = True
        """Loop forever, advancing the sampler epoch each pass (training).

        Set False for a single pass (e.g. validation)."""

        shuffle: bool = True
        """Shuffle sample order via the sampler. The order is fully checkpointable and resumes exactly."""

        shuffle_seed: int = 42
        """Base seed; pass N uses the sampler's (shuffle_seed, epoch=N) permutation."""

        drop_last: bool = False
        """Drop the trailing partial batch; the sampler also pad-trims so every
        rank sees an equal number of samples without resampling."""

        num_initial_load_workers: int = 0
        """Number of workers to use for the initial load of the dataset shards,
        that is the loading with ArrowDataset.from_file. If 0, the loading is done in
        the main process. If > 0, a multiprocessing pool is used to load the shards
        in parallel. This can speed up the initial loading of large datasets."""

        loss_enabled_turns: list[str] = field(default_factory=lambda: ["assistant"])
        """Set of turn roles whose tokens contribute to the loss. Tokens from
        other roles are masked to IGNORE_INDEX. The default is {"assistant"} for
        instruction-tuned LLMs, but you can add "user" or "system" if you want to
        train on those tokens too."""

        max_docs_per_batch: int | None = None
        """Skip batches packing more than this many documents. ``None`` (the
        default) keeps every batch.

        Varlen attention turns every document boundary into a cu_seqlens segment,
        and flash-attn's varlen backward sizes its fp32 dq accumulator as
        ``(total_q + num_docs * 128) * num_heads * head_dim``. The ``num_docs * 128``
        term is pure padding, so a batch packing thousands of near-empty documents
        can demand tens of GiB for a buffer that is over 90% padding, and OOM a GPU
        that handles ordinary batches with room to spare. Set this when a corpus is
        known to contain such degenerate rows; skipping them costs a negligible
        fraction of the data."""

    def __init__(
        self,
        config: Config,
        *,
        dp_world_size: int,
        dp_rank: int,
        tokenizer: BaseTokenizer,
        seq_len: int,
        local_batch_size: int,
        **kwargs,  # noqa: ARG002 -- the torchtitan Config.build passes extra keywords that are unused here.
    ):
        """Build the map-style dataset, its ``DistributedSampler``, and the dataloader.

        Args:
            config: Dataloader configuration.
            dp_world_size: Number of data-parallel ranks.
            dp_rank: Data-parallel rank of this process.
            tokenizer: Tokenizer passed to the underlying dataset.
            seq_len: Training sequence length.
            local_batch_size: Per-rank batch size.
            **kwargs: Extra keywords from the torchtitan builder; ignored.

        Raises:
            ValueError: If ``config.dataset`` is set, or if ``max_docs_per_batch`` is
                set while ``infinite`` is False.
        """
        if config.dataset != "":
            msg = "dataset argument should be empty, dataset is read from dataset_path."
            raise ValueError(msg)

        if config.max_docs_per_batch is not None and not config.infinite:
            msg = (
                "max_docs_per_batch is set but infinite=False, which would hang "
                "training. Each rank decides on its own batches, so ranks skip a "
                "different number of times; over a single finite pass the ranks "
                "that skipped most run out of batches first, and the ranks still "
                "iterating then block forever in a collective waiting for them. "
                "With infinite=True the epoch rolls over instead of ending, so "
                "every rank keeps yielding one batch per step no matter how many "
                "it skipped. Either set infinite=True or leave max_docs_per_batch "
                "unset (None) to keep every batch."
            )
            raise ValueError(msg)

        dataset = PretokenizedDataset(
            dataset_path=config.dataset_path,
            tokenizer=tokenizer,
            seq_len=seq_len,
            num_initial_load_workers=config.num_initial_load_workers,
            loss_enabled_turns=config.loss_enabled_turns,
        )

        # Explicit num_replicas/rank so this works without an initialized
        # process group (e.g. single-process tests). With drop_last the sampler
        # trims to an even multiple of dp_world_size so all ranks stay in sync.
        sampler = DistributedSampler(
            dataset,
            num_replicas=dp_world_size,
            rank=dp_rank,
            shuffle=config.shuffle,
            seed=config.shuffle_seed,
            drop_last=config.drop_last,
        )

        dataloader_kwargs = {
            "sampler": sampler,
            "batch_size": local_batch_size,
            "drop_last": config.drop_last,
            "num_workers": config.num_workers,
            "persistent_workers": config.persistent_workers,
            "pin_memory": config.pin_memory,
            "prefetch_factor": config.prefetch_factor,
        }

        # torchtitan annotates `dataset` as IterableDataset, but only forwards it to
        # StatefulDataLoader, which accepts map-style datasets used with a sampler.
        super().__init__(
            dataset,  # pyright: ignore[reportArgumentType]
            dp_rank=dp_rank,
            dp_world_size=dp_world_size,
            **dataloader_kwargs,
        )

        self._sampler = sampler
        self._infinite = config.infinite
        self._max_docs_per_batch = config.max_docs_per_batch
        # Sampler epoch, persisted so shuffle order is reproducible across resumes.
        self._epoch = 0

    def _num_docs(self, inputs: dict[str, torch.Tensor]) -> int:
        """Return how many documents the batch packs.

        Document boundaries are positions that reset to 0, the same convention
        ``create_varlen_metadata_for_document`` uses to build cu_seqlens, so this
        counts exactly the segments varlen attention will see for the batch.
        """
        return int((inputs["positions"] == 0).sum())

    # StatefulDataLoader.__iter__ returns a _BaseDataLoaderIter; the trainer only uses
    # iter()/next(), and checkpoint state still flows through super().__iter__().
    def __iter__(  # pyright: ignore[reportIncompatibleMethodOverride]
        self,
    ) -> Iterator[tuple[dict[str, torch.Tensor], torch.Tensor]]:
        """Yield ``(inputs, labels)`` batches, advancing the sampler epoch on each pass.

        With ``infinite`` the iteration re-loops forever; otherwise it stops after
        one pass. Batches packing more than ``max_docs_per_batch`` documents are
        skipped.
        """
        while True:
            self._sampler.set_epoch(self._epoch)
            # After load_state_dict, the first StatefulDataLoader pass resumes
            # mid-epoch; once consumed, subsequent passes start fresh.
            for inputs, labels in super().__iter__():
                # Each rank decides from its own batch alone, so no collective is
                # needed: every rank still yields exactly one batch per step and
                # stays in lockstep, it just consumes a few more samples from its
                # own shard of the sampler order.
                if self._max_docs_per_batch is not None:
                    num_docs = self._num_docs(inputs)
                    if num_docs > self._max_docs_per_batch:
                        logger.warning(
                            f"Skipping batch packing {num_docs} documents, over the "
                            f"max_docs_per_batch={self._max_docs_per_batch} limit; it "
                            f"would add {num_docs * 128} padding rows to the varlen "
                            f"attention backward dq accumulator."
                        )
                        continue
                yield inputs, labels
            if not self._infinite:
                logger.warning("Map-style dataset has run out of data")
                return
            self._epoch += 1
            logger.warning(f"Map-style dataset is being re-looped (epoch {self._epoch})")

    def state_dict(self) -> dict:
        """Return the dataloader state, including the sampler epoch for this rank."""
        state = super().state_dict()
        # StatefulDataLoader only checkpoints the within-epoch position; persist
        # the epoch too so the sampler permutation matches on resume.
        state[f"{self._rank_id}_epoch"] = self._epoch
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore the sampler epoch (if present) and the dataloader state."""
        if state_dict:
            self._epoch = state_dict.get(f"{self._rank_id}_epoch", 0)
        super().load_state_dict(state_dict)
