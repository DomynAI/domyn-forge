# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""EdgeThinker on pretokenized ``*.arrow`` shards (template for real runs).

Fill in the tokenizer and dataset paths, then:

    NGPU=4 ./scripts/launch/train.sh scripts/launch/configs/domynedge/edgethinker.py

The dataset is a directory of shards from the pretokenization pipeline: rows
of ``input_ids`` / ``position_ids`` / optional ``token_type_ids`` with length
``seq_len + 1``.
"""

from torchtitan.components.checkpoint import CheckpointManager
from torchtitan.components.loss import CrossEntropyLoss
from torchtitan.components.metrics import MetricsProcessor
from torchtitan.components.optimizer import (
    LRSchedulersContainer,
    default_adamw,
)
from torchtitan.config import ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_checkpoint import SelectiveAC
from torchtitan.trainer import Trainer

from domyn_forge.datasets.token_datasets import PretokenizedDataLoader
from domyn_forge.launch import run
from domyn_forge.models.domynedge import model_registry

SEQ_LEN = 8192

TRAIN_CONFIG = Trainer.Config(
    hf_assets_path="/path/to/domyn/tokenizer",
    metrics=MetricsProcessor.Config(log_freq=10),
    model_spec=model_registry("edgethinker", max_seq_len=SEQ_LEN),
    dataloader=PretokenizedDataLoader.Config(
        dataset_path="/path/to/pretokenized/train",
        num_workers=1,
        persistent_workers=True,
        pin_memory=True,
        prefetch_factor=4,
        shuffle=True,
        shuffle_seed=42,
        drop_last=True,
    ),
    parallelism=ParallelismConfig(
        data_parallel_replicate_degree=1,
        data_parallel_shard_degree=-1,
        spmd_backend="partial_dtensor",
    ),
    optimizer=default_adamw(lr=1e-4),
    loss=CrossEntropyLoss.Config(),
    lr_scheduler=LRSchedulersContainer.Config(
        warmup_steps=50,
        decay_ratio=0.2,
        decay_type="linear",
        min_lr_factor=0.0,
    ),
    training=TrainingConfig(
        local_batch_size=1,
        seq_len=SEQ_LEN,
        steps=1000,
        disable_cuda_graphs=True,
    ),
    checkpoint=CheckpointManager.Config(
        enable=True,
        interval=500,
        last_save_model_only=False,
    ),
    activation_checkpoint=SelectiveAC.Config(),
)


if __name__ == "__main__":
    run(TRAIN_CONFIG)
