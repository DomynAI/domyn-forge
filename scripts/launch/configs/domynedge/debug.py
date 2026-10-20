# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

"""DomynEdge debug model on a packed copy of TorchTitan's c4_test sample (single node).

Build the dataset once, then train on a local / interactive node:

    uv run scripts/data/pack_c4_test.py
    NGPU=2 ./scripts/launch/train.sh scripts/launch/configs/domynedge/debug.py
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

from domyn_forge.datasets import PretokenizedDataLoader
from domyn_forge.launch import run
from domyn_forge.models.domynedge import model_registry

# Paths are relative to the repo root.
TRAIN_CONFIG = Trainer.Config(
    hf_assets_path="assets/domyn-edge/tokenizers/pretraining",
    metrics=MetricsProcessor.Config(log_freq=1),
    model_spec=model_registry("debugmodel", vocab_size=115328),
    # Built by scripts/data/pack_c4_test.py with the same tokenizer and seq_len.
    dataloader=PretokenizedDataLoader.Config(dataset_path="data/c4_test_packed"),
    parallelism=ParallelismConfig(spmd_backend="partial_dtensor"),
    optimizer=default_adamw(lr=8e-4),
    loss=CrossEntropyLoss.Config(),
    lr_scheduler=LRSchedulersContainer.Config(
        warmup_steps=2,
        decay_ratio=0.8,
        decay_type="linear",
        min_lr_factor=0.0,
    ),
    training=TrainingConfig(
        local_batch_size=8,
        seq_len=2048,
        steps=1000,
        disable_cuda_graphs=True,
    ),
    checkpoint=CheckpointManager.Config(
        enable=True,
        interval=1000,
        last_save_model_only=False,
    ),
    activation_checkpoint=SelectiveAC.Config(),
)


if __name__ == "__main__":
    run(TRAIN_CONFIG)
