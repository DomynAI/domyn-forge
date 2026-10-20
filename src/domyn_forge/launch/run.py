# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0

import os
import sys

import torch.distributed
import tyro
from torchtitan.tools.logging import init_logger, logger
from torchtitan.trainer import Trainer


def run(train_config: Trainer.Config) -> None:
    """Train with ``train_config``, applying CLI overrides from ``sys.argv``.

    Called from a run config's ``__main__`` block; torchrun executes the config
    file directly. Overrides use torchtitan's dotted syntax, e.g.
    ``--training.steps 200 --optimizer.lr 5e-5``.
    """
    init_logger()
    config = tyro.cli(type(train_config), args=sys.argv[1:], default=train_config)
    logger.info(f"Running on RANK:{os.environ.get('RANK')}, WORLD_SIZE:{os.environ.get('WORLD_SIZE')}")
    trainer = config.build()
    try:
        trainer.train()
    finally:
        trainer.close()
        if torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
