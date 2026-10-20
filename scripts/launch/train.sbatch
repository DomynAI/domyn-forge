#!/bin/bash
# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0
#
# Run a config file on several nodes with Slurm. Submit from the repository root:
#
#   sbatch --account=<account> scripts/launch/train.sbatch scripts/launch/configs/domynedge/edgethinker.py
#
# Extra arguments are forwarded as config overrides, e.g. --training.steps 100.
# Adjust the #SBATCH lines below (or override them on the sbatch command line) for your slurm-based cluster.

#SBATCH --job-name=domyn-forge
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --partition=boost_usr_prod

set -ex

if [ $# -lt 1 ]; then
    echo "usage: sbatch $0 <config.py> [overrides...]" >&2
    exit 1
fi

# The first node hosts the torchrun rendezvous.
HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
HEAD_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$HEAD_NODE" hostname --ip-address)

export LOG_RANK=${LOG_RANK:-0}
export PYTHONUNBUFFERED=1

# One torchrun per node (--ntasks-per-node=1), each starting one process per GPU.
# Compute nodes may have no internet access, hence --no-sync: run `uv sync` on the login node first.
PYTORCH_ALLOC_CONF="expandable_segments:True" \
srun uv run --no-sync torchrun \
    --nnodes="$SLURM_NNODES" \
    --nproc_per_node="$SLURM_GPUS_ON_NODE" \
    --rdzv_backend c10d \
    --rdzv_id "$SLURM_JOB_ID" \
    --rdzv_endpoint "$HEAD_NODE_IP:29500" \
    --local-ranks-filter "$LOG_RANK" --role rank --tee 3 \
    "$@"
