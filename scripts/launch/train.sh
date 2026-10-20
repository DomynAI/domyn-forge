#!/usr/bin/bash
# SPDX-FileCopyrightText: 2026 Domyn
# SPDX-License-Identifier: Apache-2.0
#
# Run a config file on a single node (local / interactive).
#
#   NGPU=4 ./scripts/launch/train.sh scripts/launch/configs/domynedge/debug.py
#
# Extra arguments are forwarded as config overrides, e.g. --training.steps 100.
set -ex

if [ $# -lt 1 ]; then
    echo "usage: $0 <config.py> [overrides...]" >&2
    exit 1
fi
CONFIG_FILE=$1
shift

NGPU=${NGPU:-"8"}
export LOG_RANK=${LOG_RANK:-0}

PYTORCH_ALLOC_CONF="expandable_segments:True" \
uv run torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
    --local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
    ${CONFIG_FILE} "$@"
