#!/usr/bin/env bash
# Stage, check, export and qualify one Domyn Edge LiteRT-LM bundle.
#
# Usage, from the repository root inside the toolchain image (release.Dockerfile):
#   scripts/litert/release.sh SOURCE_DIR RUN_DIR MAX_NLL_INCREASE
#
# SOURCE_DIR is a local download of domyn/Domyn-Edge-SFT. Every output, including the bundle, goes into RUN_DIR,
# which must not contain a previous run. The release workflow runs this script with networking disabled, so the
# workflow and a local run execute the same commands.
set -euo pipefail

if [[ $# -ne 3 ]]; then
    echo "Usage: $0 SOURCE_DIR RUN_DIR MAX_NLL_INCREASE" >&2
    exit 2
fi
source_dir=$1 run=$2 max_nll_increase=$3
if [[ -e "$run/bundle" || -e "$run/v5" ]]; then
    echo "$run already holds an export; use a new run directory" >&2
    exit 2
fi
mkdir -p "$run"
set -o noclobber
set -x

python -m pip freeze > "$run/toolchain-freeze.txt"
# Hash the weights and every script, recipe, template and case file this run can use (not the docs or evidence).
{
    sha256sum "$source_dir"/*.safetensors
    find scripts/litert -path scripts/litert/evidence -prune -o -type f \
        \( -name '*.py' -o -name '*.sh' -o -name '*.json' -o -name '*.jinja' -o -name '*.jsonl' \) -print \
        | LC_ALL=C sort | xargs sha256sum
} > "$run/input-sha256.txt"

python -m scripts.litert.export.stage_v5_checkpoint "$source_dir" "$run/v5"
# 1,027 tokens cross the 1,025-token sliding window.
python -m scripts.litert.checks.check_v5_parity "$source_dir" "$run/v5" --tokens 1027 --force-math --check migration \
    > "$run/migration-parity.jsonl"
python -m scripts.litert.checks.check_v5_parity "$source_dir" "$run/v5" --tokens 1027 --dtype float32 --force-math \
    --check canonical-cache --atol 0.001 --rtol 0.001 > "$run/cache-parity.jsonl"

python -m scripts.litert.export.release_export "$run/v5" "$run/bundle" --compact-prefill-mask > "$run/export.log" 2>&1
bundle="$run/bundle/model.litertlm"

python -m scripts.litert.checks.inspect_decoder "$bundle" > "$run/decoder.json"
python -m scripts.litert.checks.verify_bundle "$bundle" --cache-dir "$run/runtime-cache" > "$run/smoke.jsonl"
python -m scripts.litert.checks.verify_bundle "$bundle" --case floor --cache-dir :memory > "$run/floor.jsonl"
python -m scripts.litert.checks.compare_quality "$run/v5" "$bundle" --cache-dir "$run/runtime-cache" \
    --max-nll-increase "$max_nll_increase" > "$run/quality.jsonl"
sha256sum "$bundle" > "$run/bundle-sha256.txt"
