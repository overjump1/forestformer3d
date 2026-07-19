#!/usr/bin/env bash
# Run the ForestFormer3D container fully OFFLINE (--network none proves no
# network is needed). GPU access via nvidia-container-toolkit.
#
# Usage:
#   scripts/run_offline.sh                 # interactive shell in the container
#   scripts/run_offline.sh <command...>    # run a single command, e.g.:
#   scripts/run_offline.sh python tools/test.py \
#       configs/oneformer3d_qs_radius16_qp300_2many.py \
#       work_dirs/clean_forestformer/epoch_3000_fix.pth
#
# Env overrides: IMAGE (default forestformer3d:offline), SHM_SIZE (64g),
#                GPUS (all), NETWORK (none).
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${IMAGE:-forestformer3d:offline}"
SHM_SIZE="${SHM_SIZE:-64g}"
GPUS="${GPUS:-all}"
NETWORK="${NETWORK:-none}"

mkdir -p data/ForAINetV2 work_dirs

TTY_FLAGS=""
if [ -t 0 ] && [ -t 1 ]; then TTY_FLAGS="-it"; fi

exec docker run --rm $TTY_FLAGS \
    --gpus "$GPUS" \
    --shm-size="$SHM_SIZE" \
    --network "$NETWORK" \
    -v "$PWD/data/ForAINetV2:/workspace/data/ForAINetV2" \
    -v "$PWD/work_dirs:/workspace/work_dirs" \
    "$IMAGE" "$@"
