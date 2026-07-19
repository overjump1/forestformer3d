#!/usr/bin/env bash
# Container entrypoint. Host volumes are typically mounted over
# /workspace/data/ForAINetV2 (dataset) and /workspace/work_dirs (outputs),
# which hides the copies baked into the image. Restore anything that the
# mounted directories are missing, then exec the requested command.
set -euo pipefail

PRISTINE=/opt/forestformer3d

# Pretrained checkpoint: restore into (possibly mounted) work_dirs.
CKPT=/workspace/work_dirs/clean_forestformer/epoch_3000_fix.pth
if [ ! -f "$CKPT" ] && [ -f "$PRISTINE/checkpoints/epoch_3000_fix.pth" ]; then
    mkdir -p "$(dirname "$CKPT")"
    cp "$PRISTINE/checkpoints/epoch_3000_fix.pth" "$CKPT"
    echo "[entrypoint] restored pretrained checkpoint -> $CKPT"
fi

# Data-prep scripts and meta_data: copy into the mounted dataset dir without
# overwriting anything the user already has there (cp -n = no clobber).
if [ -d /workspace/data/ForAINetV2 ] && [ -d "$PRISTINE/pristine_data/ForAINetV2" ]; then
    cp -rn "$PRISTINE/pristine_data/ForAINetV2/." /workspace/data/ForAINetV2/ || true
fi

exec "$@"
