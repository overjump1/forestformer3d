#!/usr/bin/env bash
# Build the ForestFormer3D image. ONLINE PHASE — requires internet access.
#
# If nvidia-smi is available on this machine, the CUDA extensions are compiled
# ONLY for the GPU(s) detected here (much less disk/time than the broad
# default). Override with the CUDA_ARCH_LIST env var, e.g.:
#   CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX" scripts/build.sh   # broad/portable
#   CUDA_ARCH_LIST="8.6" scripts/build.sh                   # RTX 30xx / A5000
#
# Extra args are passed straight to `docker build`, e.g.:
#   scripts/build.sh --build-arg SKIP_CHECKPOINT=1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${IMAGE:-forestformer3d:offline}"

# Auto-detect the local GPU architecture(s) unless explicitly overridden.
if [ -z "${CUDA_ARCH_LIST:-}" ] && command -v nvidia-smi >/dev/null 2>&1; then
    caps="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null \
            | tr -d ' ' | sort -u | paste -sd';' - || true)"
    if [ -n "$caps" ]; then
        CUDA_ARCH_LIST="${caps}+PTX"
        echo "Detected local GPU compute capability: $caps -> building for $CUDA_ARCH_LIST"
        echo "(set CUDA_ARCH_LIST to override, e.g. for a different target machine)"
    fi
fi

BUILD_ARGS=()
if [ -n "${CUDA_ARCH_LIST:-}" ]; then
    BUILD_ARGS+=(--build-arg "CUDA_ARCH_LIST=$CUDA_ARCH_LIST")
fi

docker build -t "$IMAGE" "${BUILD_ARGS[@]}" "$@" .
echo ""
echo "Built $IMAGE."
echo "To move it to an air-gapped machine:"
echo "  docker save $IMAGE | gzip > forestformer3d-offline.tar.gz"
echo "  # then on the target machine:"
echo "  docker load < forestformer3d-offline.tar.gz"
