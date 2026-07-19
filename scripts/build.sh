#!/usr/bin/env bash
# Build the ForestFormer3D image. ONLINE PHASE — requires internet access.
# The image targets an RTX A5000 (compute capability 8.6) by default; for a
# different GPU pass e.g.:
#   scripts/build.sh --build-arg CUDA_ARCH_LIST="8.0"
# Extra args are passed straight to `docker build`, e.g.:
#   scripts/build.sh --build-arg SKIP_CHECKPOINT=1
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

IMAGE="${IMAGE:-forestformer3d:offline}"
docker build -t "$IMAGE" "$@" .
echo ""
echo "Built $IMAGE."
echo "To move it to an air-gapped machine:"
echo "  docker save $IMAGE | gzip > forestformer3d-offline.tar.gz"
echo "  # then on the target machine:"
echo "  docker load < forestformer3d-offline.tar.gz"
