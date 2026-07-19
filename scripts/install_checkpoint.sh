#!/usr/bin/env bash
# Fetch the pretrained ForestFormer3D checkpoint (epoch_3000_fix.pth) and
# install it at $TARGET. Used during the Docker image build (online phase),
# but can also be run on a host to pre-fetch the checkpoint.
#
# Resolution order:
#   1. $CHECKPOINT_URL      - direct URL to a .pth file or a .zip containing it
#   2. Zenodo record $ZENODO_RECORD - files matching $CHECKPOINT_MATCH are
#      downloaded (zips extracted) and searched for the checkpoint
set -euo pipefail

ZENODO_RECORD="${ZENODO_RECORD:-16742708}"
CHECKPOINT_URL="${CHECKPOINT_URL:-}"
# Match model-ish assets only, to avoid pulling the (huge) dataset archives.
CHECKPOINT_MATCH="${CHECKPOINT_MATCH:-(?i)(\.pth$|model|checkpoint|weights)}"
CHECKPOINT_NAME="${CHECKPOINT_NAME:-epoch_3000_fix.pth}"
TARGET="${TARGET:-/opt/forestformer3d/checkpoints/epoch_3000_fix.pth}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

if [ -n "$CHECKPOINT_URL" ]; then
    echo "Downloading checkpoint from CHECKPOINT_URL=$CHECKPOINT_URL"
    fname="$WORKDIR/$(basename "${CHECKPOINT_URL%%\?*}")"
    python - "$CHECKPOINT_URL" "$fname" <<'PY'
import shutil, sys, urllib.request
req = urllib.request.Request(sys.argv[1], headers={"User-Agent": "forestformer3d-offline-fetch/1.0"})
with urllib.request.urlopen(req, timeout=60) as r, open(sys.argv[2], "wb") as f:
    shutil.copyfileobj(r, f, 1024 * 1024)
PY
    if [[ "$fname" == *.zip ]]; then
        python -c "import sys, zipfile; zipfile.ZipFile(sys.argv[1]).extractall(sys.argv[2])" \
            "$fname" "$WORKDIR/extracted"
    fi
else
    echo "Fetching checkpoint from Zenodo record $ZENODO_RECORD (match: $CHECKPOINT_MATCH)"
    if ! python "$SCRIPT_DIR/zenodo_fetch.py" --record "$ZENODO_RECORD" \
            --match "$CHECKPOINT_MATCH" --out "$WORKDIR" --extract; then
        echo "" >&2
        echo "Could not find a matching model asset in Zenodo record $ZENODO_RECORD." >&2
        echo "Inspect the record with:" >&2
        echo "    python scripts/zenodo_fetch.py --record $ZENODO_RECORD --list" >&2
        echo "then rebuild with --build-arg CHECKPOINT_URL=<direct-file-url>" >&2
        echo "or --build-arg CHECKPOINT_MATCH=<regex>." >&2
        exit 1
    fi
fi

# Locate the checkpoint among the downloaded/extracted files.
found="$(find "$WORKDIR" -type f -name "$CHECKPOINT_NAME" | head -n1)"
if [ -z "$found" ]; then
    found="$(find "$WORKDIR" -type f -name '*.pth' | head -n1)"
    [ -n "$found" ] && echo "WARNING: $CHECKPOINT_NAME not found; using $(basename "$found") instead."
fi
if [ -z "$found" ]; then
    echo "ERROR: no .pth checkpoint found in the downloaded assets." >&2
    echo "Rebuild with --build-arg CHECKPOINT_URL=<direct-file-url> pointing at the model file." >&2
    exit 1
fi

mkdir -p "$(dirname "$TARGET")"
cp "$found" "$TARGET"
echo "Installed checkpoint: $TARGET ($(du -h "$TARGET" | cut -f1))"
