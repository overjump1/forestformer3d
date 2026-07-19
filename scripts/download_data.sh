#!/usr/bin/env bash
# Download the ForAINetV2 dataset (and pretrained model) from Zenodo.
# ONLINE PHASE — run this on a machine with internet, then move ./data and
# ./work_dirs to the offline machine together with the docker image.
#
# Files land in ./zenodo_downloads (kept for re-runs; md5-verified), archives
# are extracted, and the expected layout is assembled:
#   data/ForAINetV2/{train_val_data,test_data}/
#   work_dirs/clean_forestformer/epoch_3000_fix.pth
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."

RECORD="${ZENODO_RECORD:-16742708}"
DL=zenodo_downloads

echo "== Files in Zenodo record $RECORD:"
python3 scripts/zenodo_fetch.py --record "$RECORD" --list
echo ""
python3 scripts/zenodo_fetch.py --record "$RECORD" --out "$DL" --extract

mkdir -p data/ForAINetV2 work_dirs/clean_forestformer

# Place extracted dataset folders where the pipeline expects them.
for name in train_val_data test_data meta_data; do
    if [ ! -d "data/ForAINetV2/$name" ]; then
        src="$(find "$DL/extracted" -type d -name "$name" | head -n1 || true)"
        if [ -n "$src" ]; then
            echo "Installing $name/ -> data/ForAINetV2/$name"
            cp -r "$src" "data/ForAINetV2/$name"
        fi
    fi
done

# Place the pretrained checkpoint.
if [ ! -f work_dirs/clean_forestformer/epoch_3000_fix.pth ]; then
    ckpt="$(find "$DL" -type f -name 'epoch_3000_fix.pth' | head -n1 || true)"
    [ -z "$ckpt" ] && ckpt="$(find "$DL" -type f -name '*.pth' | head -n1 || true)"
    if [ -n "$ckpt" ]; then
        echo "Installing checkpoint -> work_dirs/clean_forestformer/epoch_3000_fix.pth"
        cp "$ckpt" work_dirs/clean_forestformer/epoch_3000_fix.pth
    fi
fi

echo ""
echo "== Resulting layout:"
find data/ForAINetV2 -maxdepth 1 -mindepth 1 | sort || true
ls -lh work_dirs/clean_forestformer/ 2>/dev/null || true

if [ ! -d data/ForAINetV2/train_val_data ] || [ ! -d data/ForAINetV2/test_data ]; then
    echo ""
    echo "NOTE: could not auto-detect train_val_data/ and/or test_data/ in the"
    echo "downloaded archives. Inspect $DL/extracted/ and arrange the files as:"
    echo "  data/ForAINetV2/train_val_data/*.ply"
    echo "  data/ForAINetV2/test_data/*.ply"
fi
