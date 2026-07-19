# ForestFormer3D — Docker image for offline use

Everything needed to **build a Docker image** of
[SmartForest-no/ForestFormer3D](https://github.com/SmartForest-no/ForestFormer3D)
(ICCV 2025 — end-to-end segmentation of forest LiDAR 3D point clouds) and then
**run it fully offline** — inference, training, and data preprocessing all work
with `--network none`.

The upstream source code is vendored in this repository at commit
[`6a75c37`](https://github.com/SmartForest-no/ForestFormer3D/commit/6a75c3735e4a4108d02ee944a8b93177f2360a4f)
(see [Attribution & license](#attribution--license)). The upstream docs are
kept as [`UPSTREAM_README.md`](UPSTREAM_README.md) and the original Dockerfile
as `Dockerfile.upstream`.

Compared to the upstream image, this one is **offline-complete**: all the
manual post-install steps from the upstream readme are baked into the build —
`laspy[lazrs]`, the CUDA-enabled `torch-points-kernels`/`torch-cluster`
builds, the patched `mmengine`/`mmdet3d` files from
`replace_mmdetection_files/`, the compiled `segmentator` extension, and the
pretrained checkpoint from Zenodo.

## Requirements

- **Online machine (build phase):** Docker, internet access, ~40 GB free disk.
  No GPU needed to build.
- **Offline machine (run phase):** NVIDIA GPU + driver, Docker with
  [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html).
  Upstream recommends an A100 for training; inference works on smaller GPUs
  (reduce `chunk`/cylinder radius in the config if you hit OOM — see the
  upstream readme).

The image compiles its CUDA extensions for
`TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"` — V100, RTX 20xx, A100/A40,
RTX 30xx natively, and newer GPUs (e.g. RTX 40xx) via PTX JIT. To target a
single architecture (faster build, smaller image):
`scripts/build.sh --build-arg CUDA_ARCH_LIST="8.0"`.

## Phase 1 — online machine

```bash
# 1. Build the image (compiles extensions, bakes in the pretrained model)
scripts/build.sh

# 2. Download the ForAINetV2 dataset from Zenodo record 16742708
#    -> data/ForAINetV2/{train_val_data,test_data}, work_dirs/.../epoch_3000_fix.pth
scripts/download_data.sh
```

Moving to an air-gapped machine:

```bash
docker save forestformer3d:offline | gzip > forestformer3d-offline.tar.gz
# copy forestformer3d-offline.tar.gz + the data/ and work_dirs/ folders over, then:
docker load < forestformer3d-offline.tar.gz
```

Build knobs (all optional): `--build-arg SKIP_CHECKPOINT=1` builds without the
pretrained model; `--build-arg CHECKPOINT_URL=<url>` pins the exact model file
(direct `.pth` or `.zip` URL) if the Zenodo auto-discovery picks the wrong
asset — inspect the record with
`python3 scripts/zenodo_fetch.py --record 16742708 --list`;
`--build-arg MAX_JOBS=8` speeds up compilation on big machines.

## Phase 2 — offline machine

`scripts/run_offline.sh` runs the container with `--gpus all` and
`--network none` (no network available, by construction), mounting
`./data/ForAINetV2` and `./work_dirs` from the host. With no arguments it
opens a shell; otherwise it runs the given command.

```bash
# a) Preprocess the point clouds (once, after placing/downloading the data)
scripts/run_offline.sh bash -c \
  "cd data/ForAINetV2 && python batch_load_ForAINetV2_data.py && cd /workspace && python tools/create_data_forainetv2.py forainetv2"

# b) Inference with the baked-in pretrained model
scripts/run_offline.sh python tools/test.py \
  configs/oneformer3d_qs_radius16_qp300_2many.py \
  work_dirs/clean_forestformer/epoch_3000_fix.pth

# c) Training
scripts/run_offline.sh python tools/train.py \
  configs/oneformer3d_qs_radius16_qp300_2many.py --work-dir work_dirs/my_run
```

`docker compose run --rm forestformer3d ...` is an equivalent alternative
(see `docker-compose.yml`). For training, raise the shared memory:
`SHM_SIZE=128g scripts/run_offline.sh ...`.

Notes:
- The container entrypoint restores the baked-in checkpoint and the data-prep
  scripts/meta_data into the mounted volumes if they're missing, so an empty
  `work_dirs/` mount still has the pretrained model available.
- To test your own point clouds: put `.ply` files in
  `data/ForAINetV2/test_data/`, add their base names to
  `data/ForAINetV2/meta_data/test_list.txt`, re-run step (a), then step (b).
  Details (including `.las`/`.laz` input and dense-plot two-pass inference via
  `tools/inference_bluepoint.sh`) are in [`UPSTREAM_README.md`](UPSTREAM_README.md).
- Outputs land in `./work_dirs/` on the host.

## What's in the image

| Component | Version / source |
|---|---|
| Base | `pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel` (Python 3.10) |
| OpenMMLab | mmengine 0.7.3, mmcv 2.0.0 (cu116), mmdet 3.0.0, mmsegmentation 1.0.0, mmdetection3d @`22aaa47` (+ project patches from `replace_mmdetection_files/`) |
| Sparse/point ops | spconv-cu116 2.3.6, MinkowskiEngine @`02fc608`, torch-scatter 2.0.9, torch-points-kernels 0.7.0, torch-cluster 1.6.1 (all compiled with CUDA) |
| Superpoints | `segmentator/` (bundled, built via CMake; CMakeLists vendored from Karbo123/segmentator @`76efe46`) |
| Model | `epoch_3000_fix.pth` from [Zenodo 16742708](https://zenodo.org/records/16742708), baked at `/opt/forestformer3d/checkpoints/` and restored to `work_dirs/clean_forestformer/` on start |

## Attribution & license

This repository vendors the official ForestFormer3D implementation by the
SmartForest project (Binbin Xiang et al., NIBIO), which builds on
[OneFormer3D](https://github.com/filaPro/oneformer3d) and is licensed under
**CC BY-NC 4.0** (see [LICENSE](LICENSE)); this repository is under the same
license. If you use this work, please cite:

```bibtex
@inproceedings{xiang2025forestformer3d,
  title     = {ForestFormer3D: A Unified Framework for End-to-End Segmentation of Forest LiDAR 3D Point Clouds},
  author    = {Binbin Xiang and Maciej Wielgosz and Stefano Puliti and Kamil Král and Martin Krůček and Azim Missarov and Rasmus Astrup},
  booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
  year      = {2025}
}
```
