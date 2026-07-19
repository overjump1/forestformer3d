# ForestFormer3D — offline-runnable image.
#
# Build ONCE on a machine WITH internet access (compiles the CUDA extensions
# and bakes in the pretrained checkpoint); afterwards the container runs with
# no network at all (see scripts/run_offline.sh, which uses --network none).
#
# Build args:
#   CUDA_ARCH_LIST   GPU architectures to compile for (default covers
#                    V100/RTX20xx/A100/RTX30xx; newer GPUs run via +PTX JIT)
#   MAX_JOBS         parallel compile jobs for the CUDA extensions
#   SKIP_CHECKPOINT  set to 1 to build without downloading the Zenodo model
#   CHECKPOINT_URL   direct URL to the model (.pth or .zip) instead of
#                    auto-discovery via the Zenodo API
#   CHECKPOINT_MATCH regex used to pick the model asset from the Zenodo record
FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# 8.6 = RTX A5000 / RTX 30xx (Ampere). For other/multiple GPUs pass e.g.
# --build-arg CUDA_ARCH_LIST="7.0;7.5;8.0;8.6+PTX"
ARG CUDA_ARCH_LIST="8.6"
ARG MAX_JOBS=4
ARG DEBIAN_FRONTEND=noninteractive

ENV PATH=/usr/local/cuda/bin:$PATH \
    LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH \
    TORCH_CUDA_ARCH_LIST=$CUDA_ARCH_LIST \
    FORCE_CUDA=1 \
    MAX_JOBS=$MAX_JOBS

# The NVIDIA apt repos shipped in the base image have rotated/expired signing
# keys; we don't need any apt packages from them (the -devel image already
# contains the full CUDA toolkit), so drop them instead of re-keying.
RUN rm -f /etc/apt/sources.list.d/cuda.list /etc/apt/sources.list.d/nvidia-ml.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates \
        git unzip ninja-build \
        libgl1 libgomp1 libglib2.0-0 libsm6 libxext6 libxrender1 \
        libopenblas-dev \
    && (apt-get install -y --no-install-recommends gcc-9 g++-9 \
        || (apt-get install -y --no-install-recommends software-properties-common \
            && add-apt-repository -y ppa:ubuntu-toolchain-r/test \
            && apt-get update \
            && apt-get install -y --no-install-recommends gcc-9 g++-9)) \
    && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 \
    && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 60 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# segmentator's CMakeLists requires cmake >= 3.18, newer than the distro's.
RUN pip install --no-cache-dir "cmake>=3.18,<3.28" debugpy

# OpenMMLab stack (pinned; --no-deps keeps pip from touching the base torch)
RUN pip install --no-cache-dir --no-deps \
        mmengine==0.7.3 \
        mmdet==3.0.0 \
        mmsegmentation==1.0.0 \
        git+https://github.com/open-mmlab/mmdetection3d.git@22aaa47fdb53ce1870ff92cb7e3f96ae38d17f61 \
    && pip install --no-cache-dir --no-deps mmcv==2.0.0 \
        -f https://download.openmmlab.com/mmcv/dist/cu116/torch1.13.0/index.html

# MinkowskiEngine (compiled; needs openblas + nvcc; no GPU required at build)
RUN git clone https://github.com/NVIDIA/MinkowskiEngine.git /tmp/MinkowskiEngine \
    && cd /tmp/MinkowskiEngine \
    && git checkout 02fc608bea4c0549b0a7b00ca1bf15dee4a0b228 \
    && python setup.py install --blas=openblas --force_cuda \
    && rm -rf /tmp/MinkowskiEngine

# torch-scatter v2.0.9 (compiled)
RUN git clone --branch 2.0.9 --depth 1 https://github.com/rusty1s/pytorch_scatter.git /tmp/pytorch_scatter \
    && cd /tmp/pytorch_scatter \
    && pip install --no-cache-dir --no-deps --no-build-isolation . \
    && rm -rf /tmp/pytorch_scatter

# Remaining pinned Python packages (same set as the upstream image)
RUN pip install --no-cache-dir --no-deps \
        spconv-cu116==2.3.6 \
        cumm-cu116==0.4.9 \
        addict==2.4.0 \
        yapf==0.33.0 \
        termcolor==2.3.0 \
        packaging==23.1 \
        numpy==1.24.1 \
        rich==13.3.5 \
        opencv-python==4.7.0.72 \
        pycocotools==2.0.6 \
        Shapely==1.8.5 \
        scipy==1.10.1 \
        terminaltables==3.1.10 \
        numba==0.57.0 \
        llvmlite==0.40.0 \
        pccm==0.4.7 \
        ccimport==0.4.2 \
        pybind11==2.10.4 \
        ninja==1.11.1 \
        lark==1.1.5 \
        pyquaternion==0.9.9 \
        lyft-dataset-sdk==0.0.8 \
        pandas==2.0.1 \
        python-dateutil==2.8.2 \
        matplotlib==3.5.2 \
        pyparsing==3.0.9 \
        cycler==0.11.0 \
        kiwisolver==1.4.4 \
        scikit-learn==1.2.2 \
        joblib==1.2.0 \
        threadpoolctl==3.1.0 \
        cachetools==5.3.0 \
        nuscenes-devkit==1.1.10 \
        trimesh==3.21.6 \
        open3d==0.17.0 \
        plotly==5.18.0 \
        dash==2.14.2 \
        plyfile==1.0.2 \
        flask==3.0.0 \
        werkzeug==3.0.1 \
        click==8.1.7 \
        blinker==1.7.0 \
        itsdangerous==2.1.2 \
        importlib_metadata==2.1.2 \
        zipp==3.17.0 \
        tensorboard==2.15.1 \
        tensorboard-data-server==0.7.2 \
        protobuf \
        absl-py \
        future \
        MarkupSafe==2.0.1 \
        markdown \
        grpcio \
        google-auth-oauthlib \
        google-auth \
        requests-oauthlib \
        oauthlib

# laspy (+ lazrs backend) for the .las/.laz data-prep scripts — the upstream
# readme installs these manually at runtime; baked in here for offline use.
RUN pip install --no-cache-dir --no-deps laspy==2.5.3 lazrs

# torch-points-kernels / torch-cluster, compiled with CUDA support so the
# readme's post-install "reinstall to fix points_cuda" step is unnecessary.
RUN pip install --no-cache-dir --no-deps --no-build-isolation torch-points-kernels==0.7.0 \
    && pip install --no-cache-dir --no-deps --no-build-isolation torch-cluster==1.6.1

# ---- Pretrained checkpoint (baked into the image; online phase only) -------
ARG SKIP_CHECKPOINT=0
ARG ZENODO_RECORD=16742708
ARG CHECKPOINT_URL=""
ARG CHECKPOINT_MATCH="(?i)(\.pth$|model|checkpoint|weights)"
COPY scripts/zenodo_fetch.py scripts/install_checkpoint.sh /opt/forestformer3d/scripts/
RUN if [ "$SKIP_CHECKPOINT" = "1" ]; then \
        echo "SKIP_CHECKPOINT=1 -> not baking the pretrained model"; \
    else \
        ZENODO_RECORD=$ZENODO_RECORD CHECKPOINT_URL=$CHECKPOINT_URL \
        CHECKPOINT_MATCH=$CHECKPOINT_MATCH \
        TARGET=/opt/forestformer3d/checkpoints/epoch_3000_fix.pth \
        bash /opt/forestformer3d/scripts/install_checkpoint.sh; \
    fi

# ---- Project source --------------------------------------------------------
WORKDIR /workspace
COPY . /workspace

# Build the bundled superpoint segmentator (CPU, C++). `make install`
# symlinks site-packages/segmentator -> /workspace/segmentator, so the source
# dir must stay in the image.
RUN cd /workspace/segmentator/csrc \
    && mkdir -p build && cd build \
    && cmake .. \
        -DCMAKE_PREFIX_PATH="$(python -c 'import torch; print(torch.utils.cmake_prefix_path)')" \
        -DPYTHON_INCLUDE_DIR="$(python -c 'from distutils.sysconfig import get_python_inc; print(get_python_inc())')" \
        -DPYTHON_LIBRARY="$(python -c 'import distutils.sysconfig as s; print(s.get_config_var("LIBDIR") + "/libpython3.10.so")')" \
        -DCMAKE_INSTALL_PREFIX="$(python -c 'from distutils.sysconfig import get_python_lib; print(get_python_lib())')" \
    && make -j"$MAX_JOBS" \
    && make install

# Apply the project's patched mmengine/mmdet3d files (the upstream readme
# has users copy these manually inside the running container).
RUN cp /workspace/replace_mmdetection_files/loops.py \
        "$(python -c 'import mmengine.runner as m, os; print(os.path.dirname(m.__file__))')/" \
    && cp /workspace/replace_mmdetection_files/base_model.py \
        "$(python -c 'import mmengine.model.base_model as m, os; print(os.path.dirname(m.__file__))')/" \
    && cp /workspace/replace_mmdetection_files/transforms_3d.py \
        "$(python -c 'import mmdet3d.datasets.transforms as m, os; print(os.path.dirname(m.__file__))')/"

# Pristine copy of the data-prep scripts/meta_data: the entrypoint restores
# these into a host-mounted (initially empty) dataset directory.
RUN cp -r /workspace/data /opt/forestformer3d/pristine_data

ENV PYTHONPATH=/workspace

# Fail the (online) build if anything is missing, instead of failing offline.
RUN python -c "import torch, mmengine, mmdet, mmdet3d, mmseg; \
import MinkowskiEngine, spconv, torch_scatter, torch_cluster; \
from torch_points_kernels import instance_iou; \
import segmentator, laspy, lazrs, open3d, trimesh, plyfile; \
import oneformer3d; \
print('sanity imports OK; torch', torch.__version__, 'cuda', torch.version.cuda)"

RUN cp /workspace/scripts/docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh \
    && chmod +x /usr/local/bin/docker-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/docker-entrypoint.sh"]
CMD ["/bin/bash"]
