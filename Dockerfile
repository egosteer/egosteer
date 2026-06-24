# EgoSteer environment image. It intentionally does not include repository code.
# Build:
#   docker build -t egosteerai/inference-server:1.0.0 -t egosteerai/inference-server:latest .
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG MAX_JOBS=4

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/root/workspace/egosteer \
    HF_HOME=/root/workspace/checkpoints/hf_cache \
    HUGGINGFACE_HUB_CACHE=/root/workspace/checkpoints/hf_cache/hub \
    TORCHINDUCTOR_FX_GRAPH_CACHE=1 \
    TORCHINDUCTOR_AUTOGRAD_CACHE=1 \
    TORCHINDUCTOR_CACHE_DIR=/root/.cache/torchinductor \
    TRITON_CACHE_DIR=/root/.cache/triton \
    TORCHINDUCTOR_FORCE_CUDA_CODE_CACHE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.10 python3.10-dev python3-pip python3-setuptools python3-wheel \
      build-essential cmake git wget curl ca-certificates pkg-config \
      ffmpeg libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
      libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev \
      libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev \
      libatlas-base-dev gfortran libhdf5-dev ninja-build \
    && ln -sf /usr/bin/python3.10 /usr/local/bin/python \
    && ln -sf /usr/bin/pip3 /usr/local/bin/pip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /root/workspace/egosteer

RUN python -m pip install --upgrade pip wheel setuptools \
    && python -m pip install \
      torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
      --extra-index-url https://download.pytorch.org/whl/cu128

RUN python -m pip install packaging ninja psutil \
    && MAX_JOBS="$MAX_JOBS" python -m pip install flash-attn --no-build-isolation

RUN python -m pip install \
      hydra-core \
      omegaconf \
      wandb \
      tqdm \
      numpy \
      zarr \
      transformers==5.6.2 \
      einops \
      imageio \
      Pillow \
      opencv-python \
      albumentations \
      PyRender \
      trimesh \
      "manotorch @ git+https://github.com/lixiny/manotorch.git" \
      scipy \
      matplotlib \
      seaborn \
      xformers==0.0.35 \
      websockets \
      msgpack \
      scikit-learn \
      webdataset \
      dill \
      psutil \
      --extra-index-url https://download.pytorch.org/whl/cu128

RUN mkdir -p /root/workspace/checkpoints /root/workspace/egosteer/outputs

EXPOSE 8765

CMD ["bash"]
