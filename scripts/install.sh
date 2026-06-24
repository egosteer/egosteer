#!/bin/bash

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y build-essential cmake git wget curl unzip software-properties-common apt-transport-https ca-certificates gnupg lsb-release
sudo apt-get install -y libjpeg-dev libpng-dev libtiff-dev libavcodec-dev libavformat-dev libswscale-dev libv4l-dev libxvidcore-dev libx264-dev libgtk-3-dev libatlas-base-dev gfortran
sudo apt-get install -y python3-dev python3-pip python3-venv libhdf5-dev pkg-config

# Add NVIDIA repository and install NCCL
echo "Adding NVIDIA repository and installing NCCL..."
# make sure the right version of the system is used
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get install -y libnccl2 libnccl-dev
rm -f cuda-keyring_1.0-1_all.deb

# Install pdsh for the multi-node launcher; numactl for per-rank NUMA binding
# (see scripts/numa_bind_wrapper.sh).
sudo apt-get install -y pdsh numactl

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -y -n egosteer python=3.10
conda activate egosteer

# Install EgoSteer dependencies inside the repository root.
cd "${ROOT_DIR}"
pip install torch==2.10.0+cu128 torchvision==0.25.0+cu128 torchaudio==2.10.0+cu128 \
    --extra-index-url https://download.pytorch.org/whl/cu128

# Install FlashAttention after PyTorch so it can build against the active torch/CUDA toolchain.
# Keep it out of requirements.txt because this dependency usually needs an environment-specific build step.
pip install packaging ninja psutil
pip install flash-attn --no-build-isolation

pip install -r requirements.txt
