#!/usr/bin/env bash
# export TORCH_LOGS="+dynamo"
export TORCH_LOGS=recompiles,guards,cudagraphs
# Enable graph cache and CUDA code cache for torch.compile
export TORCHINDUCTOR_FX_GRAPH_CACHE=1
export TORCHINDUCTOR_AUTOGRAD_CACHE=1
export TORCHINDUCTOR_CACHE_DIR=$HOME/.cache/torchinductor
export TRITON_CACHE_DIR=$HOME/.cache/triton
export HF_HOME="${HF_HOME:-/root/workspace/checkpoints/hf_cache}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-${HF_HOME}/hub}"
export TORCHINDUCTOR_FORCE_CUDA_CODE_CACHE=1
python -m src.serving.serve_policy 2>&1 | tee output.log
