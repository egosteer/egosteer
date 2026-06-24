#!/bin/bash
# Offline dataset evaluation for EgoSteer flow matching.
#
# Loads a trained checkpoint and its corresponding training config,
# samples from WebDataset shards, runs flow inference, and saves
# action metrics + visualizations.
#
# Usage:
#   bash scripts/run_eval.sh <checkpoint_path> <train_config_path> [extra hydra overrides...]
#
# Examples:
#   # Basic evaluation (pass the normalizer used for training/finetuning)
#   bash scripts/run_eval.sh /path/to/ckpt.pt /path/to/config.yaml normalizer_path=/path/to/normalizer.pkl
#
#   # RTC test: pin first 4 GT action steps as condition
#   bash scripts/run_eval.sh /path/to/ckpt.pt /path/to/config.yaml \
#       normalizer_path=/path/to/normalizer.pkl inference_delay=4
#
#   # Custom output, fewer shards, smaller batch
#   bash scripts/run_eval.sh /path/to/ckpt.pt /path/to/config.yaml \
#       normalizer_path=/path/to/normalizer.pkl num_shards=5 batch_size=8 output_dir=outputs/my_eval
#
# Key config fields (see src/config/eval_config.yaml for full list):
#   normalizer_path  Normalizer used for training/finetuning (required)
#   num_shards       Number of shards to randomly sample (default: 10)
#   num_samples      Max samples for metric computation (default: 200)
#   batch_size       Inference batch size (default: 16)
#   num_workers      DataLoader workers (default: 32)
#   flow_steps       Override flow inference steps (default: from training config)
#   inference_delay  RTC test: pin first N GT action steps (default: 0 = off)

set -euo pipefail

if [ $# -lt 2 ]; then
    echo "Usage: bash scripts/run_eval.sh <checkpoint_path> <train_config_path> [overrides...]"
    exit 1
fi

CKPT="$1"; shift
TRAIN_CFG="$1"; shift

python evaluate.py \
    checkpoint_path="$CKPT" \
    train_config_path="$TRAIN_CFG" \
    "$@"
