#!/usr/bin/env bash
#
# Convert a remote FSDP2 DCP checkpoint into a single .pt (CPU-only, on the
# remote) and rsync it back here, together with .hydra/config.yaml if present.
#
# Reuses src/utils/convert_fsdp_checkpoint.py unchanged (scp'd to the remote and
# run there). Passwordless SSH to the host is assumed.
#
# Usage: pull_checkpoint.sh <conda-env> <ssh-host> <remote-ckpt-dir> <local-dir>

set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "Usage: $0 <conda-env> <ssh-host> <remote-ckpt-dir> <local-dir>" >&2
    exit 1
fi

CONDA_ENV="$1"
HOST="$2"
REMOTE_CKPT="${3%/}"
LOCAL_DIR="${4%/}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONVERTER="${SCRIPT_DIR}/../src/utils/convert_fsdp_checkpoint.py"
REMOTE_PT="${REMOTE_CKPT}/model.pt"

mkdir -p "$LOCAL_DIR"

# 1. Convert DCP -> pt on the remote (CPU only), unless it is already done.
if ssh "$HOST" "test -s '$REMOTE_PT'"; then
    echo "Remote model.pt already exists, skipping conversion."
else
    scp -q "$CONVERTER" "$HOST:/tmp/convert_fsdp_checkpoint.py"
    ssh "$HOST" "source \"\$(conda info --base)/etc/profile.d/conda.sh\" && conda activate '$CONDA_ENV' && \
        CUDA_VISIBLE_DEVICES='' python /tmp/convert_fsdp_checkpoint.py --checkpoint '$REMOTE_CKPT'"
fi

# 2. Download the .pt, resumable (--partial --append works on openrsync too).
rsync -a --partial --append --progress "$HOST:$REMOTE_PT" "$LOCAL_DIR/"

# 3. Download .hydra/config.yaml if it exists.
if ssh "$HOST" "test -f '$REMOTE_CKPT/.hydra/config.yaml'"; then
    rsync -a "$HOST:$REMOTE_CKPT/.hydra/config.yaml" "$LOCAL_DIR/"
fi

echo "Done -> $LOCAL_DIR"
