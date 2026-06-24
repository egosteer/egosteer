#!/usr/bin/env bash
set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: $0 <container_name> <checkpoint_root>"
  echo "Example: $0 egosteer ./egosteer-checkpoints"
  exit 1
fi

CONTAINER_NAME="$1"
HOST_CHECKPOINT_ROOT="$2"
IMAGE="egosteerai/inference-server:latest"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONTAINER_APP_ROOT="/root/workspace/egosteer"
CONTAINER_CHECKPOINT_ROOT="/root/workspace/checkpoints"

if docker ps -a --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' already exists. Remove it first:"
  echo "  docker rm -f ${CONTAINER_NAME}"
  exit 1
fi

if [ ! -d "${HOST_CHECKPOINT_ROOT}" ]; then
  echo "Checkpoint root does not exist: ${HOST_CHECKPOINT_ROOT}"
  echo "Pass an existing EgoSteer model bundle directory as the second argument."
  exit 1
fi

CHECKPOINT_ROOT="$(cd "${HOST_CHECKPOINT_ROOT}" && pwd)"

GPU_OPTIONS=()
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_OPTIONS+=(--gpus all)
fi

echo "Starting ${CONTAINER_NAME} from ${IMAGE}"
echo "Mounting repo: ${SCRIPT_DIR} -> ${CONTAINER_APP_ROOT}"
echo "Mounting checkpoints: ${CHECKPOINT_ROOT} -> ${CONTAINER_CHECKPOINT_ROOT}"
echo "Serving config: ${CONTAINER_APP_ROOT}/src/config/experiment/inference.yaml"
echo "Startup command: long-running environment shell"

docker run -d \
  "${GPU_OPTIONS[@]}" \
  --name "${CONTAINER_NAME}" \
  --network=host \
  --ipc=host \
  --shm-size=32g \
  -w "${CONTAINER_APP_ROOT}" \
  -v "${SCRIPT_DIR}":"${CONTAINER_APP_ROOT}" \
  -v "${CHECKPOINT_ROOT}":"${CONTAINER_CHECKPOINT_ROOT}" \
  "${IMAGE}" \
  bash -lc 'exec sleep infinity'

sleep 2
if docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "Container '${CONTAINER_NAME}' is running."
  echo "Shell: docker exec -it ${CONTAINER_NAME} bash"
  echo "Start server: bash scripts/run_server.sh"
else
  echo "Failed to start container '${CONTAINER_NAME}'."
  docker logs "${CONTAINER_NAME}" || true
  exit 1
fi

if [ -t 0 ] && [ -t 1 ]; then
  docker exec -it "${CONTAINER_NAME}" bash
fi
