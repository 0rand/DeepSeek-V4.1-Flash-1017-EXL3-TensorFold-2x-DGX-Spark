#!/usr/bin/env bash
# Build the engine image on the head at the pinned TensorFold commit, copy it to the worker, and check that both nodes
# hold the same image ID. ~10 min (the NGC base pull is ~20 GB the first time).
set -euo pipefail
# shellcheck source=../scripts/common.sh
. "$(dirname "$0")/../scripts/common.sh"

log "building $IMAGE from $TF_REPO @ $TF_COMMIT"
docker build --build-arg TF_REPO="$TF_REPO" --build-arg TF_COMMIT="$TF_COMMIT" -t "$IMAGE" "$ROOT/docker"
log "copying $IMAGE to the worker ($WORKER_SSH)"
docker save "$IMAGE" | ssh -o BatchMode=yes "$WORKER_SSH" docker load
head_id=$(docker images "$IMAGE" --format '{{.ID}}')
worker_id=$(wssh "docker images '$IMAGE' --format '{{.ID}}'")
[ "$head_id" = "$worker_id" ] || { log "image IDs differ: head $head_id, worker $worker_id"; exit 1; }
log "both nodes hold $IMAGE ($head_id)"
