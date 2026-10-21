#!/usr/bin/env bash
# Copy an image tag that is already built on the head to the worker, and check that both nodes hold the same image ID.
# It builds nothing. docker/build.sh cannot do this step for a tag of an experiment: it rebuilds the ordinary image first.
#   experiments/tool-output/transfer.sh tensorfold-dsv41:07aa02585f64-tool-output
set -euo pipefail
# shellcheck source=../../scripts/common.sh
. "$(dirname "$0")/../../scripts/common.sh"

tag=${1:-}
if [[ ! "$tag" =~ ^[A-Za-z0-9][A-Za-z0-9._/:@-]*$ ]]; then
  log "usage: experiments/tool-output/transfer.sh <image tag, for example tensorfold-dsv41:07aa02585f64-tool-output>"
  exit 1
fi
docker image inspect "$tag" >/dev/null 2>&1 || { log "no image $tag on the head: build it first"; exit 1; }
log "copying $tag to the worker ($WORKER_SSH)"
docker save "$tag" | ssh -o BatchMode=yes "$WORKER_SSH" docker load
head_id=$(docker images "$tag" --format '{{.ID}}')
worker_id=$(wssh "docker images '$tag' --format '{{.ID}}'")
[ -n "$head_id" ] && [ "$head_id" = "$worker_id" ] || { log "image IDs differ: head $head_id, worker $worker_id"; exit 1; }
log "both nodes hold $tag ($head_id)"
