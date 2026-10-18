#!/usr/bin/env bash
# Stop and remove both rank containers. Their logs go to logs/ first; the caches stay on disk.
set -uo pipefail
# shellcheck source=scripts/common.sh
. "$(dirname "$0")/scripts/common.sh"

mkdir -p "$LOGS"
stamp=$(date -u +%Y%m%dT%H%M%SZ)
docker logs tfv41-r0 > "$LOGS/r0-$stamp.log" 2>&1 || true
wssh "docker logs tfv41-r1" > "$LOGS/r1-$stamp.log" 2>&1 || true
docker stop -t 60 tfv41-r0 >/dev/null 2>&1; docker rm tfv41-r0 >/dev/null 2>&1
wssh "docker stop -t 60 tfv41-r1 >/dev/null 2>&1; docker rm tfv41-r1 >/dev/null 2>&1"
log "stopped (logs in $LOGS/*-$stamp.log)"
