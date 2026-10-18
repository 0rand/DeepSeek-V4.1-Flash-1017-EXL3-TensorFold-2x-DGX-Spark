# Shared by start.sh, stop.sh, status.sh and verify.sh: reads .env and defines the helpers. Source it; do not run it.
# shellcheck shell=bash

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ ! -f "$ROOT/.env" ]; then
  echo "no $ROOT/.env: copy .env.example to .env and edit it" >&2
  exit 1
fi
set -a
# shellcheck disable=SC1091
. "$ROOT/.env"
set +a

API="http://${BIND_HOST/0.0.0.0/127.0.0.1}:${PORT}/v1"
STORE_PORT=29551                 # TensorFold's rank-0 rendezvous port (--master-port default)
LOGS="$ROOT/logs"

log() { echo "[tfv41 $(date -u +%H:%M:%SZ)] $*"; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }
auth() { [ -n "${API_KEY_FILE:-}" ] && printf 'Authorization: Bearer %s' "$(grep -v '^#' "$API_KEY_FILE" | head -1 | sed 's/^.*: *//')"; }
healthy() { curl -fsS -m 10 -H "$(auth)" "$API/models" 2>/dev/null | grep -q "\"$SERVED_NAME\""; }
