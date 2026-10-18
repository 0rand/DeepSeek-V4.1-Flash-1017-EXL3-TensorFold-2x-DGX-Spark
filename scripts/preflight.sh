#!/usr/bin/env bash
# Checks before a start, on both nodes. It refuses (exit 1) on the faults that otherwise show as a hang or a crash
# minutes into the start: missing files, different images, a bad RoCE GID, no memory, no disk, a busy port.
#   scripts/preflight.sh          (start.sh runs it first; PREFLIGHT=0 ./start.sh skips it)
set -uo pipefail
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"

bad=0
wrun() { ssh -o BatchMode=yes -o ConnectTimeout=10 "$WORKER_SSH" "$@"; }   # (stdin open: wssh closes it)
ok() { echo "[ ok ] $*"; }
no() { echo "[FAIL] $*"; bad=1; }

wssh true 2>/dev/null && ok "ssh to the worker ($WORKER_SSH)" || { no "ssh to the worker ($WORKER_SSH) failed"; exit 1; }
ping -c 2 -W 2 "$WORKER_IP" >/dev/null 2>&1 && ok "the worker answers on the CX7 link ($WORKER_IP)" \
  || no "no answer from $WORKER_IP: check the cable and docs/NETWORK.md"

check_node() {  # $1 = label, $2 = command runner ("" for here, wssh for the worker)
  local who=$1 run=$2 out
  out=$($run bash -s <<EOF
set -u
for d in "$MODEL_VIEW/model" "$MODEL_VIEW/engram"; do [ -d "\$d" ] || echo "FAIL missing \$d"; done
[ -f "$MODEL_VIEW/model/chat_template.jinja" ] || echo "FAIL missing $MODEL_VIEW/model/chat_template.jinja"
n=\$(ls "$MODEL_VIEW"/model/*.safetensors 2>/dev/null | wc -l); [ "\$n" -ge 43 ] || echo "FAIL \$n of 43 weight shards"
docker image inspect "$IMAGE" --format '{{.Id}}' 2>/dev/null | sed 's/^/IMAGE /' || echo "FAIL no image $IMAGE"
for dev in \$(echo "$RDMA_DEVICES" | tr ',' ' '); do
  if [ "$GID_INDEX" = "-1" ]; then echo "OK gid auto (\$dev)"; continue; fi
  g=\$(cat /sys/class/infiniband/\$dev/ports/1/gids/$GID_INDEX 2>/dev/null)
  t=\$(cat /sys/class/infiniband/\$dev/ports/1/gid_attrs/types/$GID_INDEX 2>/dev/null)
  case "\$g" in ""|0000:0000:0000:0000:0000:0000:0000:0000) echo "FAIL \$dev GID $GID_INDEX is empty: see docs/NETWORK.md";;
    *) [ "\$t" = "RoCE v2" ] && echo "OK \$dev GID $GID_INDEX \$g (\$t)" || echo "FAIL \$dev GID $GID_INDEX is \$t, not RoCE v2";; esac
done
avail=\$(awk '/MemAvailable/ {print int(\$2/1048576)}' /proc/meminfo); [ "\$avail" -ge 100 ] \
  && echo "OK \$avail GiB memory available" || echo "FAIL only \$avail GiB memory available (need ~100): stop other GPU programs"
mkdir -p "$CACHE_DIR"; free=\$(df -BG --output=avail "$CACHE_DIR" | tail -1 | tr -dc 0-9)
[ -d "$CACHE_DIR/tf-ds-rank" ] && need=5 || need=110
[ "\$free" -ge "\$need" ] && echo "OK \$free GiB free for the cache" || echo "FAIL \$free GiB free in $CACHE_DIR (need \$need)"
docker ps --format '{{.Names}}' | grep -q '^tfv41-r' && echo "FAIL a tfv41 rank already runs: ./stop.sh first" || true
EOF
)
  while IFS= read -r line; do
    case "$line" in
      FAIL*) no "$who: ${line#FAIL }" ;;
      OK*) ok "$who: ${line#OK }" ;;
      IMAGE*) echo "${line#IMAGE }" > "/tmp/tfv41-image-$who" ;;
    esac
  done <<< "$out"
}
check_node head ""
check_node worker wrun
if [ -s /tmp/tfv41-image-head ] && cmp -s /tmp/tfv41-image-head /tmp/tfv41-image-worker; then
  ok "both nodes hold $IMAGE ($(cut -c8-19 /tmp/tfv41-image-head))"
else
  no "the nodes hold different images (or none) for $IMAGE: run docker/build.sh"
fi
rm -f /tmp/tfv41-image-head /tmp/tfv41-image-worker
ss -ltn | grep -q ":$PORT " && no "port $PORT is in use on the head" || ok "port $PORT is free"
[ "$bad" = 0 ] && echo "preflight: all checks passed" || { echo "preflight: fix the FAIL lines, then start again"; exit 1; }
