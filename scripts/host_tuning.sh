#!/usr/bin/env bash
# Tunes host memory for the engine on both nodes. Run it once (it asks for your sudo password on each node).
# Why: the engine holds about 102 GiB of the 128 GiB. Under load the kernel swaps, reclaims and compacts memory, and
# long prompts then read 10-30% slower with 1-4 s stalls (runs/2026-10-18/). When memory runs out, the head can also
# stop (no ssh) until you cut its power. The engine's RUNBOOK.md asks for the swap part.
#   scripts/host_tuning.sh          apply on the head and the worker (you can run it again; the result is the same)
#   scripts/host_tuning.sh --undo   remove the changes on both nodes and put back the sysctl values from before
#                                   the first run (kept in /var/lib/tensorfold-host-tuning/sysctl.before)
# Changes on each node:
#   1. /etc/systemd/system/tensorfold.slice with MemorySwapMax=0. start.sh puts both rank containers in this slice.
#   2. /etc/sysctl.d/99-tensorfold.conf (kernel defaults in brackets):
#        vm.swappiness = 10               [60]   swap other programs less
#        vm.compaction_proactiveness = 0  [20]   no background compaction while a prompt runs
#        vm.extfrag_threshold = 500       [500]  the kernel compacts memory before it reclaims it
#        vm.min_free_kbytes = 1048576     [set at boot]  keep 1 GiB free for the kernel
#        vm.watermark_scale_factor = 50   [10]   free memory in the background before programs must wait
#      Measured with 2 bench runs each (248 prompt chunks): background compaction on gave 46 slow chunks (over
#      1.3 s), the first version of this script (extfrag_threshold 1000, kernel watermarks) gave 5, these values
#      gave 2 (both at 1.31 s) and 128K prompts at 1,986 and 1,981 tokens/s (runs/2026-10-18/INDEX.md).
#   3. earlyoom (if installed) stops the engine, not the node, when memory runs out. It sends SIGTERM to the
#      process named "tensorfold" below 1 GiB available and SIGKILL below 0.5 GiB. It ignores swap, because the
#      engine cannot use swap. The engine stops its own growth at 2 GiB available (TF_DS_MEM_FLOOR_GIB), and a
#      1M-token prompt took the head down to 1.7 GiB, so a healthy engine is not stopped. Your old file stays as
#      /etc/default/earlyoom.pre-tensorfold.
set -euo pipefail
# shellcheck source=common.sh
. "$(dirname "$0")/common.sh"

node_script() {
  cat <<'EOF'
set -euo pipefail
eo=/etc/default/earlyoom
conf=/etc/sysctl.d/99-tensorfold.conf
before=/var/lib/tensorfold-host-tuning/sysctl.before
keys="vm.swappiness vm.compaction_proactiveness vm.extfrag_threshold vm.min_free_kbytes vm.watermark_scale_factor"
if [ "${1:-}" = "--undo" ]; then
  rm -f /etc/systemd/system/tensorfold.slice "$conf"
  systemctl daemon-reload
  if [ -f "$before" ]; then
    sysctl -q -p "$before"
    rm -f "$before"
    note="sysctl values from before the first run are back"
  else
    # No saved values (a run of an older version of this script): kernel defaults, then the other sysctl files.
    sysctl -q -w vm.swappiness=60 vm.compaction_proactiveness=20 vm.extfrag_threshold=500 vm.watermark_scale_factor=10
    sysctl -q --system || true
    note="no saved sysctl values: kernel defaults set, vm.min_free_kbytes goes back at the next boot"
  fi
  if [ -f "$eo.pre-tensorfold" ]; then
    mv "$eo.pre-tensorfold" "$eo"
    systemctl restart earlyoom || true
  fi
  echo "$(hostname): undone ($note)"; exit 0
fi
# Keep the live values before the first run, so --undo can put them back. With $conf present they are ours.
if [ ! -f "$conf" ] && [ ! -f "$before" ]; then
  mkdir -p "${before%/*}"
  for k in $keys; do echo "$k = $(sysctl -n "$k")"; done > "$before"
fi
printf '[Unit]\nDescription=TensorFold rank containers (no swap)\n\n[Slice]\nMemorySwapMax=0\n' \
  > /etc/systemd/system/tensorfold.slice
systemctl daemon-reload
systemctl start tensorfold.slice
printf '%s\n' "vm.swappiness = 10" "vm.compaction_proactiveness = 0" "vm.extfrag_threshold = 500" \
  "vm.min_free_kbytes = 1048576" "vm.watermark_scale_factor = 50" > "$conf"
sysctl -q -p "$conf"
if [ -f "$eo" ]; then
  # -M: SIGTERM / SIGKILL limits in KiB of MemAvailable. -s 100,100: both signals also when swap is free. -p: earlyoom runs first.
  args="-r 3600 -p -M 1048576,524288 -s 100,100 --prefer '^tensorfold' --avoid '^(systemd|sshd|dockerd|containerd|dbus-daemon|NetworkManager|gdm|Xorg|nvidia)'"
  [ -f "$eo.pre-tensorfold" ] || cp "$eo" "$eo.pre-tensorfold"
  { grep -v '^EARLYOOM_ARGS=' "$eo.pre-tensorfold" || true; printf 'EARLYOOM_ARGS="%s"\n' "$args"; } > "$eo.new"
  mv "$eo.new" "$eo"
  systemctl restart earlyoom || true
fi
echo "$(hostname): swappiness $(cat /proc/sys/vm/swappiness), compaction_proactiveness" \
  "$(cat /proc/sys/vm/compaction_proactiveness), min_free_kbytes $(cat /proc/sys/vm/min_free_kbytes)," \
  "tensorfold.slice swap $(systemctl show -p MemorySwapMax --value tensorfold.slice)," \
  "earlyoom $(systemctl is-active earlyoom 2>/dev/null || true)"
EOF
}

case "${1:-}" in ""|--undo) ;; *) echo "usage: scripts/host_tuning.sh [--undo]" >&2; exit 2 ;; esac
# Root runs the script from mktemp files (random names, owner-only), never from a fixed path in /tmp.
tmp=$(mktemp)
rtmp=""
cleanup() { rm -f "$tmp"; [ -z "$rtmp" ] || ssh -n -o BatchMode=yes "$WORKER_SSH" "rm -f '$rtmp'" || true; }
trap cleanup EXIT
node_script > "$tmp"
log "head"
sudo bash "$tmp" "${1:-}"
log "worker ($WORKER_SSH)"
rtmp=$(ssh -n -o BatchMode=yes "$WORKER_SSH" mktemp)
scp -q "$tmp" "$WORKER_SSH:$rtmp"
ssh -t "$WORKER_SSH" "sudo bash '$rtmp' ${1:-}"
log "done. Restart the server (./stop.sh, then ./start.sh): the engine sizes its memory at start."
