#!/usr/bin/env bash
# pinhot.sh CONTAINER CPUS N | show CONTAINER: pin the N busiest threads of the engine (by CPU time) to CPUS.
# "show" lists the 6 busiest threads: tid, current cpu, CPU ticks, allowed cpus. Undo: pinhot.sh CONTAINER 0-19 N.
set -u
P=$(docker inspect -f '{{.State.Pid}}' "$1")
top() { for t in /proc/$P/task/*; do awk '{print $1, $39, $14+$15}' "$t/stat"; done | sort -k3 -n -r | head -"$1"; }
if [ "${2:-}" = "" ] || [ "$2" = show ]; then
  top 6 | while read -r tid cpu ticks; do echo "$tid cpu=$cpu ticks=$ticks allowed=$(taskset -c -p "$tid" | awk '{print $NF}')"; done
  exit 0
fi
top "$3" | while read -r tid _ _; do sudo -n taskset -c -p "$2" "$tid" >/dev/null && echo "pinned $tid to $2"; done
