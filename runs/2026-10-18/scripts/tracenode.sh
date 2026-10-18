#!/usr/bin/env bash
# tracenode.sh start|stop OUTDIR CONTAINER: trace this node while a bench runs (needs passwordless sudo).
#   mem.txt     memtrace.bt: direct reclaim and compaction stalls (epoch prefix per line)
#   threads.txt every 0.5 s: epoch, then tid:cpu:cputicks for the 4 busiest engine threads
#   hw/         hwsample.sh: GPU clocks, CPU, PSI, vmstat counters
# Start it with setsid and all fds redirected. PIDs go to OUTDIR/pids, so stop kills only what start began.
set -u
cmd=$1; out=$2; ctr=${3:-}
stamp() { perl -MTime::HiRes=time -ne 'BEGIN{$|=1} printf "%.3f %s", time, $_'; }
case "$cmd" in
  start)
    mkdir -p "$out"; : > "$out/pids"
    sudo -n bpftrace -B none /tmp/memtrace.bt 2>&1 | stamp > "$out/mem.txt" &
    echo $! >> "$out/pids"
    bash /tmp/hwsample.sh 3600 "$out/hw" >/dev/null 2>&1 &
    echo $! >> "$out/pids"
    P=$(docker inspect -f '{{.State.Pid}}' "$ctr")
    hot=$(for t in /proc/$P/task/*; do awk '{print $1, $14+$15}' "$t/stat"; done | sort -k2 -n -r | head -4 | awk '{print $1}')
    echo "$hot" > "$out/hot-tids"
    ( while :; do
        line=$(date +%s.%N)
        for t in $hot; do line="$line $(awk '{print $1 ":" $39 ":" $14+$15}' /proc/$P/task/$t/stat 2>/dev/null)"; done
        echo "$line"; sleep 0.5
      done > "$out/threads.txt" ) &
    echo $! >> "$out/pids" ;;
  stop)
    sudo -n pkill -INT -x bpftrace
    while read -r p; do pkill -P "$p" 2>/dev/null; kill "$p" 2>/dev/null; done < "$out/pids"
    pkill -f 'nvidia-smi --query-gpu=timestamp,clocks.sm' 2>/dev/null
    sleep 2; pgrep -x bpftrace >/dev/null && sudo -n pkill -9 -x bpftrace
    wc -l "$out/mem.txt" "$out/threads.txt" ;;
esac
