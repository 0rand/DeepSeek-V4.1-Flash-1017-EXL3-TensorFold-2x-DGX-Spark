#!/usr/bin/env bash
# hwsample.sh SECONDS OUTDIR: samples a node while a benchmark runs (read-only, no docker exec).
#   gpu.csv   nvidia-smi every 0.5 s: clocks, power, temperature, throttle reasons, utilization
#   host.txt  every 0.5 s: epoch, CPU busy %, mean core MHz, PSI cpu/memory/io avg10 and total, free MB,
#             pgscan/pgsteal/compact_stall/pswpout counters, context switches, interrupts
set -u
secs=$1; out=$2; mkdir -p "$out"
nvidia-smi --query-gpu=timestamp,clocks.sm,clocks.mem,power.draw,temperature.gpu,clocks_event_reasons.active,utilization.gpu,utilization.memory \
  --format=csv,noheader -lms 500 > "$out/gpu.csv" 2>&1 &
gp=$!
end=$(( $(date +%s) + secs ))
read -r _ u0 n0 s0 i0 w0 q0 sq0 st0 _ < /proc/stat
while [ "$(date +%s)" -lt "$end" ]; do
  sleep 0.5
  read -r _ u n s i w q sq st _ < /proc/stat
  busy=$(( (u+n+s+q+sq+st) - (u0+n0+s0+q0+sq0+st0) )); tot=$(( busy + (i+w) - (i0+w0) ))
  u0=$u; n0=$n; s0=$s; i0=$i; w0=$w; q0=$q; sq0=$sq; st0=$st
  mhz=$(cat /sys/devices/system/cpu/cpu*/cpufreq/scaling_cur_freq 2>/dev/null | awk '{s+=$1; n++} END {if (n) printf "%d", s/n/1000}')
  psi=$(for r in cpu memory io; do awk -v r=$r '/^some/ {split($2,a,"="); split($5,b,"="); printf "%s=%s/%s ", r, a[2], b[2]}' /proc/pressure/$r; done)
  vm=$(awk '/^(pgscan_kswapd|pgscan_direct|pgsteal_kswapd|pgsteal_direct|compact_stall|pswpout|pgmajfault|pgmigrate_success|compact_daemon_migrate_scanned|compact_migrate_scanned|thp_fault_alloc) /{printf "%s=%s ", $1, $2}' /proc/vmstat)
  cs=$(awk '/^ctxt/{c=$2} /^intr/{i=$2} END{printf "ctxt=%s intr=%s", c, i}' /proc/stat)
  free=$(awk '/MemFree/{f=$2} /MemAvailable/{a=$2} END{printf "%d avail=%d", f/1024, a/1024}' /proc/meminfo)
  echo "$(date +%s.%N | cut -c1-14) busy=$(( tot > 0 ? 100*busy/tot : 0 )) mhz=$mhz free=$free $psi$vm$cs"
done > "$out/host.txt"
kill $gp 2>/dev/null
