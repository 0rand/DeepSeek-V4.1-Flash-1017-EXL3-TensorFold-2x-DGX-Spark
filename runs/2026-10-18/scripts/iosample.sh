# iosample.sh SECONDS: per second: nvme reads/s, MB/s, pgsteal (direct+kswapd)/s, pgscan_direct/s, MemFree MB
dev=$(lsblk -no PKNAME $(df --output=source /home | tail -1) 2>/dev/null | head -1); dev=${dev:-nvme0n1}
rd() { awk -v d=$dev "\$3==d {print \$4, \$6}" /proc/diskstats; }
vm() { awk "/^pgsteal_(direct|kswapd) /{s+=\$2} /^pgscan_direct /{d=\$2} END{print s, d}" /proc/vmstat; }
read r0 s0 < <(rd); read p0 q0 < <(vm)
for i in $(seq 1 $1); do sleep 1; read r1 s1 < <(rd); read p1 q1 < <(vm)
  echo "$(date -u +%H:%M:%S) $dev reads/s $((r1-r0)) MB/s $(((s1-s0)/2048)) steal/s $((p1-p0)) scan_direct/s $((q1-q0)) free_MB $(awk "/MemFree/{print int(\$2/1024)}" /proc/meminfo)"
  r0=$r1; s0=$s1; p0=$p1; q0=$q1; done
