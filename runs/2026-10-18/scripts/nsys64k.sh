#!/usr/bin/env bash
# nsys64k.sh: both ranks under nsys (software CUDA trace), 6 s captured in the middle of one 64K prompt, kernel summary
# per rank, then production serving again.
set -u
W=10.100.80.2
cd ~/deploy/tensorfold-v41-20261017 && ./tfv41.sh stop >/dev/null 2>&1
N="launch --trace=cuda-sw,nvtx --cuda-flush-interval=500"
ENTRY0=nsys PRE0="$N --session-new=tf0 tensorfold" ENTRY1=nsys PRE1="$N --session-new=tf1 tensorfold" \
  timeout 1500 ./tfv41-prof.sh start 2>&1 | tail -1
cd ~/deploy/v41-1017-20261017/kitbench-fd23
KB=$HOME/deploy/tensorfold-v41-20261017/tf-lane/tools/dsv41/kit_bench.py
B="python3 -I -u src127.py $KB --base http://172.17.0.1:8888 --model DeepSeek-v4.1-Flash-1017-TF"
$B prefill --lengths 8192 --reps 1 --out /tmp/ns-warm.json >/dev/null
($B prefill --lengths 65536 --reps 1 --out /tmp/ns-64k.json >/dev/null &)
sleep 24
docker exec tfv41-r0 nsys start --session=tf0 --sample=none --cpuctxsw=none --force-overwrite=true --output=/cache/prof/r0-sw &
ssh -n $W "docker exec tfv41-r1 nsys start --session=tf1 --sample=none --cpuctxsw=none --force-overwrite=true --output=/cache/prof/r1-sw"
wait; sleep 6
docker exec tfv41-r0 nsys stop --session=tf0 >/dev/null 2>&1 &
ssh -n $W "docker exec tfv41-r1 nsys stop --session=tf1" >/dev/null 2>&1
wait; sleep 5
grep MemAvailable /proc/meminfo
for r in 0 1; do
  run="docker exec tfv41-r$r sh -c 'cd /cache/prof && nsys stats -r cuda_gpu_kern_sum --format csv --force-export=true -o r$r-sw r$r-sw.nsys-rep >/dev/null 2>&1; ls -la r$r-sw*'"
  if [ $r = 0 ]; then bash -c "$run"; else ssh -n $W "$run"; fi
done
scp -q $W:tfv41-cache/prof/r1-sw_cuda_gpu_kern_sum.csv ~/tfv41-cache/prof/ 2>/dev/null
for r in 0 1; do echo "== rank $r"; python3 /tmp/kernsum.py ~/tfv41-cache/prof/r$r-sw_cuda_gpu_kern_sum.csv 22; done
cd ~/deploy/tensorfold-v41-20261017 && ./tfv41.sh stop >/dev/null 2>&1 && ./tfv41.sh start 2>&1 | tail -1
