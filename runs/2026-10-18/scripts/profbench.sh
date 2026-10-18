#!/usr/bin/env bash
# profbench.sh TAG [PASSES]: trace both nodes (tracenode.sh) during PASSES runs of pf.sh, then save the rank logs.
# Output: ~/prof/TAG/{head,worker}/ traces, pf-TAG<n>-*.json, r0.log, r1.log (docker logs -t since the start).
set -u
tag=$1; passes=${2:-3}; W=10.100.80.2
o=$HOME/prof/$tag; mkdir -p "$o"
since=$(date -u +%Y-%m-%dT%H:%M:%SZ)
setsid bash /tmp/tracenode.sh start "$o/head" tfv41-r0 >/dev/null 2>&1 < /dev/null &
ssh -n $W "setsid bash /tmp/tracenode.sh start /tmp/prof-$tag tfv41-r1 >/dev/null 2>&1 < /dev/null &"
sleep 8
cd ~/deploy/v41-1017-20261017/kitbench-fd23
for n in $(seq 1 "$passes"); do
  echo "== pass $n $(date -u +%T)"; ./pf.sh "$tag$n"; mv pf-"$tag$n"-*.json "$o/"
done
bash /tmp/tracenode.sh stop "$o/head"
ssh -n $W "bash /tmp/tracenode.sh stop /tmp/prof-$tag"
mkdir -p "$o/worker"; scp -q -r "$W:/tmp/prof-$tag/*" "$o/worker/"
docker logs -t --since "$since" tfv41-r0 > "$o/r0.log" 2>&1
ssh -n $W "docker logs -t --since $since tfv41-r1" > "$o/r1.log" 2>&1
echo "== done $(date -u +%T) -> $o"
