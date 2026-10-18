#!/usr/bin/env bash
# pf.sh TAG: one 8K warm prompt (kernel builds), then cold prefill 8K x2 and 32K/64K/128K with kit_bench. Prints tok/s.
cd "$(dirname "$0")"
KB=$HOME/deploy/tensorfold-v41-20261017/tf-lane/tools/dsv41/kit_bench.py
run() { python3 -I -u src127.py $KB --base http://172.17.0.1:8888 --model DeepSeek-v4.1-Flash-1017-TF "$@" >/dev/null; }
run prefill --lengths 8192 --reps 1 --out pf-$1-warm.json
run prefill --lengths 8192 --reps 2 --out pf-$1-8k.json
run prefill --lengths 32768,65536,131072 --reps 1 --out pf-$1-long.json
for f in pf-$1-warm pf-$1-8k pf-$1-long; do python3 -I -c "import json,sys;[print(sys.argv[1],r[\"length\"],round(r[\"ttft_s\"],2),round(r[\"prefill_tps\"])) for r in json.load(open(sys.argv[1]+\".json\"))[\"result\"]]" $f; done
