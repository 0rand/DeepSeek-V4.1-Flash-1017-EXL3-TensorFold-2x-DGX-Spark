#!/usr/bin/env bash
# bertholomus kit_bench.py against our TensorFold server, with their published settings (README v0.6.0 / v0.5.1 tables).
set -u
D=$(cd "$(dirname "$0")" && pwd)
cd "$D" || exit 1
KB=$HOME/deploy/tensorfold-v41-20261017/tf-lane/tools/dsv41/kit_bench.py
MODEL=DeepSeek-v4.1-Flash-1017-TF
run() { python3 -I -u "$D/src127.py" "$KB" --base http://172.17.0.1:8888 --model "$MODEL" "$@"; }
stamp() { echo "[kb $(date -u +%H:%M:%SZ)] $*"; }
stamp "start, kit_bench sha256 $(sha256sum "$KB" | cut -c1-16)"
run ready --timeout 120 --out ready.json || { stamp "server not ready"; exit 1; }
stamp "decode set a (512 tokens, 3 reps)";  run decode --set a --tokens 512 --reps 3 --out decode-a.json
stamp "decode set b (384 tokens, 3 reps)";  run decode --set b --tokens 384 --reps 3 --out decode-b.json
stamp "concurrent set b (2,4 streams, 384 tokens, 9 reps)"; run concurrent --set b --streams 2,4 --tokens 384 --reps 9 --out conc-b.json
stamp "sustained 4 streams 90 s set b"; run sustained --streams 4 --seconds 90 --set b --out sustained.json
stamp "prefill 8K (best of 2)"; run prefill --lengths 8192 --reps 2 --out prefill-8k.json
stamp "prefill 32K/64K/128K"; run prefill --lengths 32768,65536,131072 --reps 1 --out prefill-long.json
for i in 1 2 3; do stamp "depth 128K run $i"; run depth --lengths 131072 --tokens 256 --out depth-$i.json; done
stamp "KB_DONE"
