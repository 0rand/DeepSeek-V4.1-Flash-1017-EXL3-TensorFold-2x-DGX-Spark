#!/usr/bin/env bash
# Deploy a TensorFold image, smoke-test structured output (roll back on failure), then run the suite,
# the --parallel grammar check and the benchmark. Usage: [SKIP_BENCH=1] deploy_gate.sh NEW_TAG ROLLBACK_TAG [NOTE]
set -u
NEW=$1 OLD=$2
TF=~/deploy/tensorfold-v41-20261017 V=~/deploy/v41-1017-20261017
log() { echo "[gate $(date -u +%H:%M:%SZ)] $*"; }
swap() {   # set IMAGE in the launcher, then restart both ranks
  sed -i "s/^IMAGE=tensorfold-dsv41:[0-9a-f]* .*/IMAGE=tensorfold-dsv41:$1   # $2/" "$TF/tfv41.sh"
  (cd "$TF" && ./tfv41.sh stop && ./tfv41.sh start) > "$TF/logs-gate-$1.log" 2>&1
}
rollback() { log "ROLLBACK to $OLD: $*"; swap "$OLD" "rollback after gate failure of $NEW"; log "ROLLED_BACK"; exit 1; }

swap "$NEW" "${3:-gate candidate}" || rollback "start failed"
grep -q serving "$TF/logs-gate-$NEW.log" || rollback "not serving"
log "serving $NEW"
cd "$V"
python3 -I - <<'EOF' || rollback "smoke failed"
import http.client, json, sys
def post(body):
    c = http.client.HTTPConnection("172.17.0.1", 8888, timeout=120, source_address=("127.0.0.1", 0))
    c.request("POST", "/v1/chat/completions", json.dumps({"model": "x", "max_tokens": 200, **body}),
              {"Content-Type": "application/json"})
    r = c.getresponse(); d = json.loads(r.read() or b"{}"); return r.status, d
schema = {"type": "json_schema", "json_schema": {"name": "t", "strict": True, "schema": {"type": "object",
          "properties": {"title": {"type": "string"}}, "required": ["title"], "additionalProperties": False}}}
s, d = post({"messages": [{"role": "user", "content": "Title for: fixing nginx 502"}], "response_format": schema,
             "chat_template_kwargs": {"enable_thinking": False}})
print("smoke schema", s, json.dumps(d)[:200], flush=True)
ok = s == 200 and isinstance(json.loads(d["choices"][0]["message"]["content"]).get("title"), str)
s2, d2 = post({"messages": [{"role": "user", "content": "Say hi."}], "chat_template_kwargs": {"enable_thinking": False}})
print("smoke plain", s2, json.dumps(d2)[:160], flush=True)
sys.exit(0 if ok and s2 == 200 else 1)
EOF
log "SMOKE_OK"
# Optional: ROUTER_ENV_FILE is a file that sets V41_ROUTER_API_KEY (the suite router tests need it).
if [ -n "${ROUTER_ENV_FILE:-}" ]; then set -a; . "$ROUTER_ENV_FILE"; set +a; fi
timeout 3600 python3 -I -u v41_suite.py > suite-$NEW.log 2>&1; echo "SUITE_EXIT=$?" >> suite-$NEW.log
cp suite-results.json suite-results-$NEW.json
log "$(grep -a 'SUITE_EXIT' suite-$NEW.log) $(grep -ac '^\[PASS' suite-$NEW.log) pass $(grep -ac '^\[FAIL' suite-$NEW.log) fail"
grep -q "SUITE_EXIT=0" "suite-$NEW.log" || rollback "suite failed"
curl -s -m 10 --interface 127.0.0.1 http://172.17.0.1:8888/v1/models > /dev/null || rollback "server dead after suite"
timeout 1800 python3 -I -u v41_grammar_check.py 2 > "grammar-check-$NEW.log" 2>&1; echo "CHECK_EXIT=$?" >> "grammar-check-$NEW.log"
log "$(tail -2 "grammar-check-$NEW.log" | tr '\n' ' ')"
grep -q "CHECK_EXIT=0" "grammar-check-$NEW.log" || rollback "grammar check failed"
curl -s -m 10 --interface 127.0.0.1 http://172.17.0.1:8888/v1/models > /dev/null || rollback "server dead after grammar check"
if [ "${SKIP_BENCH:-0}" = 1 ]; then log "GATE_DONE (bench skipped)"; exit 0; fi
timeout 3600 python3 -I -u v41_bench.py > bench.log 2>&1; echo "BENCH_EXIT=$?" >> bench.log
log "$(tail -1 bench.log)"
log "GATE_DONE"
