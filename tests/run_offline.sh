#!/usr/bin/env bash
# Run the offline tests: no server, no GPU, no .env. CI runs this script; run it before you open a pull request.
#   tests/run_offline.sh
# It runs tests/host_tuning_test.sh, then the Python unittest files (test_*.py and *_test.py) in tests/ and in
# each experiments/*/tests/ folder. A test that needs a GPU, a server or the tensorfold package must skip itself
# (unittest.skipUnless). The exit code is 1 when any step fails.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT" || exit 1
PY=${PYTHON:-python3}
FAILS=0

step() {  # $1 = name, then the command
  local name=$1
  shift
  echo "== $name"
  "$@"
  local rc=$?
  if [ "$rc" = 0 ]; then
    echo "[PASS] $name"
  elif [ "$rc" = 5 ] && [ "${1:-}" = "$PY" ]; then
    echo "[SKIP] $name: no test files"            # unittest exits with code 5 when it finds no tests
  else
    echo "[FAIL] $name: exit code $rc"
    FAILS=$((FAILS + 1))
  fi
}

step "host_tuning_test.sh" bash tests/host_tuning_test.sh

for dir in tests experiments/*/tests; do
  [ -d "$dir" ] || continue
  for pattern in 'test_*.py' '*_test.py'; do
    step "unittest $dir/$pattern" "$PY" -m unittest discover -s "$dir" -t "$dir" -p "$pattern" -v
  done
done

echo
if [ "$FAILS" = 0 ]; then
  echo "run_offline: no step failed"
  exit 0
fi
echo "run_offline: $FAILS step(s) failed"
exit 1
