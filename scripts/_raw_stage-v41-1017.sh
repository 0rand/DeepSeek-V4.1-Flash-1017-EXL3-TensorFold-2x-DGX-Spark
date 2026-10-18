#!/usr/bin/env bash
# Stage DeepSeek-V4.1-Flash-1017 EXL3 2.9 bpw (LibertAIDAI) and the 1017 Engram
# tables on both DGX Spark nodes. This script downloads, verifies, builds the
# model view, and copies it to the worker. It does not stop or start a service.
set -euo pipefail

HF=$HOME/.local/bin/hf
WORKER=10.100.80.2
EXL3_REPO=LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw
EXL3_REV=afc7aa0aab98ee735c0f54e0a122019af1ac08f1
BASE_REPO=deepseek-ai/DeepSeek-V4.1-Flash-1017
BASE_REV=1bae0e1088993b9b82de08f59de5be5a34e42170
HUB=$HOME/.cache/huggingface/hub
EXL3_SNAP=$HUB/models--LibertAIDAI--DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw/snapshots/$EXL3_REV
BASE_SNAP=$HOME/deploy/v41-1017-20261017/engram-src-1017
VIEW=$HOME/.cache/huggingface/model-views/deepseek-v41-1017-exl3-2.9bpw-afc7aa0a
STATE=$HOME/deploy/v41-1017-20261017
SHARD47_SHA=76a05106ece259be83d1bfea0dcb54f23377864ee60fd321690f35e6698a2af5
SHARD48_SHA=9dc51433e195bc78b5ad846b68643cc0351d3b174e5e4309222d3b7402c83afd

# The Xet path used too much memory on this cluster before. Use plain HTTP.
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_DOWNLOAD_TIMEOUT=180 HF_HUB_ETAG_TIMEOUT=30

note() { echo "[stage-1017 $(date -u +%H:%M:%SZ)] $*"; }
fail() { note "ERROR: $*"; exit 1; }

dl() {
  local label=$1; shift
  for attempt in 1 2 3 4 5; do
    if "$HF" download "$@" >/dev/null; then note "$label download complete"; return 0; fi
    note "$label attempt $attempt failed; retry in 60 s"; sleep 60
  done
  fail "$label failed after 5 attempts"
}

phase=${1:-all}

if [[ $phase == all || $phase == download ]]; then
  note "download start"
  dl "EXL3" "$EXL3_REPO" --revision "$EXL3_REV" --exclude "assets/*"
fi

if [[ $phase == all || $phase == download || $phase == engram ]]; then
  # The hf CLI refuses files over 50 GB without hf_xet. Use resumable curl
  # against the resolve URL (it redirects to the CDN and accepts ranges).
  note "Engram download start (curl)"
  mkdir -p "$BASE_SNAP"
  base_url="https://huggingface.co/$BASE_REPO/resolve/$BASE_REV"
  pids=()
  for f in model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json config.json; do
    curl -sSfL --retry 20 --retry-all-errors --retry-delay 30 --connect-timeout 30 \
      -C - -o "$BASE_SNAP/$f" "$base_url/$f" &
    pids+=($!)
  done
  for p in "${pids[@]}"; do wait "$p" || fail "Engram curl failed (rerun phase engram to resume)"; done
  note "Engram download complete"
fi

if [[ $phase == all || $phase == verify ]]; then
  note "verify start (sha256 against the HF LFS ids)"
  curl -sf --max-time 60 "https://huggingface.co/api/models/$EXL3_REPO/tree/$EXL3_REV?recursive=1" > "$STATE/exl3-tree.json"
  python3 - "$STATE/exl3-tree.json" "$EXL3_SNAP" "$SHARD47_SHA" "$SHARD48_SHA" "$BASE_SNAP" > "$STATE/expected.sha256" <<'PY'
import json, sys
tree, snap, s47, s48, base = sys.argv[1:]
for f in json.load(open(tree)):
    if f["type"] == "file" and "lfs" in f and not f["path"].startswith("assets/"):
        print(f"{f['lfs']['oid']}  {snap}/{f['path']}")
print(f"{s47}  {base}/model-00047-of-00048.safetensors")
print(f"{s48}  {base}/model-00048-of-00048.safetensors")
PY
  note "$(wc -l < "$STATE/expected.sha256") LFS files to hash"
  # Hash in parallel; sha256sum -c accepts the lines in any order.
  split -n l/6 "$STATE/expected.sha256" "$STATE/sha-part."
  ls "$STATE"/sha-part.* | xargs -P6 -I{} sh -c 'sha256sum -c --quiet "{}" > "{}.out" 2>&1 || echo FAIL >> "{}.out"'
  if cat "$STATE"/sha-part.*.out | grep -q .; then cat "$STATE"/sha-part.*.out; fail "sha256 mismatch"; fi
  rm -f "$STATE"/sha-part.*
  note "verify OK"
fi

if [[ $phase == all || $phase == view ]]; then
  note "build view $VIEW"
  rm -rf "$VIEW.staging"; mkdir -p "$VIEW.staging/model" "$VIEW.staging/engram"
  for f in "$EXL3_SNAP"/*.json "$EXL3_SNAP"/*.safetensors; do
    ln "$(readlink -f "$f")" "$VIEW.staging/model/$(basename "$f")"
  done
  for f in model-00047-of-00048.safetensors model-00048-of-00048.safetensors model.safetensors.index.json config.json; do
    ln "$(readlink -f "$BASE_SNAP/$f")" "$VIEW.staging/engram/$f"
  done
  n=$(ls "$VIEW.staging/model"/model-*.safetensors | wc -l)
  [[ $n == 43 ]] || fail "view has $n EXL3 shards, expected 43"
  rm -rf "$VIEW"; mv "$VIEW.staging" "$VIEW"
  (cd "$VIEW" && find . -type f -printf '%P %s\n' | sort) > "$STATE/view-files.txt"
  note "view OK: $(wc -l < "$STATE/view-files.txt") files, $(du -sbL "$VIEW" | cut -f1) bytes"
fi

if [[ $phase == all || $phase == worker ]]; then
  note "copy view to worker"
  ssh -n -o BatchMode=yes "$WORKER" "mkdir -p $VIEW.staging"
  rsync -a --partial --info=stats1 "$VIEW/" "$WORKER:$VIEW.staging/"
  ssh -n -o BatchMode=yes "$WORKER" "cd $VIEW.staging && find . -type f -printf '%P %s\n' | sort" > "$STATE/worker-view-files.txt"
  cmp "$STATE/view-files.txt" "$STATE/worker-view-files.txt" || fail "worker file list differs"
  # Hash on the worker against the same expected sums, with paths remapped.
  sed -e "s#$EXL3_SNAP/#$VIEW.staging/model/#" -e "s#$BASE_SNAP/#$VIEW.staging/engram/#" "$STATE/expected.sha256" \
    | ssh -o BatchMode=yes "$WORKER" "cat > /tmp/v41-1017.sha256 && split -n l/6 /tmp/v41-1017.sha256 /tmp/v41-1017-part. && ls /tmp/v41-1017-part.* | xargs -P6 -I{} sh -c 'sha256sum -c --quiet {} || echo FAIL' > /tmp/v41-1017-verify.out 2>&1; cat /tmp/v41-1017-verify.out; rm -f /tmp/v41-1017-part.*" > "$STATE/worker-verify.out"
  if grep -q . "$STATE/worker-verify.out"; then cat "$STATE/worker-verify.out"; fail "worker sha256 mismatch"; fi
  ssh -n -o BatchMode=yes "$WORKER" "rm -rf $VIEW && mv $VIEW.staging $VIEW"
  note "worker view OK"
fi

note "DONE phase=$phase"
