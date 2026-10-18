#!/usr/bin/env bash
# TensorFold DeepSeek-V4.1-Flash-1017 (Python engine, Elevatormusic/TensorFold branch v41-dsml-tool-choice), TP=2.
#   tfv41.sh start    start rank 0 (head), wait for its store port, then rank 1 (worker); wait for /v1/models
#   tfv41.sh stop     stop and remove both rank containers (caches stay on disk)
#   tfv41.sh status   containers and API
# The vLLM boot unit (dgx-v41-1017-serving) must be stopped first: both use port 8888 and ~100 GiB a node.
set -euo pipefail

IMAGE=tensorfold-dsv41:fd2365042f56   # release candidate: PR 1-3 with review rounds 1-2
WORKER=10.100.80.2
MASTER=10.100.80.1
VIEW=$HOME/.cache/huggingface/model-views/deepseek-v41-1017-exl3-2.9bpw-afc7aa0a
CACHE=$HOME/tfv41-cache
NAME=DeepSeek-v4.1-Flash-1017-TF
ALIAS=DeepSeek-v4.1-Flash-1017-EXL3   # the router's current upstream name: no router change needed
API=http://172.17.0.1:8888/v1
LOG=$HOME/deploy/tensorfold-v41-20261017/logs

log() { echo "[tfv41 $(date -u +%H:%M:%SZ)] $*"; }
wssh() { ssh -n -o BatchMode=yes -o ConnectTimeout=10 "$WORKER" "$@"; }
healthy() { curl --interface 127.0.0.1 -fsS -m 10 "$API/models" 2>/dev/null | grep -q "\"$NAME\""; }

run_args() {  # $1 = rank
  local r=$1
  cat <<EOF
-d --name tfv41-r$r --gpus all --network host --ipc host --shm-size 32g
--ulimit memlock=-1 --ulimit stack=67108864 --device /dev/infiniband --cap-add IPC_LOCK --cap-add SYS_PTRACE
-v $VIEW/model:/model:ro -v $VIEW/engram:/engram:ro -v $CACHE:/cache
-e TF_DS_REPLAY=1 -e TF_DS_REPLAY_FLOOR=1 -e TF_DS_PREFILL_CHUNK=2048
-e TF_DS_RANK_CACHE=/cache/tf-ds-rank -e TF_DS_RANK_CACHE_READERS=32
-e TF_DS_WARM_LENGTHS=1,17,33,131,514,1024,2113 -e TF_DS_ENGRAM=/engram
-e TF_RDMA_DEVICES=rocep1s0f1,roceP2p1s0f1 -e TF_RDMA_GID_INDEX=3
-e NCCL_IB_HCA=rocep1s0f1,roceP2p1s0f1 -e NCCL_IB_GID_INDEX=3 -e NCCL_SOCKET_IFNAME=enp1s0f1np1
-e TORCH_EXTENSIONS_DIR=/cache/torch_extensions -e HF_HUB_OFFLINE=1
--entrypoint tensorfold $IMAGE serve /model --tp 2 --rank $r --master $MASTER
--host 172.17.0.1 --port 8888 --name $NAME --alias $ALIAS
--context 1048576 --vision --parallel 4 --mtp-drafts 5 --temperature 1.0 --top-p 1.0 --no-update-check
EOF
}

case "${1:-status}" in
  start)
    mkdir -p "$CACHE" "$LOG"; wssh "mkdir -p $CACHE"
    if docker ps --format '{{.Names}}' | grep -qE '^(dsv41-1017|glm53|sparkrun_)'; then
      log "another model is running on the head; stop it first"; exit 1
    fi
    log "starting rank 0 (head)"
    docker rm -f tfv41-r0 >/dev/null 2>&1 || true
    # shellcheck disable=SC2046
    docker run $(run_args 0 | tr '\n' ' ') >/dev/null
    for i in $(seq 1 120); do ss -ltn | grep -q ':29551 ' && break; sleep 2; done
    ss -ltn | grep -q ':29551 ' || { log "rank 0 store port 29551 never listened"; docker logs --tail 40 tfv41-r0; exit 1; }
    log "starting rank 1 (worker)"
    wssh "docker rm -f tfv41-r1 >/dev/null 2>&1 || true"
    wssh "docker run $(run_args 1 | tr '\n' ' ')" >/dev/null
    log "waiting for $API/models (first start writes the rank caches: slow)"
    for i in $(seq 1 360); do
      healthy && { log "serving $NAME"; exit 0; }
      for c in tfv41-r0; do docker ps --format '{{.Names}}' | grep -q "^$c$" || { log "$c exited"; docker logs --tail 60 $c; exit 1; }; done
      wssh "docker ps --format '{{.Names}}'" | grep -q '^tfv41-r1$' || { log "tfv41-r1 exited"; wssh "docker logs --tail 60 tfv41-r1"; exit 1; }
      sleep 10
    done
    log "not ready after 60 min"; exit 1
    ;;
  stop)
    docker logs tfv41-r0 > "$LOG/r0-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 || true
    wssh "docker logs tfv41-r1" > "$LOG/r1-$(date -u +%Y%m%dT%H%M%SZ).log" 2>&1 || true
    docker stop -t 60 tfv41-r0 >/dev/null 2>&1 || true; docker rm tfv41-r0 >/dev/null 2>&1 || true
    wssh "docker stop -t 60 tfv41-r1 >/dev/null 2>&1; docker rm tfv41-r1 >/dev/null 2>&1" || true
    log "stopped"
    ;;
  status)
    docker ps -a --format '{{.Names}} {{.Status}}' | grep tfv41 || echo "no head rank"
    wssh "docker ps -a --format '{{.Names}} {{.Status}}'" | grep tfv41 || echo "no worker rank"
    healthy && echo "API: serving $NAME" || echo "API: not serving"
    ;;
  *) echo "usage: $0 start|stop|status" >&2; exit 2 ;;
esac
