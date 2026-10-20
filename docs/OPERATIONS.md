# Operations

## Start, stop, check

```bash
./start.sh
```

`start.sh` first runs `scripts/build_extensions.sh`. It compiles the CUDA extensions on both nodes before the weights load, and removes lock files that a stopped compile left. A compile while the weights fill the memory can make earlyoom stop the engine; a left lock file makes the next start wait forever. Logs: `logs/build-extensions-head.log` and `logs/build-extensions-worker.log`.

```bash
./status.sh
```

```bash
./verify.sh
```

```bash
./stop.sh
```

- **Start times.** The first start writes each rank's weight cache to `CACHE_DIR` (~15 min). Later starts read it back and come up in ~2-4 min.
- **Short prompts are fast from the first request.** Compiled kernels are kept in `CACHE_DIR/triton` and `CACHE_DIR/nv`, and the warm-up covers prompts of 1 to 17 tokens. Before this, a 3-16 token prompt took 2.6-6.1 s the first time after each start; now it takes 0.1 s.
- **First long prompt.** The first long prompt after a start (8K+ tokens) takes ~6 s longer than later ones. The README's 8K row includes this. A longer warm-up list does not remove it; the likely cause is cold Engram reads from NVMe.
- **Prompts survive a restart.** With `SPILL_GIB` above 0 (default 64), kept prompts also go to `CACHE_DIR/tf-ds-spill` on each node. After a restart, a conversation continues from disk (an 18K-token prompt: 2.7 s instead of 16.8 s). Old entries go after 7 days or past the size limit.

## When the lane breaks

The engine keeps serving when one request fails:
- A fault in one request's own work (its grammar, its sampling, its client going away) ends only that request.
- A device fault (CUDA error, out of memory) or a lost rank breaks the lane. NCCL is aborted, and the lane cannot serve again in that process.

A broken lane ends both rank processes with **exit code 75** after 5 s (`TF_DS_EXIT_ON_BREAK=1`, the default). The 5 s give the live requests time to get their error.

- **Restart by hand.** Run:

```bash
./stop.sh && ./start.sh
```

- **Restart on its own.** Use a supervisor, with these rules:
  - Both ranks must restart together, rank 0 first, because rank 1 connects to rank 0's port 29551.
  - The simplest supervisor is a systemd unit on the head that runs `./start.sh`, with `ExecStop=./stop.sh` and `Restart=on-failure`, and that checks `./status.sh`.
  - `RESTART_POLICY=on-failure` in `.env` (docker's own restart) restarts each container alone. It works when rank 0 comes back first, but no tool enforces that order.
- **Out of memory.** With `scripts/host_tuning.sh`, earlyoom stops the engine when a node has less than 1 GiB available, so the node stays reachable by ssh. `journalctl -u earlyoom` shows when it did. Then restart by hand as above.
- **Out of memory under load.** Some users see earlyoom stop the engine with several long requests at the same time, or with many images in one agent session ([#21](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/issues/21)).
  Short tests often do not show it. To find it, run this on the head while the load runs:

  ```bash
  watch -n 1 "grep MemAvailable /proc/meminfo"
  ```

  If `MemAvailable` goes near 1 GiB (1,048,576 kB), change one setting in `.env` at a time, then run `./stop.sh && ./start.sh`:
  1. `PARALLEL=2`: fewer prompts fill at the same time, so the peak is lower. Each request still gets the full context.
  2. `MEM_FLOOR_GIB=3`: the engine stops its growth earlier, so more memory stays free for peaks.
  3. Do not turn off earlyoom or lower its limit. Without it, a node that runs out of memory can stop answering ssh until a power cycle.

  These settings are not measured on this recipe yet. Issue #21 tracks the memory trace and the engine fix.
- **Boot.** Do not let a second server also start at boot (a vLLM unit, another model). Both servers want port 8888 and ~100 GB per node, and the second one fails or runs a node out of memory.

## Things that hurt on DGX Spark

- **Do not `docker exec` into a rank container while it serves.** The exec process joins the container's cgroup, and its memory counts against the rank.
- **Do not stop the display manager on a Spark** (`systemctl stop gdm`). On the measured pair it powered the node off.
- **`/health` 200 is not proof that TP=2 works.** Only a real completion proves it; `./verify.sh` sends five.
- **Memory.** Each node holds ~100 GB for its half of the weights, the KV pool and the 1M context. Close other GPU programs before you start.

## Logs

- `docker logs tfv41-r0` (head) and `ssh <worker> docker logs tfv41-r1`.
- `./stop.sh` saves both logs to `logs/` first.
- Each request prints one line with its numbers:

```text
[tensorfold] done req-... prompt=... cached=... tokens=... tok/s=... ttft=...s rounds=... accepted=.../...
```

## Update the engine

1. Set `TF_COMMIT` and `IMAGE` in `.env`.
2. Build the image and copy it to the worker:

```bash
./docker/build.sh
```

3. Deploy with rollback:

```bash
./scripts/deploy_gate.sh <new-tag> <old-tag>
```

`scripts/deploy_gate.sh` deploys the new image, runs a live structured-output request, the 32-test suite and the `--parallel` grammar check, and goes back to the old image if any step fails.
