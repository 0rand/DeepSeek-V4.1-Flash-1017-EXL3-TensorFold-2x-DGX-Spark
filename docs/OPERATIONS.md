# Operations

## Start, stop, check

The optional self-contained display-memory variant is launched with
`./start-display-kv.sh` after `./stop.sh`. See [Display KV](DISPLAY-KV.md) for
sudo requirements, safety guards, probes, metrics and rollback. The ordinary
launcher below does not enable display-backed KV.

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

## Native tool-output errors (display-KV variant)

The staged display-KV image uses [strict tool-output validation](TOOL-OUTPUT.md).
`model_output_error` / `unavailable_tool_call` from
`source=tensorfold.tool_output_guard` is a terminal model-response rejection,
not a broken distributed lane. Do not enable an unoffered tool, retry the same
generation automatically or restart the lane to hide this error.

Prose before the first raw protocol candidate can stream. The remainder and all
executable calls wait for final validation. For a content-latency rollback, set
`TOOL_OUTPUT_STREAMING=0` in `.env` and use
`./stop.sh && ./start-display-kv.sh`. Validation stays enabled. Source changes
require the explicit `--build` option on the first restart; plain launches reuse
the saved image. Have head-node sudo authorization available before stopping a
healthy service, because DRM preparation needs it. Do not change the engine pin
to a new upstream release without porting the experiment's checked transforms.

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
- **CUDA OOM with free memory.** The ordinary upstream image sets a fixed PyTorch allocator ceiling after warm-up, leaving `MEM_FLOOR_GIB` available to the host. Updated display images instead refresh the budget during serving, release unused allocator cache under pressure, and bound attention-indexing scratch; see [Live Memory Guard](DISPLAY-KV.md#live-memory-guard). Build the updated display image explicitly; restarting an older saved image does not apply the fix. Keep `PREFILL_CHUNK=512` and a positive memory floor: disabling it risks a host OOM. Smaller chunks can reduce prefill throughput; the configured context is not a guarantee that every workload fits.
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
