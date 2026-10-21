# Display-Only KV Staging

Experimental. Two-node integrity, inference, prefix reuse and basic concurrency
have been validated; full-window and sustained-load validation remain open.
Nothing in this directory is activated by the normal `start.sh`.

The staged installer also adds [DeepSeek native tool-output validation](../../docs/TOOL-OUTPUT.md).
The current draft rejects malformed, unavailable and schema-invalid calls
atomically. Prose streams until the first possible protocol marker, then the
remainder and all calls wait for final validation. `TOOL_OUTPUT_STREAMING=0`
restores whole-answer content buffering without disabling validation. It has CPU
parser/fake-HTTP coverage; earlier live checks do not validate this revision.
See the [2026-10-21 continuation record](../../runs/2026-10-21/parser-guardrails/README.md)
for the old-image baseline, CPU gates and pending new-image deployment.

The supported entry point is the self-contained root `./start-display-kv.sh`.
With explicit `--build` it prepares a source-tagged image on both nodes; normal
launches reuse the saved image without rebuilding or transferring. It prepares
both DRM modules with this
repo's `scripts/display_drm.sh`, probes the mapping, and creates a fresh launch
bundle. See [Display KV](../../docs/DISPLAY-KV.md) for usage and rollback.

## Deployment history

The 2026-10-19 deployment uses `tensorfold-dsv41:display-kv-fe26bd75d49e` with
strict tool-output rejection, `--thinking --reasoning-effort high`, and the
saved-image launcher. Below are the original display-allocation validation
results from the earlier cutover.

The 2026-10-18 cutover uses image
`tensorfold-dsv41:07aa02585f64-display-kv-stage1`, ID
`b3ffd2d45d5d189e622d49c71e5867ca744ee72667c129789a30bda384623a58`,
on both nodes. Both mapped the full 1792 MiB and passed two complete SM-side
write/read sweeps before the engine started. `/health` confirms a 2,109,440
position display-backed pool, a 1,048,576-token request context and zero ordinary
KV/indexer backing bytes. The real EngineMetrics adapter successfully sampled it.

Passed after startup: all five `verify.sh` checks, the user's
`tool-eval-bench --perf-only --benchy-runs 1 --base-url http://localhost:8100 --concurrency 1 --depth 65536 --model DeepSeek-V4.1-Flash-1017`,
24K-prefix reuse (24,064 cached tokens), and four simultaneous uncached 32K+
requests. The single benchmark run reported estimated prefill 1183 tok/s and
generation 44.6 tok/s; this is not a controlled performance comparison.

Use `./stop.sh` and then `./start-display-kv.sh` for future display-mode restarts;
the legacy bundle remains available for this saved deployment. Root `start.sh`
still selects the ordinary image. DRM preparation is runtime-only: after a
reboot it must be repeated with ranks stopped, followed by integrity probes.

## Design

- Map 1792 MiB of the NVIDIA DRM display reservation per rank, with no ordinary
  RAM prefix and no fallback to ordinary KV allocations if the mapping fails.
- Carve native packed compressed-KV and indexer planes from the mapping. The
  largest 2048-position-aligned pool that fits is selected from the model config;
  tensor padding and 256-byte plane alignment are included.
  With this model the staged calculation is 2,109,440 positions, using
  1,877,406,720 bytes of the 1,879,048,192-byte mapping. The remaining ~1.57 MiB
  is smaller than another aligned extent; these are calculated, not measured.
- Keep each request limited to the existing `--context`. Shared pool positions
  are not absolute RoPE positions, and must not enlarge the RoPE tables.
- Leave tokens, rings, compressor scratch, weights and graph workspaces in
  ordinary memory. The ordinary KV/indexer allocation is removed, not duplicated.
- Find the NVIDIA-owned DRM card by device identity. A connected
  `simple-framebuffer` console on another card is not a physical-monitor veto.
  A connected output with an enabled scanout encoder on the selected NVIDIA
  card stops the experiment. Connected-but-disabled outputs are not scanning
  out. Missing or unknown encoder state fails closed.
- Move mapped cache rows using bounded GPU-kernel copies, including overlaps.
  Do not route them through copy engines or CPU reads.
- Disk prompt spill uses the reused GLM tier's bounded CUDA staging path for
  mapped rows, never CPU reads or copy engines. `SPILL_GIB` is inherited from
  `.env`; a nonzero value enables per-rank SSD storage, and both ranks must
  independently write and restore. In-memory prefix reuse remains enabled and
  must be tested. Queued writes are drained on SIGTERM before exit.

The display reservation is outside normal PyTorch allocator accounting. It does
not add to `nvidia-smi`'s memory total. Existing host memory guards remain enabled.

## Cache Telemetry

The staged image adds on-demand KV reporting to the existing `/health` JSON and
Prometheus `/metrics` endpoints, including when the display option is off. No
periodic server task, CUDA allocation, device synchronization, or GPU-memory
query is added. EngineMetrics already polls these endpoints; its TensorFold
adapter needs no changes. Existing authentication behaviour is unchanged.

`/health` exposes the adapter's `pool_tokens`, `pool_free_tokens`, `kept_prompts`,
`kept_bytes`, `streams`, and `context_length`. It also reports used pool positions,
largest free extent, free-extent count, retained pool positions, compressed KV
and indexer payload bytes, ordinary backing bytes, display-mapping bytes, and
the backing type (`cuda` or `display`). Byte figures are local to rank 0, not
summed across the two replicated ranks. Rings, weights, RoPE, graph workspaces,
and host allocations are not included in KV/indexer payload bytes.

`kept_bytes` counts unique retained snapshot tensors outside the shared pool;
it does not count the already allocated KV planes twice. Pool occupancy includes
extents reserved for live requests and kept prompts, including aligned slack;
it is not the fraction of all KV tensors containing meaningful tokens.

`/metrics` exports corresponding `tensorfold:pool_*`, `tensorfold:kept_*`, and
`tensorfold:kv_cache_*_bytes` gauges, plus `tensorfold:kv_pool_usage_ratio` and
`tensorfold:kv_cache_backing_info{backing="display"}` (or `cuda`). The existing
`tensorfold:kv_cache_usage_ratio{pool="0"}` reports the maximum live stream's
occupied positions divided by its request-context limit, separately from pool
occupancy. Publishing one series avoids EngineMetrics' last-label-set parser
silently discarding other streams. During prefill it uses completed positions,
not the full submitted prompt; with no live streams it is zero even if kept
prompts still reserve pool space. Other TensorFold families retain their
existing stream-ratio behaviour.

Snapshots are approximate across fields during scheduler transitions, like the
existing metrics endpoint. A larger display pool must not change the reported
per-request context. Live HTTP validation is deferred until the image is built
and the service is explicitly restarted; a restart without rebuilding the image
will not install these new fields.

## CPU-Only Checks

From the recipe root, with a pristine pinned TensorFold checkout available:

```bash
TF_DISPLAY_SOURCE=/tmp/opencode/tfv41-display-source PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/display-kv/tests -v
```

These tests do not import the real Torch runtime, access the GPU, load a CUDA
library, map display memory, or contact either running rank. Source integration
checks need the checkout at the exact pinned revision; allocator tests do not.

Create an isolated launch bundle without starting anything:

```bash
python3 experiments/display-kv/prepare.py experiments/display-kv/bundle
```

The bundle snapshots the current recipe scripts and `.env`, substituting only
the experimental image, `PREFILL_CHUNK=512`, the display environment and DRM
device passthrough; `SPILL_GIB` is inherited from `.env`, and a nonzero value
enables the display-safe SSD spill tier. It has its own logs and is not used by
the normal launcher. Treat it as a snapshot: recreate it under a new name after
relevant recipe changes. It reuses model and compiled-kernel/rank-cache paths.

## Deferred Cutover

Only after the user confirms the current test has finished:

1. Save the benchmark output and stop the ordinary ranks with the recipe's
   `./stop.sh`. Never stop GDM or the display manager.
2. Build the separate image from the recipe root, then transfer it:

```bash
docker build -t tensorfold-dsv41:07aa02585f64-display-kv-stage1 experiments/display-kv
docker save tensorfold-dsv41:07aa02585f64-display-kv-stage1 | ssh 192.168.0.88 docker load
```

3. Use this repo's `scripts/display_drm.sh` for runtime DRM preparation on both
   nodes, with both model ranks stopped. Root `start-display-kv.sh` performs this
   step and the following integrity probes automatically.
   Confirm that the target NVIDIA card is unused for physical scanout, and
   identify it again after any module reload. Do not apply a global
   `connected` connector check or stop the firmware framebuffer console.
   Module preparation requires operator sudo; this experiment does not perform
   it automatically and makes no persistent modprobe or initramfs changes.
   The staged image also includes `/opt/display-kv-stage/probe.py`: with both
   ranks stopped, run it on each node in a disposable container with `--gpus all`,
   `--device /dev/dri`, `--ulimit memlock=-1`, and argument `--confirm-idle`.
   It writes the whole reservation and verifies it through SM kernels twice.
4. Start the isolated bundle, NOT the ordinary launcher:

```bash
bash experiments/display-kv/bundle/start.sh
```

5. Validate mapping integrity and the receipt in the rank logs, both ranks'
   capacity agreement, and real inference before benchmarking. Run
   `bash experiments/display-kv/bundle/verify.sh`, uncached 8K/32K/128K prompts,
   overlapping in-memory prefix reuse, and concurrent streams with nonzero
   extent bases. Confirm a request cannot exceed 1M even though the shared pool
   is larger. Measure headroom on both hosts, not only CUDA's free-memory count.
   SSD spill write, eviction, restore, CRC-failure handling and two-rank
   agreement have passed with 32 GiB per rank, four concurrent streams and
   170K-197K-token prompts; a restart restored a 237568-token prompt from disk
   in 0.05 s. Re-run them after any spill or display change.
6. Compare equivalent workloads with the ordinary baseline. Full-window,
   near-full-pool, image input and long mixed-concurrency tests are required
   before calling this production-ready. Disk spill is enabled through the
   reused GLM tier; validate it on both ranks rather than assuming the ordinary
   recipe's behavior carries over.

No build, transfer, DRM preparation, probe, or experimental startup is performed
by `prepare.py` or the CPU tests.

## Rollback

Stop the experimental ranks with `./stop.sh`.
Only after their mappings are gone, use `./start-display-kv.sh --restore-drm`
to restore DRM defaults. Start the ordinary recipe with `./start.sh`.
Its `.env`, image tag and start scripts are not modified by this experiment.
Do not run the normal and experimental launchers simultaneously: they use the
same rank container names, networking and model-cache paths.
