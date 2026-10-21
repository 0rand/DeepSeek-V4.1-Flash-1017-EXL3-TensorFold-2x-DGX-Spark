# Display-Backed KV

`./start-display-kv.sh` is an opt-in, self-contained launcher for the experimental
display-only DeepSeek variant. The ordinary `start.sh` remains unchanged.

## Start

Run from the head after normal setup, host tuning, and the base image build:

```bash
./stop.sh
./start-display-kv.sh --build
./verify.sh
```

The launcher reads the current `.env`, including `WORKER_SSH`, `DISPLAY_KV_BASE`,
paths, networking, context, API binding and authentication settings. It does not
modify that file. This branch uses the 1.0.0 engine base for both lanes:
`IMAGE` and `DISPLAY_KV_BASE` default to `tensorfold-dsv41:07aa02585f64`. The
source installer requires the pinned TensorFold revision
`07aa02585f64cfb04c60a96de6554c0d4708ed46`; it refuses source drift rather than
guessing how to patch another version.

The first launch needs `--build`. After that, `./start-display-kv.sh` reuses the
image saved in `.display-kv-image`: it never builds or transfers an image, even
when source files changed. Use `--build` to explicitly prepare an update and
launch it, or `--build-only` to prepare it separately. A missing image or changed
image ID on either node causes a clear refusal, not an automatic build/copy.

Thinking defaults are controlled by `.env`: `THINKING=1` and
`REASONING_EFFORT=high`. Valid launcher efforts are `low`, `medium`, `high` and
`xhigh`. Requests can still override these server defaults. Each fresh display
launch snapshot inherits the current values; existing snapshots are unchanged.

Both ranks must be stopped even for `--build-only`. Building while weights are
resident risks exhausting host memory. Do not run another model server or GPU
workload during preparation or probes. The head requires sudo authorization;
the worker requires working `sudo -n` over key-based SSH. Never stop GDM or the
firmware framebuffer console to make the checks pass.

The startup sequence is:

1. Refuse running DeepSeek ranks and acquire a local launcher lock.
2. Load the saved image reference, or build a source-tagged image only when
   `--build` or `--build-only` was explicitly requested.
3. Verify image IDs on both nodes. Transfer a missing/different image only in
   explicit build mode, then atomically save the selected reference.
4. Authorize sudo, then prepare runtime DRM settings on both nodes.
5. Run two complete SM-side integrity sweeps across the 1792 MiB mapping on each
   node in disposable probe containers.
6. Create a private fresh launch snapshot in
   `experiments/display-kv/bundle-launch-*` and start it with ordinary preflight.

Generated snapshots preserve the latest recipe settings but select the variant
image, native packed KV, 512-token prefill chunks and inherit `SPILL_GIB` from
`.env`; a nonzero value enables the display-safe SSD spill tier. No stale
previous bundle or configuration snapshot is reused. Each bundle has its own
startup/build logs. `./stop.sh` still stops both rank container names and saves
their logs in the root recipe's `logs/` directory.

## Commands

| Command | Effect |
|---|---|
| `./start-display-kv.sh --check` | Read-only DRM/card/compute status on both nodes; safe with ranks running |
| `./start-display-kv.sh --build` | Explicitly prepare an updated image on both nodes, then prepare DRM, probe and start |
| `./start-display-kv.sh --build-only` | Build and copy the derived image; no DRM change, allocation probe or startup |
| `./start-display-kv.sh` | Use the saved image, prepare DRM, probe and start; never build or copy |
| `./start-display-kv.sh --restore-drm` | Restore boot-default DRM module parameters on both nodes; no server startup |
| `./stop.sh` | Stop both ranks and save their container logs |

Unreadable or empty DRM parameter files are reported as unknown, not as proof
of readiness. Some 580.x drivers do not expose their parameter values correctly;
the allocation and integrity probes are authoritative.

## Safety And Recovery

All DRM operations are local to `scripts/display_drm.sh`, streamed to the worker
over SSH. No checkout, script, binary or configuration from another model repo
is required. The helper discovers NVIDIA-owned cards and render nodes rather
than assuming `card0` or `card1`.

Preparation and restoration refuse active compute processes, active NVIDIA
scanout encoders, NVIDIA DRM mappings/open file descriptors, unknown connected
encoder state, or a positive module reference count. The simple-framebuffer TTY
is a different device and is not a reason to reject an idle NVIDIA reservation.
The helper reloads only `nvidia_drm`, never the compute driver. It does not write
modprobe configuration or rebuild initramfs. Reboot restores boot defaults.

After a cutover failure, the launcher stops only rank containers belonging to
its own image and attempts to restore DRM defaults on both nodes. It does not
silently start an ordinary-memory fallback. If another GPU/DRM user appears,
restoration can safely refuse; inspect the error, stop that workload yourself,
then use `--restore-drm`. The generated launcher does not force-remove a rank
that appeared concurrently.

To return to the normal engine:

```bash
./stop.sh
./start-display-kv.sh --restore-drm
./start.sh
```

Do not run the normal and display launchers concurrently. The display launcher
lock serializes display launches only; both variants intentionally share rank
names, API ports and model-cache paths.

## Capacity And Metrics

The 1792 MiB display mapping per rank holds a 2,109,440-position shared KV/indexer
pool, not a larger context for an individual request. Rings, weights, snapshots,
RoPE and workspaces remain in ordinary memory. The reservation is not added to
`nvidia-smi`'s normal VRAM accounting.

`/health` and `/metrics` report capacity, free/used positions, largest free
extent, retained prompts, snapshot bytes, and display versus ordinary backing.
EngineMetrics' TensorFold adapter can read these fields without changes. Byte
figures are per rank, not summed across replicated ranks. See the
[implementation notes](../experiments/display-kv/README.md#cache-telemetry) for
precise field semantics.

The staged image also includes [tool-output hardening](TOOL-OUTPUT.md) for the
DeepSeek app: an unoffered tool call becomes an explicit model-output error,
without retries or tool permission changes. Read its streaming tradeoff before
deploying the new image.

Disk prompt spill is enabled through the reused GLM spill tier. Mapped rows move
through the tier's bounded CUDA staging path (`_by_kernel`), never through CPU
reads or copy engines. `SPILL_GIB` is inherited from `.env`; a nonzero value
enables per-rank SSD storage, and both ranks must independently write and
restore. Queued writes are drained on SIGTERM before exit. In-memory prefix
reuse is enabled. Two-node integrity, basic inference, 24K-prefix reuse, four
concurrent 32K+ requests and a depth-65536 benchmark have passed. SSD write,
eviction, restore, CRC handling and four-stream concurrent restore have also
passed. Full-window and near-full-pool operation are not yet established as
production-safe.

## Disk Spill

The display-safe tier stores kept prompts under `TF_DS_SPILL_DIR` (default
`~/.cache/tensorfold`) up to `SPILL_GIB` per rank. `TF_DS_SPILL_HIGHWATER`,
`TF_DS_SPILL_MIN`, `TF_DS_SPILL_MIN_FREE_GIB`, `TF_DS_SPILL_KEEP_BUILDS`,
`TF_SPILL_WRITERS`, `TF_SPILL_DIRECT` and `TF_SPILL_FLUSH_S` tune background
writes. `/health["spill"]` and the `spill_*` Prometheus families expose entries,
bytes, write/load counters, restore latency, failures and free disk. A restart
keeps only prompts whose writes reached disk; `docker stop` sends SIGTERM, and
the shutdown hook waits up to 55 seconds for queued writes (inside `docker stop -t 60`).
The spill signature ignores the driver version, so nodes with frozen, differing
drivers can share stored prompts; the model, TensorFold sources, CUDA/torch
version, GPU name, KV layout and display size still gate compatibility.

Live validation on the two-node display recipe: 32 GiB per rank, four concurrent
streams, 170K-197K-token prompts, disk restores at 4608, 176640 and 195584
tokens, 26 entries and 43 writes with no write, CRC or load failures; both
ranks' entry counts matched. A restart restored a 237568-token prompt from disk
in 0.05 s (41 ms restore p50). Full-window and near-full-pool operation remain
unvalidated.

## Live Memory Guard

Updated display images replace the upstream allocator cap frozen after warm-up
with a live budget, refreshed before and after each prefill chunk and parallel
decode round on both ranks. The budget is current PyTorch reserved bytes plus
host `MemAvailable` minus `MEM_FLOOR_GIB`, capped at the CUDA device total.
Reclaimed host RAM therefore becomes usable without restarting the engine.
Near the floor, unused PyTorch cache blocks are returned to the host rather
than retaining a prefill peak until earlyoom kills the process.

Positive floors below 1.5 GiB are raised to 1.5 GiB: earlyoom sends SIGTERM at
1 GiB, so a 0.5 GiB reserve cannot protect the engine. An explicit zero still
disables the guard and is unsafe on these unified-memory nodes. The guard is
not a reservation against other processes or non-PyTorch allocations, and
genuine memory exhaustion can still break the distributed lane. Startup logs
identify the live guard; a broken step logs its last budget in bytes.

For a controlled test without a PyTorch allocator cap, add
`MEM_ALLOCATOR_CAP=0` to `.env` and keep `MEM_FLOOR_GIB` positive. Updated
launchers pass this as `TF_DS_MEM_CAP=0` on both ranks. This retains pressure
cleanup and budget telemetry, but the calculated ceiling is advisory only.
Startup logs explicitly say `hard cap DISABLED; cleanup only`. The default is
`MEM_ALLOCATOR_CAP=1`. This switch is implemented by the display overlay, not
the ordinary upstream image. It is read at startup, not changed live.

Keep earlyoom enabled during this test. Without the allocator cap, a memory
spike can instead trigger earlyoom or a real CUDA/host OOM; this is a diagnostic
mode, not proof that every workload fits. `MEM_FLOOR_GIB=0` is different: it
turns off cleanup and telemetry as well, and is not needed for the cap-only test.

Long-context attention indexing now tiles its packed ordering keys (16 MiB
per key tile, or one complete row if larger) and builds them in place. This
preserves the original score ordering and lower-index tie rule. The tile limit
does not include the input scores, output, index vector or sorting workspace.
A GPU regression case with 512 x 32768 scores and k=256 reduced incremental
peak allocated scratch from 448.2 MiB to 35.2 MiB, with identical indices.

Existing saved images do not acquire source changes automatically. With both
ranks stopped, run `./start-display-kv.sh --build-only` to build/copy the new
image, then `./start-display-kv.sh` to launch it. A regular start never rebuilds.

## Offline Tests

No live Docker, SSH, sudo or DRM operations occur in these launcher/helper tests:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/display_drm_test.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/start_display_kv_test.py
```

Engine-patch and telemetry checks need a pristine checkout of the pinned
TensorFold revision. Most use fake tensors. Top-k equivalence tests use CPU
PyTorch when installed; CUDA equivalence, graph capture and peak-memory tests
run only when CUDA is available. No model weights are loaded:

```bash
TF_DISPLAY_SOURCE=/path/to/TensorFold PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/display-kv/tests -v
```
