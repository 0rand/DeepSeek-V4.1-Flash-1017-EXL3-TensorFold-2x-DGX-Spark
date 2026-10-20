# Display-Backed KV

`./start-display-kv.sh` is an opt-in, experimental launcher for a display-only
DeepSeek variant. `./start.sh` is not changed and does not start this variant.

The feature is a capacity feature. It moves the KV pool out of ordinary host
memory. It does not give more host-memory headroom, and the saved tester runs show
a lower prefill tok/s (see Cost And Capacity). Read that section before you use it.

## Which launcher uses which code

| | `./start.sh` | `./start-display-kv.sh` |
|---|---|---|
| Image | `IMAGE` from `.env` (stock) | `tensorfold-dsv41:display-kv-<hash>` (derived) |
| KV pool | ordinary memory | 1792 MiB display mapping on each rank |
| SSD spill | stock spill of the base image | patched spill of the derived image |
| Spill folder | `dsv41-spill` under `TF_DS_SPILL_DIR` | `spill-<compat>/rank<N>` under `TF_DS_SPILL_DIR` |
| Spill settings | `TF_DS_SPILL_DIRECT` (stock) | `TF_SPILL_*` and `TF_DS_SPILL_*` (patched) |
| Tool-output guard | not in the image | ON by default ([details](TOOL-OUTPUT.md)) |
| Memory guard | stock | live budget, hard cap off ([details](#live-memory-guard)) |
| Prefill chunk | 2048 | 512 (`PREFILL_CHUNK` in `.env`) |

The patched spill code exists only in the derived image. `apply.py` runs only in
`experiments/display-kv/Dockerfile`, and it writes a new image tag. The stock
image is never changed. The two spill folders have different names, so the two
lanes never read each other's files.

One exception: if you start the derived image with `./start.sh` (for example by
setting `IMAGE` to the variant), that run uses the patched spill and has the
tool-output guard code. The guard stays off in this case unless
`TF_DS_STRICT_TOOL_OUTPUT=1` is set. Do not do this. Use the display launcher.

Settings that only the display launcher uses are in a commented block at the end
of `.env.example`. `./start.sh` ignores them.

## Start

Run from the head after normal setup, host tuning and the base image build:

```bash
./stop.sh
./start-display-kv.sh --build
./verify.sh
```

The launcher reads the current `.env`, including `WORKER_SSH`, `IMAGE`, paths,
networking, context, API binding and authentication settings. It does not change
that file. The source installer needs the pinned TensorFold revision
`07aa02585f64cfb04c60a96de6554c0d4708ed46`. It refuses source drift instead of
guessing how to patch another version. The error names the file and shows the
expected and found SHA-256 values. A base image with a local patch counts as
drift: build a pristine base from the committed Dockerfile first.

The first launch needs `--build`. After that, `./start-display-kv.sh` reuses the
image saved in `.display-kv-image`. It never builds or transfers an image, even
when source files changed. Use `--build` to prepare an update and launch it, or
`--build-only` to prepare it separately. A missing image, or a changed image ID on
either node, gives a clear refusal and no automatic build or copy.

Thinking is controlled by `.env`: `THINKING` (default `0` in `.env.example`) and
`REASONING_EFFORT` (`low`, `medium`, `high` or `xhigh`; empty means `high`). A
request can still override these server defaults. Each new launch bundle copies
the current values. Existing bundles do not change.

Both ranks must be stopped, also for `--build-only`. A build while the weights
are resident risks a host-memory exhaustion. Do not run another model server or
GPU workload during preparation or probes. The head needs sudo authorization:
the launcher tries `sudo -n true` first and calls `sudo -v` only when that fails,
so a user with passwordless sudo can run it without a terminal. If both fail, it
stops before any DRM change. The worker needs `sudo -n` over key-based SSH. Never
stop GDM or the firmware framebuffer console to make the checks pass.

The start sequence is:

1. Refuse to run when DeepSeek ranks are running, and take a local launcher lock.
2. Load the saved image reference, or build an image tagged with the source hash
   only when `--build` or `--build-only` was given.
3. Check the image IDs on both nodes. Transfer a missing or different image only
   in build mode, then save the selected reference in one atomic step.
4. Authorize sudo, then prepare the runtime DRM settings on both nodes.
5. Run two complete SM-side integrity sweeps across the 1792 MiB mapping on each
   node, in disposable probe containers.
6. Make a fresh private launch bundle in `experiments/display-kv/bundle-launch-*`
   and start it with the ordinary preflight.

A bundle is a copy of `start.sh` with the display settings added, plus a
generated `.env`. It selects the derived image, native packed KV, a 512-token
prefill chunk (`PREFILL_CHUNK`) and the display-only switches. `SPILL_GIB` comes
from `.env`; a nonzero value turns the patched spill on. These variables pass
through from your shell to the container when you set them: `TF_DS_SPILL_MIN`,
`TF_DS_SPILL_MIN_FREE_GIB`, `TF_DS_SPILL_KEEP_BUILDS`, `TF_SPILL_WRITERS`,
`TF_SPILL_DIRECT` and `TF_SPILL_FLUSH_S`. No old bundle is reused. Each bundle
has its own start and build logs. `./stop.sh` stops both rank containers and
saves their logs in the `logs/` folder of the recipe root.

## Switches in .env

| Name | Default of the launcher | Effect |
|---|---|---|
| `PREFILL_CHUNK` | 512 | Prefill chunk in tokens (container `TF_DS_PREFILL_CHUNK`) |
| `MEM_ALLOCATOR_CAP` | 0 | 1 turns on the hard CUDA allocator cap (container `TF_DS_MEM_CAP`) |
| `STRICT_TOOL_OUTPUT` | 1 | 0 turns the [tool-output guard](TOOL-OUTPUT.md) off (container `TF_DS_STRICT_TOOL_OUTPUT`) |
| `SPILL_GIB` | from `.env` (64 in `.env.example`) | 0 turns the patched spill off |
| `MEM_FLOOR_GIB` | from `.env` (1.5 in `.env.example`) | Host-memory floor of the live memory guard |

## Commands

| Command | Effect |
|---|---|
| `./start-display-kv.sh --check` | Read-only DRM, card and compute status on both nodes. Safe with ranks running |
| `./start-display-kv.sh --build` | Prepare an updated image on both nodes, then prepare DRM, probe and start |
| `./start-display-kv.sh --build-only` | Build and copy the derived image. No DRM change, no probe, no start |
| `./start-display-kv.sh` | Use the saved image, prepare DRM, probe and start. Never builds or copies |
| `./start-display-kv.sh --restore-drm` | Restore the boot-default DRM module parameters on both nodes. No server start |
| `./stop.sh` | Stop both ranks and save their container logs |

An unreadable or empty DRM parameter file is reported as unknown, not as proof of
readiness. Some 580.x drivers do not show their parameter values correctly. The
allocation and integrity probes decide.

## Safety And Recovery

All DRM operations are in `scripts/display_drm.sh`, which the launcher streams to
the worker over SSH. No checkout, script, binary or configuration from another
model repo is needed. The helper finds the NVIDIA-owned cards and render nodes. It
does not assume `card0` or `card1`.

Preparation and restoration refuse to run in these cases:

- an active compute process
- an active NVIDIA scanout encoder
- an NVIDIA DRM mapping or open file descriptor
- an unknown connected-encoder state
- a module reference count above zero

The simple-framebuffer TTY is a different device. It is no reason to refuse an
idle NVIDIA reservation. The helper reloads only `nvidia_drm`, never the compute
driver. It does not write modprobe configuration and does not rebuild the
initramfs. A reboot restores the boot defaults.

After a failed cutover, the launcher stops only the rank containers of its own
image and tries to restore the DRM defaults on both nodes. It does not start an
ordinary-memory fallback by itself. If another GPU or DRM user appears,
restoration can refuse. Read the error, stop that workload yourself, then run
`--restore-drm`. The generated launcher does not force-remove a rank that
appeared at the same time.

To go back to the normal engine:

```bash
./stop.sh
./start-display-kv.sh --restore-drm
./start.sh
```

Do not run the normal launcher and the display launcher at the same time. The
display launcher lock serializes display launches only. Both variants use the same
rank names, API ports and model-cache paths.

## Cost And Capacity

The 1792 MiB display mapping on each rank holds a 2,109,440-position shared
KV/indexer pool. It is not a larger context for one request. Rings, weights,
snapshots, RoPE and workspaces stay in ordinary memory. The reservation does not
show in the normal VRAM figures of `nvidia-smi`.

`/health` and `/metrics` report capacity, free and used positions, the largest
free extent, kept prompts, snapshot bytes, and display against ordinary backing.
The TensorFold adapter of EngineMetrics reads these fields without a change.
Byte figures are per rank, not summed over the replicated ranks. See the
[implementation notes](../experiments/display-kv/README.md#cache-telemetry) for
the exact field meaning.

Two testers measured the feature on one pair of Spark nodes. The maintainers did
not repeat the runs. The numbers below are from the saved records
[`runs/2026-10-19/display-kv-tester-ab.md`](../runs/2026-10-19/display-kv-tester-ab.md)
and
[`runs/2026-10-20/display-kv-tester-kv-budget.md`](../runs/2026-10-20/display-kv-tester-kv-budget.md).
Credit: Sora39831 for the runs and hjasond for the early report.

| Measure | Stock lane | Display lane |
|---|---|---|
| Cold prefill, 32K / 128K / about 1M tokens | 2,124 / 1,923 / 1,299 tok/s | 1,680 / 1,534 / 1,092 tok/s |
| Change in prefill tok/s | | -20.9% / -20.2% / -15.9% |
| Decode, one stream | 51 to 62 tok/s | 48 to 59 tok/s |
| Prefix cache hit, 54K prompt, TTFT | 0.73 s | 0.46 s |
| Minimum host MemAvailable, 1M prefill | 1.96 GiB | 2.01 GiB |
| KV positions | 2,097,152 | 2,109,440 (+0.59%) |
| Simultaneous 1M-token streams | earlyoom kills the engine at 4 | 5 served, CUDA OOM (exit 75) at 6, with the hard cap on |

What this means:

- The display lane gives the same KV position budget (+0.59%). It does not give
  host-memory headroom: the minimum MemAvailable at 1M tokens is about the same.
  It moves the KV out of host RAM, and that raised the number of served streams
  from 4 to 5 in the overload test.
- The prefill gap (about 16% to 21%) is not clean. The display bundle uses a 512
  prefill chunk and the stock lane used 2048, so the test does not show how much
  of the gap is the chunk size and how much is the display allocation. The stock
  image of the test carried a local vision patch. The blocks ran one after the
  other, one run per size. A fair test must run both lanes with the same
  `PREFILL_CHUNK`, in alternating order.
- The SSD spill stayed at 0 entries in all saved runs. With a 2.1M-position pool
  and two streams nothing is evicted, so spill write, evict and restore were not
  tested on the final code in a saved run.
- Full-window and near-full-pool operation is not shown to be production-safe.

The author of the feature (0rand) reports additional live runs (two-node
integrity, 24K prefix reuse, four simultaneous 32K requests, a depth-65536
benchmark, SSD write, evict and restore). The repo has no saved record for these
runs, so this document gives no number from them.

## Disk Spill

The patched spill belongs to the display lane only (see the first table). It
stores kept prompts under `TF_DS_SPILL_DIR` (default `~/.cache/tensorfold`) up to
`SPILL_GIB` per rank. Both ranks write and restore on their own. Mapped rows move
through the bounded CUDA staging path of the tier (`_by_kernel`), never through
CPU reads or copy engines.

- The write path is write-through. The tier writes a kept prompt in the
  background from `TF_DS_SPILL_MIN` tokens (default 4096). A highwater setting does
  not exist for this lane; `/health` keeps the field for compatibility only.
- `TF_DS_SPILL_MIN`, `TF_DS_SPILL_MIN_FREE_GIB`, `TF_DS_SPILL_KEEP_BUILDS`,
  `TF_SPILL_WRITERS`, `TF_SPILL_DIRECT` and `TF_SPILL_FLUSH_S` tune the background
  writes. The display launcher passes them to the container when you set them.
- `/health["spill"]` and the `spill_*` Prometheus families show entries, bytes,
  write and load counters, restore latency, failures and free disk. A restore
  failure counts once in `load_failed`.
- A restart keeps only the prompts whose writes reached the disk.
- `docker stop` sends SIGTERM. Both ranks drain the queued writes before they exit.
  The wait is `TF_SPILL_FLUSH_S`, but never more than 55 seconds, below the
  `docker stop -t 60` limit. Rank 0 uses the SIGTERM handler of the pinned
  `serve()`. Rank 1 has its own handler in the follower loop, and it also drains
  when rank 0 closes the link first (`stop.sh` stops rank 0 first).
- The restore code checks the tensor descriptors of a spill file before it
  allocates: known dtype, non-negative shape, a span of exactly the shape's size,
  and spans without gap or overlap. A file that fails is dropped.
- The stored token ids use base64. A 1M-token prompt needs far less time to
  write than with a JSON list. Spill files from an older format are not read.
- The spill signature ignores the driver version, so nodes with different driver
  versions can share stored prompts. The model, the TensorFold sources, the
  CUDA and torch versions, the GPU name, the KV layout and the display size still
  decide whether a stored prompt fits.

## Live Memory Guard

The display image replaces the allocator cap that upstream freezes after warm-up
with a live budget. Both ranks refresh it before and after each prefill chunk and
each parallel decode round. The budget is the current PyTorch reserved bytes plus
the host `MemAvailable`, minus `MEM_FLOOR_GIB`, limited to the CUDA device total.
Host RAM that returns becomes usable without an engine restart. Near the floor,
the guard returns unused PyTorch cache blocks to the host. Without this, a
prefill peak stays reserved until earlyoom kills the process.

A positive floor below 1.5 GiB is raised to 1.5 GiB, because earlyoom sends
SIGTERM at 1 GiB, so a 0.5 GiB reserve cannot protect the engine. A floor of 0
turns the guard off. That is not safe on these unified-memory nodes. The guard is
no reservation against other processes or non-PyTorch allocations. A real memory
exhaustion can still break the distributed lane. The start log names the live
guard, and a broken step logs its last budget in bytes.

The hard allocator cap is OFF by default in the display launcher
(`MEM_ALLOCATOR_CAP=0`). With the cap off, the guard does cleanup and telemetry,
and the calculated ceiling is advisory. The start log says
`hard cap DISABLED; cleanup only`. Reason: a tester ran 6 simultaneous 1M-token
streams with the cap on, and a spike became a CUDA OOM and exit 75 (a rank lost).
A tester saw no lane failure with the cap off at 3 to 5 streams. That does not
prove that the cap is unneeded: no spike was large enough to make earlyoom act.
The default is not tested live after the change.

Set `MEM_ALLOCATOR_CAP=1` in `.env` to get the hard cap back. It can end in exit 75
near `MEM_FLOOR_GIB`. The display overlay implements this switch; the stock image
does not have it. The engine reads it at start and not while it runs. Keep
earlyoom on. `MEM_FLOOR_GIB=0` is a different setting: it turns off cleanup and
telemetry too.

Long-context attention indexing tiles its packed ordering keys (16 MiB per key
tile, or one full row if the row is larger). It builds them in place. The
original score order and the lower-index tie rule stay. The tile limit does not
include the input scores, the output, the index vector or the sort workspace. A
GPU regression case with 512 x 32768 scores and k=256 reduced the incremental
peak allocated scratch from 448.2 MiB to 35.2 MiB, with identical indices.

A saved image does not take source changes by itself. With both ranks stopped,
run `./start-display-kv.sh --build-only` to build and copy the new image, then
`./start-display-kv.sh` to start it. A regular start never rebuilds.

## Offline Tests

These launcher and helper tests do no live Docker, SSH, sudo or DRM operation:

```bash
PYTHONDONTWRITEBYTECODE=1 python3 tests/display_drm_test.py
PYTHONDONTWRITEBYTECODE=1 python3 tests/start_display_kv_test.py
```

The engine-patch and telemetry tests need a checkout of the pinned TensorFold
revision. They skip when it is absent. Most of them use fake tensors. The top-k
equivalence tests use CPU PyTorch when it is installed. The CUDA equivalence, graph
capture and peak-memory tests run only when CUDA is available. No model weights
are loaded:

```bash
TF_DISPLAY_SOURCE=/path/to/TensorFold PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/display-kv/tests -v
```
