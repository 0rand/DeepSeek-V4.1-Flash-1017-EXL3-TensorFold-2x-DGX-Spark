# Display-backed KV experiment

This is an opt-in display-only variant. The ordinary `start.sh`, `.env` and
base image stay unchanged. Full two-node hardware validation is still pending.
Do not use the offline test results as proof of safe serving or better performance.

## Scope and risks

The installer changes only the pinned TensorFold `model.py`, `engine.py` and
`multi.py`, and adds `display_kv.py`. It checks the source revision, file hashes,
replacement anchors and generated Python before it writes files. It does not
install tool-output guards, memory-policy changes, metrics or a replacement spill tier.

The private launch bundle uses native packed KV, a 1792 MiB display reservation
per node, prefill chunk 512 and `SPILL_GIB=0`. Spill is disabled because this
stage does not include mapped-storage-safe spill. Ordinary launch defaults do
not change. The display pool and each request's context limit are separate.

Display mapping and runtime DRM changes can fail or affect the local display.
Keep both nodes idle before a cutover. Never stop GDM or another display manager.
Never use `docker exec` in serving `tfv41-r0` or `tfv41-r1` containers. Run only
one model server at a time. Keep `BIND_HOST=127.0.0.1` unless network access and
an API key are explicitly configured. Do not change DRM while any GPU or DRM
client is active. The launcher refuses active ranks and incompatible clients.

## Opt-in use

First complete the ordinary recipe setup, including network and host tuning.
Use the head node and the repository root. Inspect state without a cutover:

```bash
./start-display-kv.sh --check
```

When the current work has finished, stop both ranks deliberately:

```bash
./stop.sh
./start-display-kv.sh --build-only
./start-display-kv.sh
./verify.sh
```

`--build-only` builds and copies the derived image but does not change DRM or
start a server. `--build` builds, copies and starts in one command. A later
launch uses the saved image and checks its ID on both nodes. The head can ask
for sudo. The worker needs passwordless sudo for runtime DRM preparation.
The launcher probes the full reservation on both nodes before it starts the
private bundle. It does not stop a running server for you.

Private bundles and logs are under `experiments/display-kv/bundle-launch-*`.
They contain a configuration snapshot. Do not publish them or their secrets.
If startup fails, the launcher attempts to restore DRM defaults. It does not
start an ordinary-memory fallback. Inspect `--check` if restore fails.

To return to the ordinary engine after all GPU and DRM clients have stopped:

```bash
./stop.sh
./start-display-kv.sh --restore-drm
./start.sh
```

## Validation status

Offline tests use temporary fixtures and mocks. Optional source tests need an
explicit `TF_DISPLAY_SOURCE` checkout. See [tests](../tests/README.md).
Full hardware validation of this split is pending: two-node mapping and probe
success, a real reply, `verify.sh` PASS on all five checks, and an ordinary
launch regression check. A health response alone is not sufficient.
No new capacity, TTFT, decode tok/s or prefill performance claim is made.

See [implementation notes](../experiments/display-kv/README.md) and
[attribution](../experiments/display-kv/NOTICE.md).
