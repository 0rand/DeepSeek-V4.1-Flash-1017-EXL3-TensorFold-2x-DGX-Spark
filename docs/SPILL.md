# Display-backed KV with SSD spill

This is an opt-in dependent image. It adds the display-safe SSD spill tier to
the display-KV experiment. The ordinary `start.sh`, `.env` and base image stay
unchanged. Full two-node hardware validation is still pending.

## Dependency

This image needs the display-KV feature. The composition installer validates the
union of pristine source hashes once, applies the display transform first, then
the spill transform, checks every generated Python file, and only then writes.
Do not run the two standalone installers in sequence on modified source.

## Build and use

Build from the repository root so both feature folders are in the context:

```bash
docker build -f experiments/display-spill/Dockerfile \
  --build-arg BASE_IMAGE=tensorfold-dsv41:07aa02585f64 \
  -t tensorfold-dsv41:display-spill .
```

Stop both ranks before a cutover:

```bash
./stop.sh
./start-display-spill.sh --build-only
./start-display-spill.sh
./verify.sh
```

`--build-only` builds and copies the image without DRM changes, probes or a
server. The launcher refuses active ranks, prepares DRM on both nodes, probes the
full display reservation, and starts a private bundle with `--spill`. It keeps
the `SPILL_GIB` value from `.env`; set it to `0` to disable disk writes.

To return to the ordinary engine after all GPU and DRM clients stop:

```bash
./stop.sh
./start-display-spill.sh --restore-drm
./start.sh
```

Never stop GDM. Never use `docker exec` in a serving rank. Run only one model
server at a time. Keep `BIND_HOST=127.0.0.1` unless network access and an API key
are explicitly configured.

## Spill settings

| Setting | Default | Effect |
|---|---|---|
| `SPILL_GIB` | `64` | Maximum stored prompt bytes per rank. `0` disables writes. |
| `TF_DS_SPILL_HIGHWATER` | `0.70` | Occupancy past which eviction candidates are written early. |
| `TF_DS_SPILL_MIN` | `4096` | Shortest prompt written to disk. |
| `TF_DS_SPILL_MIN_FREE_GIB` | `50` | No write leaves less disk free on any rank. |
| `TF_SPILL_WRITERS` | `2` | Writer threads per rank. |
| `TF_SPILL_DIRECT` | `1` | Use O_DIRECT where the file system supports it. |
| `TF_SPILL_FLUSH_S` | `60` | Shutdown drain budget. The app uses at most 55 seconds. |

The store writes safetensors files with a CRC-32 per 8 MiB, fsyncs, and renames
from `.partial`. A corrupt or short file is dropped. Every rank agrees on the
index and on each read. A read is used only when it worked on every rank.

## Validation limits

Offline tests use CPU torch and temporary directories. They do not prove a real
two-rank write, eviction, restore and correct reply. That live test is a merge
gate. Also test restart restore, a pending write during SIGTERM, one-rank
read/write failure, corrupt entries, low disk space, disabled mode, and no
collective hang. Record counters that prove disk writes and loads happened.
No performance claim is made.

## Installer protocol

`apply.py` exposes `REVISION`, `FILES`, `transform(relative, text)`, `overlays()`
and `install(checkout)`. `compose.py` combines it with the display installer.
`FILES` contains only the pristine files this feature changes. The spill module
is a new overlay; an existing file is rejected.

For a disposable source checkout:

```bash
python3 experiments/display-spill/compose.py "$CHECKOUT" \
  experiments/display-kv/apply.py experiments/display-spill/apply.py
```

The composer checks the exact revision, all pristine hashes, overlay absence and
all generated syntax before the first write. It does not provide rollback for an
I/O failure during writes. Use a disposable checkout.
