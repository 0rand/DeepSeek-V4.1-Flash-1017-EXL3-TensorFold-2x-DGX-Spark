# Opt-in tool-output guard

This experiment rejects a complete model tool call that the request did not offer.
It is a separate image. The ordinary image and launcher defaults do not change.

## Why

A model can emit a call to a tool that the request did not declare. The ordinary
parser can then execute that call. This guard validates the whole answer before
it releases executable call deltas. It returns a safe error instead.

## Build and select

Build the ordinary pinned image first. See [the main recipe](../../README.md).
Then run this command from the recipe root:

```bash
docker build -t tensorfold-dsv41:07aa02585f64-tool-output experiments/tool-output
```

Stop the current server before you select another image.
Copy the new image to the worker with the normal image transfer procedure.
Set `IMAGE=tensorfold-dsv41:07aa02585f64-tool-output` in your local `.env` on the head.
Keep `BIND_HOST=127.0.0.1`. Start only one model server.
Run `./verify.sh`. Use the authentication settings in [CONNECT](../../docs/CONNECT.md).
To return to the ordinary engine, stop the server and restore the original `IMAGE`.

This image has not had a live two-rank serving check. Do that check before production use.
No performance claim is made.

## Behaviour

- Ordinary prose streams as before, including when tools are offered.
- A possible opening tag at the end of a chunk waits for more text. Prose before it is released.
- A real call envelope is held until the whole answer is validated.
- Fenced, quoted and inline-code examples stay literal. Argument bodies are data, not Markdown.
- An unoffered call returns one safe error. It does not expose tool arguments.
- A stream sends one error frame and one done frame. It sends no call delta and no finish reason.
- A non-stream request returns HTTP 500 with `model_output_error`.
- The next valid request is not affected.
- Generic `App` behavior is unchanged. Only the DeepSeek V4.1 family enables the guard.

The guard is a tool-markup check, not a full Markdown renderer. A model that writes
unusual formatting can still be rejected or held. Review the diagnostics before use.

## Installer protocol

`apply.py` exposes `REVISION`, `FILES`, `transform(relative, text)`, `overlays()`
and `install(checkout)`. `FILES` contains only the three pristine files that this
experiment changes. Overlay keys are relative target paths.

For a separate disposable source checkout, run:

```bash
python3 experiments/tool-output/apply.py "$TF_TOOL_SOURCE"
```

The installer checks the exact git revision and all pristine SHA-256 hashes.
It rejects an existing overlay. It checks all generated Python with `ast.parse`
before the first write. It rejects repeated transformation and changed anchors.
It does not provide rollback for an I/O failure during writes. Use a disposable checkout.

An external composer can validate pristine hashes for all selected experiments,
apply their transforms in order in memory, merge non-conflicting overlays, check
all generated syntax, and then write. Do not call standalone installers in sequence
on modified source. This PR does not add a shared composition framework.

## Offline tests

```bash
tests/run_offline.sh
TF_TOOL_SOURCE="$CHECKOUT" python3 -m unittest discover -s experiments/tool-output/tests -v
```

`CHECKOUT` must refer to explicit pristine pinned source. Tests only read it.
Without `TF_TOOL_SOURCE`, source integration tests skip. Other tests use the
Python standard library, loopback HTTP and fake engines. They need no TensorFold
install, GPU or model server.
