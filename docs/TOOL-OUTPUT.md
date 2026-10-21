# Opt-in tool-output guard

This experiment rejects a complete model tool call that the request did not offer.
It is a separate image. The ordinary image and launcher defaults do not change.

## Why

A model can emit a call to a tool that the request did not declare. The ordinary
parser can then execute that call. This guard validates the whole answer before
it releases executable call deltas. It returns a safe error instead.

## Build and select

Build the ordinary pinned image first. See [the main recipe](../README.md).
Then run this command from the recipe root:

```bash
docker build -t tensorfold-dsv41:07aa02585f64-tool-output experiments/tool-output
```

Stop the current server before you select another image.
Copy the new image to the worker. `docker/build.sh` cannot do this: it rebuilds the ordinary image first. Use this helper, which copies the tag that you built and checks that both nodes hold the same image ID:

```bash
experiments/tool-output/transfer.sh tensorfold-dsv41:07aa02585f64-tool-output
```

Set `IMAGE=tensorfold-dsv41:07aa02585f64-tool-output` in your local `.env` on the head.
Keep `BIND_HOST=127.0.0.1`. Start only one model server.
Run `./verify.sh`. Use the authentication settings in [CONNECT](CONNECT.md).
To return to the ordinary engine, stop the server and restore the original `IMAGE`.

This image has not had a live two-rank serving check. Do that check before production use.
No performance claim is made.

## Behaviour

- Ordinary prose streams as before, including when tools are offered.
- A possible opening tag at the end of a chunk waits for more text. Prose before it is released.
- A real call envelope is held until the whole answer is validated.
- The guard uses the engine's own parser rules and reads no Markdown. It takes the engine's functions in `tensorfold.cuda.reply_text`: the DSML rewrite (`dsml_to_tool_calls`), the call-block expression and the Qwen and GLM readers. It reads the tool name of each call block as the engine does, and rejects the answer if a name is not offered. The reply is then the engine's own `parse_tool_calls` result on the same text. The guard and the parser therefore cannot disagree about what is a call, for any tool name.
- Every call block counts as a call, wherever it is: in prose, in quotes, in inline code, in a quote line or in a code fence. An unoffered call gives the safe error. An offered call runs as a call.
- Known limit of a guard that fails closed: an example of a tool call in a code fence is a call. If the tool is not offered, the reply is rejected. If the tool is offered, the example runs as a call. A model that explains tool-call syntax to a user can trigger this.
- What the parser does not read as a call stays in the content, and nothing runs. Examples are a call whose JSON text holds a closing tag (the engine ends the block at that tag) and a complete call inside the data of another call. The guard does not reject these. One case is stricter than the parser: a flat JSON call with an empty nested `function` object has no name for the engine (not a call), but the guard reads the top-level `name`, so an unoffered one is rejected.
- While streaming, every call opener waits until the answer is complete, like a real call. The final result then decides.
- If the answer holds no closing call tag, the guard does not call the parser (the engine's expressions are slow when many openers have no closing tag, as in a looping model).
- Streaming scans each new part of the answer once. It does not scan the whole answer again for each chunk. If a new text does not extend the text of the last call, it scans from the start.
- The content of a guarded reply passes through the same `ToolCallPolicy.content()` step as before. With `parallel_tool_calls: false`, that step removes tool text that is not a complete call.
- An unoffered call returns one safe error. It does not expose tool arguments.
- A stream sends one error frame and one done frame. It sends no call delta and no finish reason.
- A non-stream request returns HTTP 422 with `model_output_error`. A client may retry a 5xx response, and this response cannot change on a retry.
- The next valid request is not affected.
- Generic `App` behavior is unchanged. Only the DeepSeek V4.1 family enables the guard.

## Switch

The guard is on in this image. To turn it off, set `STRICT_TOOL_OUTPUT=0` in `.env`. `./start.sh` then passes `TF_DS_STRICT_TOOL_OUTPUT=0` to both ranks, and the image keeps the original parser. Only `0` and `1` are valid. Another value stops `start.sh` before any command. With the ordinary image the setting does nothing, and `./start.sh` adds no argument when the setting is absent or `1`. A restart applies a change.

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
