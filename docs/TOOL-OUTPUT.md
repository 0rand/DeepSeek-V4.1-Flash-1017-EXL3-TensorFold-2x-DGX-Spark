# Tool Output Hardening

This guard is part of the display-KV image (`experiments/display-kv/`). It
rejects a model answer that calls a tool the request did not offer. It does not
change the base image.

## On and off

| Path | State of the guard |
|---|---|
| `./start-display-kv.sh` | ON by default. The bundle sets `TF_DS_STRICT_TOOL_OUTPUT=1`. |
| `./start-display-kv.sh` with `STRICT_TOOL_OUTPUT=0` in `.env` | OFF. The bundle sets `TF_DS_STRICT_TOOL_OUTPUT=0`. |
| `./start.sh` (default launcher, stock image) | The guard code is not in the image. No change. |
| The display image started with another launcher | OFF, unless you set `TF_DS_STRICT_TOOL_OUTPUT=1` yourself. |

The switch is independent of the display-KV allocation. `STRICT_TOOL_OUTPUT` is
the only `.env` name. `./start-display-kv.sh` checks it and refuses any value
other than `0` or `1`. To turn the guard off:

```bash
# in .env
STRICT_TOOL_OUTPUT=0
```

Then start `./start-display-kv.sh` again. A running container does not change.

## Behaviour

The DeepSeek-V4.1 app rejects a complete tool-call envelope when it names a tool
that the request did not offer. One reported case: the model generates `task`
while the selected agent offers other tools only. The guard never enables a
denied tool. It never repairs arguments, never replaces a name and never retries
a rejected answer.

The guard uses the offered set after `tool_choice` filtering. If a request
declares tools and sets `tool_choice: none`, a generated tool call is still
rejected. A request without a tools field keeps the old behavior. Other
TensorFold model families keep the old behavior.

The guard reads these envelopes:

- JSON inside `<tool_call>`
- Qwen function/parameter XML inside `<tool_call>`
- GLM argument tags inside `<tool_call>`
- the DeepSeek-V4.1 spaced DSML calls block

An example stays content and does not become a call when it is quoted, in inline
code, in a fenced code block or in a Markdown blockquote. The same applies to
tool syntax inside reasoning, which the server splits off before it checks the
answer. Unquoted protocol markup counts as an attempted call. The server cannot
know that an unquoted call is only an example.

A real call can contain an unclosed code fence or a stray backtick in its
arguments. That text does not hide a later call. The classifier takes each
accepted call out of the text before it looks for literal spans in the rest.

A malformed or truncated envelope has no error class in this change. The guard
does not label it "unavailable tool", and the existing permissive parser can
still pass it. The guard does not check argument schemas. It does not make the
model more reliable.

## Streaming

For a request with tools, the server sends answer text in two parts:

1. Text before the first possible tool-call opening tag is sent at once. The
   server also holds back a trailing partial opening tag.
2. The rest of the answer waits until the full answer passes validation. All
   tool-call deltas arrive at the end, after the check.

This keeps an offered call from reaching the client before a later unoffered
call makes the answer invalid. Reasoning streams as before. A tool-enabled
answer that starts with plain text shows that text during decode. A tool-call
block arrives when decode ends.

The validation state belongs to one request. A rejection happens after the model
generates. It does not break the distributed inference lane, and the next
request runs as normal.

## Errors

In non-stream mode the server returns HTTP 422, not 500. A client may retry a 5xx
response. It should not retry a response that cannot change.

```json
{
  "error": {
    "type": "model_output_error",
    "code": "unavailable_tool_call",
    "message": "The model attempted to call a tool not offered by this request. Reason: unavailable_tool; source: tensorfold.tool_output_guard; diagnostic ID: <id>.",
    "source": "tensorfold.tool_output_guard",
    "reason": "unavailable_tool",
    "rejection_id": "<id>"
  }
}
```

In an SSE stream the HTTP status stays 200. The server sends one error frame
with the same error object, then `data: [DONE]`. It sends no normal finish and
no tool-call delta. Text sent before the first possible tag (see Streaming)
stays sent. The error does not contain the generated arguments or the raw
envelope. The diagnostic ID is also in the message text, so a client that shows
only the message can still report it.

The backend log has one bounded JSON line for each rejection. It contains:

- the same `rejection_id`
- the rejected name (96 characters at most)
- the wrapper format
- the declared and effective tool names (64 names each at most)
- the tool-choice mode
- the offsets of the matched span
- a SHA-256 fingerprint of the answer

It never contains arguments, matched text, reasoning or conversation bodies. A
declared tool that `tool_choice` excludes has `reason=disallowed_by_tool_choice`.
A name that the request does not contain has `reason=unavailable_tool`. Both use
the code `unavailable_tool_call`, and both fail closed.

Do not confuse these errors (`source=tensorfold.tool_output_guard`) with the
`invalid_raw_tool_call` error of the MediaLLMProxy raw-call guard. The proxy and
the backend write separate logs.

The server does not retry or fall back after this error. A client must treat the
code as a failed model response and must not send the same request again in a
loop.

## Tests

Tests need a checkout of the pinned TensorFold revision. They skip when it is
absent.

```bash
TF_DISPLAY_SOURCE=/path/to/TensorFold PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/display-kv/tests -v
```

- The classifier tests cover offered and unoffered JSON, XML, GLM and DSML calls,
  literals, nested argument text, open fences inside real calls and global call
  limits.
- The fake-engine HTTP tests cover stream and non-stream errors, text before a
  call, mixed-call atomicity, concurrent requests, recovery, one generation per
  rejected request and the off switch. They start a local HTTP server. They do
  not use CUDA or model weights.

`tests/live_tool_hardening.py` sends model-backed probes to a running server. It
does not run or retry the generated calls. The saved live files are in
`runs/2026-10-19/`. In those files the core rejection checks induced an unoffered
`task` call. Two other probes (mixed-call atomicity, a literal copy) did not
induce their fixtures: the model chose an offered call only, or it refused to
copy the example. Those two results are inconclusive, not a parser failure. The
deterministic HTTP tests cover the same cases. The saved files come from an
earlier state of the guard (HTTP 500 for an unoffered call). They do not record
the image or the settings. No live run exists yet for the current code (streaming
of text before a call, HTTP 422).

After you deploy, test in an agent where `task` is not available. Do not enable
the tool to make the test pass. Also check normal tool calls, tool-result round
trips, plain answers and literal syntax examples. This guard does not change
host-memory behavior or SSD spill.
