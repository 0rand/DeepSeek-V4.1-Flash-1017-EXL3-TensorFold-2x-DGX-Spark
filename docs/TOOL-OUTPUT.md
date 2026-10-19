# Tool Output Hardening

This change is included in the display-KV image installer. Live JSON rejection,
SSE rejection, offered-tool recovery and default-thinking checks passed on
2026-10-19 with image `tensorfold-dsv41:display-kv-fe26bd75d49e`. The original
base image is not modified. An actual OpenCode-session retest is still pending.

## Behaviour

The DeepSeek-V4.1 app rejects a complete tool-call envelope when it names a tool
not offered by that request. This includes the reported case of `task` being
generated while only other tools are available in the selected OpenCode agent.
It never enables a denied tool, repairs arguments, substitutes another name, or
retries a rejected model response.

The effective offered set is used after `tool_choice` filtering. If a request
declares tools but selects `tool_choice: none`, a generated tool call is still
rejected. Requests without a tools field and other TensorFold families retain
their existing behaviour.

Recognized complete envelopes are JSON, Qwen function/parameter XML and GLM
argument tags inside `<tool_call>`, plus DeepSeek-V4.1's spaced DSML calls block.
Explicitly quoted, inline-code, fenced-code and Markdown blockquote examples
remain content rather than becoming executable calls. The same applies to tool
syntax appearing inside reasoning, which is separated before answer validation.
Unquoted protocol markup is considered an attempted call; the server cannot
reliably infer that an unquoted call was merely an example.

Malformed and truncated envelopes are outside this first patch's error
classification. They are not mislabeled as unavailable tools, and can still
surface through the existing permissive parser. Argument schema validation and
model-level reliability are not fixed by this change. Existing parser JSON
normalization remains in place; argument values are not fuzzily repaired or
rewritten by the new guard.

## Streaming

For strict DeepSeek requests with tools, answer content and all executable tool
deltas are buffered until the complete answer passes validation. Reasoning can
still stream separately. This prevents an offered call from being emitted before
a later unoffered call invalidates the response. It also means plain answer text
on tool-enabled requests arrives at completion rather than token by token.

The validation state is local to each request. Rejecting a response happens
after model generation and does not break the distributed inference lane. A
subsequent request can run normally.

Non-streaming failures return HTTP 500, not a client-input HTTP 400:

```json
{
  "error": {
    "type": "model_output_error",
    "code": "unavailable_tool_call",
    "message": "The model attempted to call a tool not offered by this request."
  }
}
```

For an established SSE stream the HTTP status remains 200. The server emits
one error frame with the same error object and then `data: [DONE]`, without a
normal successful finish or executable tool deltas. The error does not include
the generated arguments or raw envelope. Logs report only the rejection code.

There is no server-side retry or fallback on this error. A client may have its
own retry policy, particularly for HTTP 500; clients should treat this explicit
error code as a failed model response rather than repeatedly submitting it.

## Activation And Tests

An explicit `./start-display-kv.sh --build` includes changes to the guard and
selects an updated source-tagged image. A plain launch reuses the saved image,
without rebuilding or copying. The guard applies to this image's DeepSeek app
whether or not its display allocation is enabled. Existing running containers
are immutable and are not changed by staging or testing this patch.

With a pristine pinned TensorFold checkout:

```bash
TF_DISPLAY_SOURCE=/path/to/TensorFold PYTHONDONTWRITEBYTECODE=1 python3 -m unittest discover -s experiments/display-kv/tests -v
```

Classifier tests cover offered/unoffered JSON, XML, GLM and DSML, literals,
nested argument text and global call limits. Fake-engine HTTP tests cover
streaming and non-streaming errors, mixed-call atomicity, concurrent requests,
recovery and one generation per rejected request. They run an ephemeral local
HTTP server, not the model service, and do not load CUDA or real model weights.

After deployment, test in an OpenCode agent where `task` remains unavailable.
Do not enable the tool to make the test pass. Also verify normal tool calls,
tool-result round trips, plain answers and literal syntax examples. Host-memory
pressure and SSD spill support are separate issues; this guard changes neither.

`tests/live_tool_hardening.py` performs model-backed API probes without executing
or retrying generated calls. Core rejection checks successfully induced an
unoffered `task` call. Additional mixed-call and literal-copy probes did not
induce their requested fixtures in the initial live run: the model selected
only an offered call or refused to reproduce the example. Those cases remain
inconclusive live, rather than evidence of a parser failure; they pass the
deterministic fake-engine HTTP tests.
