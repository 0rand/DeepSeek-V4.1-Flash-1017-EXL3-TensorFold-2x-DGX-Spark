# Native tool-output validation (draft)

The display-KV recipe's existing source installer now validates the complete
DeepSeek-V4.1 **answer** before returning content or executable calls. This is a
CPU-tested protocol/schema hardening change, not evidence of actual-model tool
reliability or a GPU performance improvement. The previously recorded live
checks for the narrower unoffered-tool guard do not validate this revision.

## Contract and compatibility changes

- Exact, case-sensitive offered names, `tool_choice`, parallel policy and call
  limits are enforced. `parallel_tool_calls: null` means the existing default
  (`true`); explicit `false` still disallows multiple calls. No admission limiter
  is added, and independent requests retain existing concurrency semantics.
- Malformed, truncated, unavailable, schema-invalid and over-limit executable
  output rejects the **whole** answer, including a valid call preceding the
  failure. The permissive legacy parser is not used to repair or truncate it.
- Strict validation also applies when `tools` is omitted, null or empty: a model
  cannot turn an unoffered executable envelope into ordinary answer text. This
  means all strict DeepSeek chat answer content is buffered until completion,
  including no-tool answers. Reasoning remains separately streamed and is not
  treated as executable answer output. Other families keep their legacy path.
- JSON, Qwen function/parameter XML, GLM argument tags and both spaced and
  unspaced DeepSeek DSML wrappers are supported. Duplicate JSON/parameter keys,
  nonfinite numbers, unexpected envelope fields and unparsed residue reject.
- Explicit quotes, inline code, fenced code and Markdown blockquotes remain
  literal content. A malformed literal example cannot hide a later executable
  attempt outside its literal span. Protocol markers inside an argument string
  remain data. Unquoted markup is an attempted call, not a guaranteed example.
- Supplied IDs are preserved exactly, not normalized or replaced. Empty,
  non-string or duplicate IDs reject; an absent ID receives a new generated ID.

## Bounded schema subset

The dependency-free validator supports boolean schemas, local JSON-pointer
`$ref`, `type`, `enum`, `const`, `allOf`/`anyOf`/`oneOf`/`not`, `if`/`then`/`else`,
object properties/requirements/additional and pattern properties/property
counts/dependent requirements, arrays/items/prefix items/length/uniqueness/
contains counts, string length/pattern and numeric bounds/`multipleOf`.
`$defs` and `definitions` are supported. Standard metadata annotations are
ignored, including `format` (it is **not** asserted). Unsupported assertion
keywords or invalid assertion shapes fail closed, including in unused branches.
This is deliberately not a complete JSON Schema implementation.

Only an ordinary instance mismatch can be inverted by `not` or used to select
another logical branch. Invalid schemas, recursive-depth failures and exhausted
work budgets are terminal across every combinator. Each call shares a 10,000
step budget across structural validation, branches, references and equality
checks. Schema and argument JSON serializations are each limited to 262,144
characters and nesting depth is capped at 64.

Regexes are a deliberately restricted Python `re` subset: sequences of literal,
dot, character-class and simple escaped atoms; optional outer `^`/`$`; fixed
atom repeats up to 128 with a conservative repeat-product bound of 256; at most
one variable atom repeat, requiring **both** anchors. Patterns are limited to 256 characters,
input strings/keys to 8,192 characters and aggregate estimated regex work to
2,000,000 per call. Groups, alternation, lookaround, backreferences, flags,
lazy/possessive and compound repeats reject **before compilation**. Existing
clients using broader regex or schema features will need compatible schemas;
there is no silent permissive fallback.

## Terminal wire error

Non-streaming rejection is HTTP 500 with this typed object (the existing
`unavailable_tool_call` code is retained for compatibility):

```json
{
  "error": {
    "type": "model_output_error",
    "code": "unavailable_tool_call",
    "source": "tensorfold.tool_output_guard",
    "reason": "invalid_executable_tool_output",
    "rejection_id": "<id>",
    "message": "The model's executable tool output was rejected. Reason: invalid_executable_tool_output; source: tensorfold.tool_output_guard; diagnostic ID: <id>."
  }
}
```

The JSON response adds `x-should-retry: false` only for this native typed error;
ordinary backend 500s retain their existing retry behavior. This is a client
retry hint, not a guarantee that all clients honor it. The server itself does
not retry generation. For an already-open SSE stream, HTTP stays 200 and one
error frame is followed by `data: [DONE]`, without executable deltas, answer
content or a successful finish. A subsequent request can recover normally.
Reasoning already streamed is not retracted.

Diagnostics contain the rejection ID, source, code, generic reason and an answer
SHA-256 fingerprint, without raw arguments, answer or reasoning. The earlier
guard's detailed wrapper/name/span diagnostics are no longer emitted. Proxy
errors are separate; do not confuse this source with proxy-side validation.

## Reproduce CPU checks

Use an existing pristine TensorFold Git checkout at the installer's exact pin:
`07aa02585f64cfb04c60a96de6554c0d4708ed46`. Only Python's standard library and
Git are needed for these checks; no downloads or package installation occur.

```bash
python3 -B experiments/display-kv/tests/run_tool_output_cpu.py --source /path/to/TensorFold
# Or: TF_DISPLAY_SOURCE=/path/to/TensorFold python3 -B experiments/display-kv/tests/run_tool_output_cpu.py
```

The runner clones into temporary directories, stages `apply.py` and `overlay/`
using the Dockerfile COPY layout, and executes the **real installer CLI** in
normal and optimized (`-O`) Python. It verifies installed bytes against the
transforms/overlay; wrong HEAD, changed source fingerprints and reapplication
must refuse without source writes. Parser/schema and fake-engine loopback HTTP
regressions then import the actually installed source in both modes. The input
checkout is read-only and checked unchanged. Targeted transformed-source tests
can also be run with `TF_DISPLAY_SOURCE` and unittest discovery using
`-p 'test_tool_output*.py'` against the pristine checkout.

The original revision and pristine source fingerprints remain unchanged in
`apply.py`: they validate **input**, not installed output. Installed source is
modified and its Git diff and overlay hash must be retained for provenance;
HEAD alone must never be presented as proof of pristine installed bytes.
Docker already copies the overlay directory; no opaque patch bundle or private
source copy is required. No running containers are changed by source testing.

## Actual-model / GPU gates still required

This draft has not built an image or loaded a model, CUDA extension or GPU.
Before promotion, explicitly build/deploy a new source-tagged image and test
real JSON/SSE rejection and recovery, valid calls/IDs and tool-result round
trips, forced/none/no-tool/nullable-parallel requests, both DSML forms, literal
examples and concurrent streams. Check client retry handling with the actual
SDK/proxy and repeat the OpenCode agent case without enabling the denied tool.
Model probes that do not induce their requested output are inconclusive, not
passing rejection evidence. Check cancellation and reasoning latency on the
real serving path. This change makes no memory, RDMA, prebuilt-extension,
host-tuning, distributed-liveness or measured throughput claims.
