# Native tool-output validation (draft)

The display-KV recipe's existing source installer validates the complete
DeepSeek-V4.1 **answer** before returning executable calls. Inert prose before
the first possible protocol marker can stream early. This is a
protocol/schema hardening change, not evidence of actual-model tool reliability
or a GPU performance improvement. The earlier image
`tensorfold-dsv41:display-kv-58e319cb2866` passed `./verify.sh` 5/5 and the
opt-in parser probe with 12 PASS, 2 INCONCLUSIVE and 0 FAIL. The two
inconclusive cases are fixture-induction limits, not parser passes. See the
[run record](../runs/2026-10-21/parser-guardrails/README.md).
The later [proxy audit](../runs/2026-10-21/parser-proxy-audit/README.md)
recorded image `tensorfold-dsv41:display-kv-0dc9c3fefb06` with the same guard
hash. These records describe past observations, not a permanent deployment.
Inspect the current local rank without entering its container:

```bash
docker inspect --format '{{.Config.Image}} {{.Image}} {{.State.StartedAt}}' tfv41-r0
```

Use the same command for `tfv41-r1` on the worker. The
[proxy mitigation staging record](../runs/2026-10-21/proxy-mitigation-staging/README.md)
separates source validation from production approval.

## Contract and compatibility changes

- Exact, case-sensitive offered names, `tool_choice`, parallel policy and call
  limits are enforced. `parallel_tool_calls: null` means the existing default
  (`true`); explicit `false` still disallows multiple calls. No admission limiter
  is added, and independent requests retain existing concurrency semantics.
- Malformed, truncated, unavailable, schema-invalid and over-limit executable
  output rejects **all executable calls**, including a valid call preceding the
  failure. Already-streamed inert prose is not retracted. The permissive legacy
  parser is not used to repair or truncate output.
- Strict validation also applies when `tools` is omitted, null or empty: a model
  cannot turn an unoffered executable envelope into ordinary answer text. This
  does not prevent ordinary no-tool answers from streaming. Reasoning remains
  separately streamed and is not treated as executable answer output. Other
  families keep their legacy path. The engine may put the last reasoning token
  and the first answer token in one delta; clients must route by field, and the
  live probe treats that shared frame as an acceptable transition, not an
  interleaving.
- JSON, Qwen function/parameter XML, GLM argument tags and both spaced and
  unspaced DeepSeek DSML wrappers are supported. Duplicate JSON/parameter keys,
  nonfinite numbers, unexpected envelope fields and unparsed residue reject.
- Explicit quotes, inline code, fenced code and lines with an explicit leading
  Markdown blockquote marker remain literal content (including prose before a
  marker on that quoted line). Delimiters on explicitly quoted lines cannot
  open or close inline state in unquoted prose. Inline spans cannot cross a
  quoted-line, blank-paragraph or block-fence boundary; multiline spans within
  uninterrupted prose remain supported. Lazy quote continuation is not inferred. A
  malformed literal example cannot hide a later executable
  attempt outside its literal span. Protocol markers inside an argument string
  remain data. Markdown scanning resumes only after a whole accepted protocol
  envelope: backticks, fence runs and quotes in its arguments cannot start
  literal state around a later call. True prose fences, including unclosed
  fences, still protect their examples. Unquoted markup is an attempted call,
  not a guaranteed example.
- Supplied IDs are preserved exactly, not normalized or replaced. Empty,
  non-string or duplicate IDs reject; an absent ID receives a new generated ID.

## Conservative streaming

With `TOOL_OUTPUT_STREAMING=1` (the display-KV default), no-tools and tool-enabled
prose streams until the first raw protocol candidate. A candidate inside a
literal quote or code fence also starts buffering. The remainder of the answer,
including every executable call, waits for the single final strict parse.
There are no partial argument deltas, early call atoms or post-envelope prose
resumption in this phase. All request policies retain whole-answer executable
atomicity. The non-streaming endpoint still returns nothing before validation.

Incomplete marker spellings are held across arbitrarily split deltas. The guard
also covers namespace-prefixed and unsupported protocol variants recognized by
its rejection scanner. A possible prefix longer than 256 characters freezes the
remainder instead of repeatedly rescanning an unbounded namespace. This is only
a latency fallback: it neither rejects nor truncates a long literal or answer.
Whitespace is preserved. A decoder rewrite or a final result inconsistent with
published prose is a terminal `stream_prefix_changed` model-output error.

This conservative design is intentional. A matching quote several lines later,
or a backtick appended to a provisional closing run, can change whether an
earlier envelope is literal. A closing tag inside an unfinished JSON string
can also look like an invalid complete call. Repeatedly parsing prefixes is
therefore not a safe early-release or early-rejection algorithm. Reproductions
and future full-hybrid prerequisites are in
[the work plan, section 12](../streaming-tool-parser-fix-plan.md#12-continuation-on-2026-10-21-conservative-prose-projection).

To restore whole-answer **content buffering**, set `TOOL_OUTPUT_STREAMING=0` in
`.env`, then run `./stop.sh && ./start-display-kv.sh`. The experimental launcher
validates the value and forwards `TF_TOOL_OUTPUT_STREAMING` to both ranks.
Strict parsing, schema validation and terminal errors stay enabled. This switch
does not change the ordinary image or require another image build. Head-node
sudo authorization is required for the launcher's DRM preparation.

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
characters. Schema structure and recursive schema-evaluation depth are capped
at 64; this is **not** a cap on full argument-instance nesting. Unconstrained
subtrees (for example, additional properties allowed by `true` or the default)
are not recursively evaluated and may exceed 64 levels, subject to serialized
size and Python's JSON parser/serializer recursion limits.

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
error frame is followed by `data: [DONE]`, without executable deltas, buffered
envelope content or a successful finish. A subsequent request can recover
normally. Reasoning and inert prose already streamed are not retracted.
An open SSE stream cannot add an HTTP retry header after its headers were sent.

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

For the 2026-10-21 continuation's CPU checks, old-image live baseline and the
new-image live results, see [the run record](../runs/2026-10-21/parser-guardrails/README.md).
The opt-in `tests/live_tool_streaming.py` captures per-frame timings, explicit
thinking modes, tool-result round trips, rejection/recovery and literal probes.
It runs only against an HTTP loopback IP, blocks redirects and proxies, never
executes tools and never retries a probe. Its JSON output contains the raw
synthetic requests and model replies. Inconclusive cases are not counted as
proof of parser conformance.
