# Parser guardrails continuation, 2026-10-21

## Status and scope

The strict foundation and conservative prose projector were integrated in source
and, after user-assisted head authorization, deployed at 07:08 UTC as
`tensorfold-dsv41:display-kv-58e319cb2866` (image ID
`sha256:3a791c6a6f9d64e898ed29ec3a09de80c93c9f7c92db4e820ca0d207bbb2f278`) on
both nodes. Live validation passed: `./verify.sh` 5/5 and the opt-in parser probe
12 PASS, 2 INCONCLUSIVE, 0 FAIL. No live rank was entered with `docker exec`,
and no second model server was started.

This work does not upgrade the engine, change the user's `.env`, alter DRM,
change memory/spill behavior or claim a performance improvement.

## Changes and upstream review

- Integrated local strict-validator commits `c65e56f` and `d719bca` through merge
  `710eed1`. They replace permissive repair/truncation with exact offered-name,
  schema, envelope, ID and request-policy validation.
- Added `ToolOutputGuard.stream`: stream inert prose before the first raw
  protocol candidate, then buffer the remainder and all calls until the one
  authoritative final parse. Literal candidates also buffer. Decoder-prefix
  disagreement is terminal. A possible prefix over 256 characters freezes the
  remainder rather than repeatedly scanning a long namespace.
- Added `TOOL_OUTPUT_STREAMING=0` content-buffering rollback to the experimental
  launcher. Strict parsing remains enabled. The ordinary image/launcher and
  pinned input revision `07aa02585f64cfb04c60a96de6554c0d4708ed46` are unchanged.
- Upstream recipe `1b3066d` / PR #35 pins engine `730d41abd8f4` and defaults to
  dropping unoffered calls. Its experiments remain pinned to the older engine.
  No blanket merge or engine-pin update was made. Upstream PR #32 at `879b19d`
  informed the conservative raw-opener projection, but still uses the permissive
  engine parser and is not the local strict branch.
- The full early-atom proposal is deferred. Prefix replays prove later quotes,
  multiline inline spans, extended backtick runs and JSON string-internal closing
  tags can invalidate provisional classification. Implicit call IDs also change
  across parses. See [work plan section 12](../../../streaming-tool-parser-fix-plan.md#12-continuation-on-2026-10-21-conservative-prose-projection).

## CPU validation

Passed:

| Gate | Result |
|---|---|
| Real staged installer, parser/schema/projection/fake HTTP, normal Python | 70 tests, PASS |
| Same installed-source suite under `python -O` | 70 tests, PASS |
| Wrong pin, drift and reapplication refusal | PASS, no source writes |
| Input pristine checkout unchanged | PASS |
| Live-probe assessment unit tests, normal / `-O` | 8 / 8 PASS each |
| Experimental bundle generation, no service actions | 1 PASS |
| Root display launcher fixtures | 19 PASS |
| DRM helper fixtures, no real DRM operations | 13 PASS |
| Shell syntax and `git diff --check` | PASS |

The installed-source tests use the actual public `apply.py` plus `overlay/`
staging layout. They import neither Torch nor CUDA. Barrier-based HTTP tests
observe content while fake generation is blocked, with and without tools.
Replays cover character and two-chunk splits, fused DSML/call boundaries,
Unicode and long namespace prefixes, literal reclassification, late rejection,
request isolation, cancellation, decoder rewrites and rollback.
The full installed-source run is saved in [cpu-installed.log](cpu-installed.log).

Reproduce:

```bash
python3 -B experiments/display-kv/tests/run_tool_output_cpu.py --source /path/to/pristine/TensorFold
python3 -B tests/live_tool_streaming_test.py -v
python3 -B -O tests/live_tool_streaming_test.py -v
```

## Old-image model-backed baseline

[baseline.json](baseline.json) contains the synthetic requests, per-frame
transcripts/timing, response headers and assessments captured at approximately
06:52 UTC. No generated tool was executed, and each probe made one generation
attempt without retry. A synthetic tool result was provided for the round trip.

**5 PASS, 7 FAIL against the proposed new contract, 2 INCONCLUSIVE.** These are
old-image observations, not validation of the newly staged strict code.

| Probe | Old-image observation |
|---|---|
| Prose, no tools | 18 content chunks before terminal, PASS |
| Prose with tools | One final content chunk, FAIL streaming goal |
| Explicit thinking off, tools enabled | No reasoning, one final content chunk, FAIL streaming goal |
| Explicit thinking on, tools enabled | 14 reasoning chunks separated from one final content chunk, FAIL prose-streaming goal |
| Valid offered call | Correct `get_status(service="tensorfold")`, PASS |
| Tool-result round trip | Synthetic result reflected correctly, PASS |
| Unoffered JSON / SSE | Rejected with existing typed code, but missing the new source diagnostics, 2 FAIL |
| Malformed complete envelope | Exact invalid fixture exposed as prose, FAIL |
| No-tools executable envelope | Exact unoffered fixture exposed as prose, FAIL |
| Truncated requested fixture | Returned an offered structured call, INCONCLUSIVE |
| Ordinary literal and fenced envelope | Exact literals preserved, 2 PASS |
| Inline-code envelope requested | Returned an offered structured call, INCONCLUSIVE |

For the two inconclusive cases, the model may have completed or changed the
requested fixture before emitting it. Without raw generated-token decoding,
these results do **not** prove that the parser repaired a truncated call or
executed a literal. Such ownership is covered by deterministic installed-source
regressions. Future exact actual-model classifier checks need raw token capture.

The baseline's general inference gate also passed: [verify-baseline.log](verify-baseline.log)
contains all five real `./verify.sh` checks. This proves current two-node
inference, not the new image's correctness.

An earlier scratch-only run helped correct two probe-assessment mistakes before
this durable baseline was captured: established SSE cannot add a late HTTP retry
header, and an offered call on a requested malformed/literal fixture is not a
proven parser failure. Offline assessment tests lock in both distinctions.

## Deployed image provenance

The launcher's fingerprint is the SHA-256 of the base image ID followed by
`sha256sum` lines for the **absolute** `$ROOT/experiments/display-kv/{Dockerfile,
.dockerignore,apply.py,probe.py,overlay/*.py}` paths. For base image
`sha256:2aede715ee121bef05c4c92894e143fe0979b4c2f7730b013d1ddbf43793da1c` and the
committed source, it is
`58e319cb2866cd0cc843b8bfdd59b835520cd9a26899250a3a50f3ea605ced9e`, so the variant
tag is `tensorfold-dsv41:display-kv-58e319cb2866`. Hashes of the image's
`apply.py` and `overlay/tool_output.py` equal the committed files
(`1ba8020b…` and `3ae12132…`), and the same image ID is present on both nodes.
Recomputing with relative paths yields a different value; use the launcher's own
computation. The earlier record's `1fe0a7c3556a` and `df444454947e` values came
from such relative-path recomputations and are withdrawn.

| Source | SHA-256 |
|---|---|
| `experiments/display-kv/apply.py` | `1ba8020b5984d131dcb63fe56494a931edb889f0363d2fa5be00376d01e06785` |
| `experiments/display-kv/overlay/tool_output.py` | `3ae1213289babe2f64bdc7a8bb8a97af65e4450746d5ee086b3169ec7f13fef2` |
| `tests/live_tool_streaming.py` | `bad8ba8db32aabd41023f46645ee4d5ca37d8e8aced18144e624b31baace3cab` |
| `tests/live_tool_streaming_test.py` | `8f151acb80c9d0e5c989579e4e915fa5e96d4e015a923544da91e70f64de81b8` |

The previous `.display-kv-image` identity was saved before the cutover. The old
stack was left running until the user could authorize the head's DRM preparation;
head `sudo -n true` still required a password while the worker's noninteractive
sudo worked. Keep that rule: do not stop a healthy stack unless the head's DRM
preparation can be completed.

Before the cutover, at 07:07 UTC: both old ranks were up on
`display-kv-40a46c96d4bd`, `./start-display-kv.sh --check` reported both cards
with inactive outputs and the live compute reference, worker SSH and
noninteractive sudo succeeded, both nodes' `/var/lib/docker` had ~1.1 TB free,
and head `sudo -n true` still required a password. The user then ran, in the
head-node terminal:

```bash
./stop.sh && ./start-display-kv.sh --build
```

which produced the image above. The remaining acceptance steps were run from
this session:

```bash
./verify.sh
python3 -B tests/live_tool_streaming.py --base-url http://127.0.0.1:8100 --output runs/2026-10-21/parser-guardrails/after.json
```

The live probe script is loopback-only, blocks redirects/proxies, runs no tools
and has bounded generation. Exit 1 means failures, 2 means only inconclusive
cases remain. Preserve those distinctions. Also validate actual SDK/proxy retry
handling and the original agent scenario without enabling an unavailable tool.

## New-image model-backed validation (07:14-07:15 UTC)

[verify-after.log](verify-after.log): all five real `./verify.sh` checks passed,
so two-node inference works on the new image.

[after.json](after.json) and [after-probes.log](after-probes.log) hold the opt-in
probe: **12 PASS, 2 INCONCLUSIVE, 0 FAIL** (exit 2). No tool was executed, each
probe ran once, and no request retried.

| Probe | New-image observation |
|---|---|
| Prose, no tools | 19 content chunks before terminal, first at 154 ms, PASS |
| Prose with tools | 32 content chunks before terminal, PASS |
| Explicit thinking off, tools on | 17 content chunks, PASS |
| Explicit thinking on, tools on | 14 reasoning frames then streamed answer, PASS |
| Valid offered call | Correct `get_status(service="tensorfold")`, PASS |
| Tool-result round trip | Synthetic result reflected, PASS |
| Unoffered JSON / SSE | Native typed rejection, terminal wire contract, 2 PASS |
| Malformed envelope | Native typed rejection, PASS |
| No-tools executable envelope | Native typed rejection, PASS |
| Truncated fixture | Offered structured call, INCONCLUSIVE |
| Plain and fenced literals | Exact literals preserved, 2 PASS |
| Inline-code literal | Model invoked the tool instead, INCONCLUSIVE |

The two inconclusive cases are the same limits as the old-image baseline: the
model may have completed or replaced the requested fixture. They are not parser
passes, and exact classifier ownership still needs raw-token capture.

The first probe run produced one FAIL, `thinking_explicit_true: Reasoning
interleaved after visible answer`. The raw frame showed
`{"reasoning_content": ".", "content": "17"}` in a single delta: the engine
emits the last reasoning token and the first answer token together. The probe
judged ordering by event position inside that frame; it now compares frame
indices and only fails reasoning in a frame after the first content frame.
[`tests/live_tool_streaming_test.py`](../../../tests/live_tool_streaming_test.py)
locks in both the shared-transition pass and the later-frame failure. This was
a missing probe rule, not a parser or stream-ordering defect.

Not yet validated live: the `TOOL_OUTPUT_STREAMING=0` rollback needs another
launcher restart. It is covered by the installed-source CPU HTTP tests
(`test_roll_back_to_whole_answer_buffering`). SDK/proxy retry behavior and the
original agent scenario also remain open.
