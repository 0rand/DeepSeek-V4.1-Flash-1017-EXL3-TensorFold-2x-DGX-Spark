# Parser guardrails continuation, 2026-10-21

## Status and scope

At 06:53 UTC the serving image was still
`tensorfold-dsv41:display-kv-40a46c96d4bd`, image ID
`sha256:97b3670b1c160a658c76a3a88decdaf1cc77ced74df66a2cffd53970cd361be0`.
The strict foundation and conservative prose projector are integrated in source,
**not yet deployed or model-validated**. Head-node sudo authorization is needed
for the display launcher's DRM preparation. No service was stopped, no live rank
was entered with `docker exec`, and no second model server was started.

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

## Staged image provenance and outstanding live gate

For the existing base image and staged source at this checkpoint, the launcher's
expected new image tag is `tensorfold-dsv41:display-kv-1fe0a7c3556a`, source
fingerprint `1fe0a7c3556a603cc4b8bee5d83d990fe01ddcaf7240fc9178d5160ead47e7aa`.
It is an expected build identity, **not proof that an image was built**.

| Source | SHA-256 |
|---|---|
| `experiments/display-kv/apply.py` | `1ba8020b5984d131dcb63fe56494a931edb889f0363d2fa5be00376d01e06785` |
| `experiments/display-kv/overlay/tool_output.py` | `3ae1213289babe2f64bdc7a8bb8a97af65e4450746d5ee086b3169ec7f13fef2` |
| `tests/live_tool_streaming.py` | `521f646258f9739a169227c63d57aec2cf94d7d96f224598bca701d65e72d5d5` |

The previous `.display-kv-image` identity was saved before considering a cutover.
The service was deliberately left running because head `sudo -n true` requires
a password. The worker's noninteractive sudo was available. Do not stop a
healthy stack without being able to complete the head's DRM preparation.

Once sudo is available, run in the head-node terminal:

```bash
./stop.sh && ./start-display-kv.sh --build
./verify.sh
python3 -B tests/live_tool_streaming.py --base-url http://127.0.0.1:8100 --output /path/to/after.json
```

The live probe script is loopback-only, blocks redirects/proxies, runs no tools
and has bounded generation. Exit 1 means failures, 2 means only inconclusive
cases remain. Preserve those distinctions. Also validate actual SDK/proxy retry
handling and the original agent scenario without enabling an unavailable tool.
