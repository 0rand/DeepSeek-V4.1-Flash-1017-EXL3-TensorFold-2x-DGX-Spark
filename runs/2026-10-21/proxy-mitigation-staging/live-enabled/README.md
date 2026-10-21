# Live enabled proxy validation, 21 October 2026

## Runtime

- Deployed proxy DLL: `cc8c6befe4ea7f5a965648656fd68ef9b7a0fe0ef7f1d3f6bd1ab93c7d9527ac`
- Source commit: `13b8561cf878e7a16fdf74d4e5277d9c69119b25`
- Live config SHA-256: `182f5d4178de532f43fb9c89c7f787e324ebb98d7186fc0aebdcaf804de9ed8f`
- Effective gates: `StrictNative`, raw cure off, wedge on, corrective retry on,
  maximum one retry, request context on, response buffer 16 MiB.
- Engine on `127.0.0.1:8100` was not changed or restarted.

The first deployment used the temporary safe profile and disabled the paths under
test. The live config was then switched to the original enabled behavior while
keeping the fixed DLL. The safe config remains available for rollback.

## Direct/proxy A/B

[results.json](direct-proxy-ab/results.json) and
[direct-proxy-ab.log](direct-proxy-ab.log) record seven request shapes on the
direct engine and production proxy.

| Outcome | Count |
|---|---:|
| PASS | 12 |
| INCONCLUSIVE | 2 |
| FAIL | 0 |

The two inconclusive cases are the four-space literal copier on both sides. The
model did not reproduce the supplied fixture exactly, so the probe does not
claim a parser result.

Observations:

- No-tools and tools prose streamed multiple content events, one finish and one
  complete DONE through the proxy.
- The valid native call passed through with one `get_status` call.
- `unoffered_json`: direct returned HTTP 500 with `x-should-retry:false`.
  Proxy returned HTTP 500 after `X-PreRouter-Tool-Retries: 1` and
  `X-PreRouter-Tool-Rejected: unavailable_tool_call`. The corrected retry did
  not fabricate an offered call.
- `unoffered_sse`: the role-only/error stream passed through with no retry,
  one typed error and one DONE. The existing visible-frame cutoff is unchanged.
- No-tools executable envelope passed through as one typed error and DONE.

## Live wedge

The live judge first returned `NO_LOOP` for a reasoning prompt, so the wedge did
not fire. A second probe used a loopback OpenAI-compatible judge returning
`NUDGE`. That judge is a test fixture only. The production advisor URL and wedge
threshold were restored after the probe.

[wedge-live.sse](wedge-live.sse) is the raw proxy response. It contains:

- the wedge banner in `reasoning_content`;
- one successful finish event;
- one complete final DONE event;
- correct final blank-event separation;
- no executable calls or native errors.

Metrics before and after are in
[wedge-metrics-before.json](wedge-metrics-before.json) and
[wedge-metrics-after.json](wedge-metrics-after.json). The proxy log recorded
`verdict=NUDGE`.

## Final normal stream

After restoring the real advisor and threshold,
[normal-live.sse](normal-live.sse) passed with nine content events, one finish,
one DONE and complete event separation.

The live effective-config check passed against active health:
[config-verify.log](config-verify.log).

## Limits

- The wedge judge was a controlled loopback stub for the forced-NUDGE case. The
  real advisor returned `NO_LOOP` for the tested prompt.
- The 2 INCONCLUSIVE literal cases are fixture-induction limits.
- No SDK, original-agent, multimodal, STT or rollback drill acceptance is
  claimed.
- No returned tool was executed.
