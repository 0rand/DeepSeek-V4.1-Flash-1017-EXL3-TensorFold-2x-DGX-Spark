# Proxy mitigation source staging, 21 October 2026

## Result

The source for work packages 0, 2, 3, 4A, 4C and the safe part of 4B is staged.
The existing model stayed up. Proxy A on port 8000 stayed active with the same
PID. Retired Proxy B stayed inactive. No installed config or binary changed.

Package 5 (STT), production recovery authorization, production cutover and a
production rollback drill are not done. Each needs a separate approval.

## Plan review

The [mitigation plan](../../../proxy-mitigation-improvement-plan-2026-10-21.md)
is suitable for source staging with these controls:

- Keep the engine, strict ownership, model aliases, sampling and bindings
  unchanged.
- Keep banner correctness separate from retry authorization. The staged fix
  does not change the existing `x-should-retry` override. Do not enable recovery
  again before a separate decision and a real client gate.
- Do not apply the safe-mode delta during staging. Review
  `ops/mediallmproxy/safe-mode.delta.json` in the proxy source checkout. Apply
  both gates before the old wedge recovery path.
- Track control sources in the proxy repository. An explicit install step must
  replace the current untracked operational copies after approval.
- Use deterministic synthetic replay for exact framing and ownership. A model
  probe cannot prove a typed rejection if the model does not induce the fixture.
- Do not start an STT model. Endpoint, placement, memory headroom and consent to
  process test audio remain open.

## Changes staged in the proxy source checkout

The proxy source is a separate checkout on top of clean commit `2e7ceb5`. The
changes are committed as `13b8561` on `WIP-2026-10-21-Proxy-tool-fix`. They are
not deployed.

| Area | Change |
|---|---|
| Terminal SSE | Forward raw events live with an inspected-size cap. Finish stops judging but drains usage and the complete DONE event. EOF never gets a fabricated finish or DONE. Both first and replacement streams use the same rules. |
| Stream resources | Bounded reasoning retention, cancellation checks, replacement stream and response disposal, and abort on a post-header forwarding failure. |
| Retry guidance | One normalized request policy: omitted/null/auto optional, none forbids executable calls, required needs at least one, named choice is exact, parallel false permits at most one. Wire `max_calls` guidance is removed. Invalid policy passes the backend response through. Literal examples remain allowed. |
| Retry streaming | The preamble reader inspects a complete initial event with a 64 KiB bound. Role-only and other visible frames keep the existing no-retry cutoff. It does not broaden retry authorization. |
| Health | Add wedge enablement, requested and effective cure flags, explicit retry policy labels, video support and redacted media URL policy. Existing fields remain. |
| Controls | Tracked Proxy A-only effective-config verifier, restart check, safe-mode delta, strict contract gate, fixed fixtures, isolated actual-DLL canary, and stage-only release manifest tool. |
| Docs | Tracked `CONFIG.md` and `INSTRUCTIONS.md` replace stale current-state claims. Retry docs use the normalized policy. |

### Release controls are deliberately stage-only

`update.sh --apply` refuses before a live directory or service change. The
production content root combines config and the live `bin` directory. The plan's
rollback needs an approved transactional content-root design. The tested
`release.py` promotion primitives do not prove a safe switch for that layout.
Do not weaken this refusal to make a demonstration pass.

## Validation

All .NET tests ran serially because host free memory was low. No other .NET
build ran at the same time.

| Check | Result | Evidence |
|---|---|---|
| Proxy unit suite | 308 passed, 0 failed, 0 skipped (287 before; 21 new) | [proxy-suite.log](proxy-suite.log) |
| Control tests, normal and `-O` Python | 25 passed each | [ops-tests.log](ops-tests.log) |
| Mandatory pass contract fixtures | 5 PASS, exit 0 | [contract-fixtures.log](contract-fixtures.log) |
| Negative contract fixtures | 1 FAIL and 1 INCONCLUSIVE, exit 1 as required | Same log |
| Actual staged DLL, synthetic upstream | 36/36 PASS across wedge/retry/safe profiles | [results](synthetic/results.json) |
| Real inference `./verify.sh` | 5/5 PASS on the running model | [verify.log](verify.log) |
| Direct/canary live probes | 22/22 PASS; three sequential no-tools runs per wedge canary, tools prose, and required native call on direct and canary | [results](live/results.json) |
| Recipe offline suite | All steps PASS; 254 tests ran, 12 skipped, no failed step | [recipe-offline.log](recipe-offline.log) |
| Current pre-cutover config check | Exit 1 as expected: old binary health lacks the new fields, and the candidate config does not yet state every media gate | [config-check-current.log](config-check-current.log) |
| Isolated controls install | Two installs to a scratch destination succeeded; the second kept a complete timestamped previous copy | Lint and install commands ran in the staging session |

Synthetic replay is byte-exact for prose, usage after finish, DONE-only,
CRLF/multiline UTF-8, comments, native calls, literal text, JSON, typed
rejections, and upstream transport truncation. It tests the built DLL through
the real HTTP route with a CPU-only fixture. The synthetic fixture for a typed
rejection was retried once only in the retry-enabled profile.

The live probes are functional correctness checks, not a performance A/B. Every
successful stream had one successful finish, one complete final DONE event, and
retained usage when present. The native-call case had exactly one offered
`get_status` call with `{}` arguments. No returned call was executed.

The live request body fixed temperature at 0.4, top_p at 1.0, thinking off,
streaming on, and 256 maximum tokens. Both sides received the same body. The
direct URL and each isolated canary used the same model at
`http://127.0.0.1:8100`.

## Limits

- The typed strict-rejection path is proved with a deterministic upstream, not
  with a reliable live model induction.
- The live probes cover text and tools. They do not accept the SDK, the original
  agent, multimodal, STT, or a real production cutover.
- The canaries disable multimodal detours. They do not prove media paths.
- No production rollback drill ran.
- The safe-mode profile was tested, not applied. It trades automatic loop and
  strict-error recovery for predictable pass-through.
- `update.sh --apply` is disabled pending the content-root switch design.
- The effective-config verifier refuses the current pre-cutover process because
  the old health response lacks the new fields. It also requires the candidate
  config to state every media gate explicitly. Do not deploy the old binary to
  make that check pass.
- The existing non-streaming retry can override `x-should-retry:false`. That
  behavior remains unchanged and still needs an explicit operational decision.
- The recipe-wide Ruff command reports 77 pre-existing findings in experiment
  overlays. None is in this change. The new operational Python passes Ruff.
- Served image and rank identities did not change during the checks. See
  [baseline.txt](baseline.txt), [provenance.txt](provenance.txt) and
  [post-check.txt](post-check.txt). No canary process stayed after the run.

## Approval gates

1. Accept the temporary loss of wedge and corrective retry, and the eventual
   Proxy A restart window.
2. Decide whether strict-error recovery returns, including the
   `x-should-retry:false` override policy.
3. Approve the STT endpoint and service placement, with consent to process test
   audio.
4. Approve the immutable release ref, target unit, rollback artifact pair and
   content-root promotion design.
5. Run the direct inference, real client/agent, production contract and rollback
   gates after that approval.

The original untracked assessment and `runs/2026/` were not changed.
