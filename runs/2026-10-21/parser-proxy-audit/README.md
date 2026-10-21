# Parser and MediaLLMProxy audit, 21 October 2026

## Result

**The strict parser fixes are implemented and deployed. The main tool path is
consistent, but the proxy is not a byte-transparent stream forwarder.** Two
proxy defects need fixes: the loop-guard path drops the SSE terminal marker,
and the corrective retry banner can change an optional tool into a required
call. The audio detour also points to a backend with no listener.

This is an audit record, not a code deployment. Three swarm reviewers checked
the parser, proxy source, and deployment controls. The audit did not change
production config, restart a service, enter a rank with `docker exec`, execute
a generated tool, or start another model server. Isolated proxy copies used
random loopback ports for synthetic replay and were stopped after the checks.
The existing untracked assessment and `runs/2026/` files were left unchanged.

Primary probes ran at approximately 15:27 to 15:32 UTC. The session was then
interrupted. Final live checks ran at 17:18 to 17:19 UTC on the same date.

## 1. Parser plan versus implementation

The current contract is **section 12** of
[the parser plan](../../../streaming-tool-parser-fix-plan.md). Sections 1 to 11
are a historical proposal, as the warning at the top of that file states.

| Requirement | Implemented state | Source evidence |
|---|---|---|
| Strict validation with tools, empty tools, or no tools | Implemented for every strict chat request | `experiments/display-kv/apply.py:428-449` |
| Exact offered names, schema, IDs, call count, and request policy | Implemented, with whole-answer executable atomicity | `experiments/display-kv/overlay/tool_output.py:548-654` |
| Prose streaming | Implemented before the first raw or incomplete protocol candidate | `experiments/display-kv/overlay/tool_output.py:434-469` |
| Candidate inside a literal | Buffers the remainder, even when the final parse keeps it literal | Plan section 12.3 and stream replay tests |
| Partial marker or long namespace | Held across chunks; a long possible prefix freezes instead of repeated unbounded scans | Same stream implementation |
| Reasoning stream | Separate from executable answer parsing | Installer server wiring and HTTP tests |
| Decoder or final-prefix disagreement | Fails closed as `stream_prefix_changed` | Stream prefix checks and installed HTTP tests |
| Early call atoms and post-envelope prose resumption | **Not implemented, intentionally deferred** | Plan sections 12.2 and 12.4 |
| `TOOL_OUTPUT_STREAMING=0` | Supported content-buffering rollback, with strict parsing still active | Launcher wiring and installed CPU regressions |

There is no separate `tool_stream.py` module in this phase. The conservative
projector is `ToolOutputGuard.stream()`. Do not use the old proposal's early
atom or bounded-literal assumptions as acceptance requirements.

The display installer still requires pristine TensorFold
`07aa02585f64cfb04c60a96de6554c0d4708ed46`. The branch's earlier integration
commit `fa0ede4` did not upgrade this display lane to the 1.1.0 engine. Current
recipe HEAD for this audit was `42bb551e22184fb7ca9f06a9509e5f226cee4d73`.

## 2. Runtime and source identity

See [initial identity](runtime-identity.txt) and
[final identity and selected config](runtime-resume.txt).

| Item | Observed value |
|---|---|
| Both serving rank images | `tensorfold-dsv41:display-kv-0dc9c3fefb06` |
| Image ID on both nodes | `sha256:4dd9d9a7f234e057f72c37661075fdaf6fd2143706e55f7abcedad748b3e9d09` |
| Both ranks | `TF_TOOL_OUTPUT_STREAMING=1` |
| Installed guard, both nodes and working tree | SHA-256 `3ae1213289babe2f64bdc7a8bb8a97af65e4450746d5ee086b3169ec7f13fef2` |
| Installed server and family app | Strict guard wiring present, `strict_tool_output=True` |
| Proxy deployed DLL and fresh local Release artifact | SHA-256 `bf284a21c6d860f7e3e489e3869a6d87d74fb5de67ddcd605b66e16767ec6461` |
| Proxy source HEAD | `2e7ceb5094aa9c2e02d70e288ef753303ef140af` |
| Proxy source branch | `fix/audio-detour-configured-baseurl`, clean |
| Proxy service | Active, same PID `1596934` at both observations |
| Retired Proxy B | Inactive and disabled |

The ranks restarted during the interrupted interval, not through an audit
command. The initial starts were approximately 08:33 UTC. Final inspection
shows rank 0 started at **15:55:52 UTC** and rank 1 at **15:56:50 UTC**. The
image and guard hashes stayed the same. Do not compare engine counters across
that restart. The final real inference gate passed again.

## 3. Deployed proxy consistency

Proxy A reads the shared JSON and `appsettings.proxy-a.json` from
`~/mediallmproxy`, selected by its service environment. It forwards HTTP to
`127.0.0.1:8100`. No separate tunnel is required for this route.

| Gate | Preset and live health |
|---|---|
| Tool output ownership | `StrictNative` |
| Effective raw-call curing | `false` |
| Strict-guard retry | Enabled, maximum 1 |
| Raw-abort retry | `false` |
| Strict source matching | `strictGuardOnly=true` |
| Request context in retry | `true` |
| Non-streaming retry buffer limit | 16,777,216 bytes |
| Sampling | Temperature 0.4, top_p 1.0, enforced |
| Vision detour | Off |
| Audio detour | On, target `127.0.0.1:8085` |
| Tool-media rehome | On, not at end |

Proxy source `OAIPreRouter.Cli/Program.cs:62-65,263-264` disables raw curing
when the selected backend is StrictNative, even if its cure flag is true.
The response path uses that effective gate at lines 817-818 and 895.
`UnavailableToolRetryService.cs:223-248` requires the native guard source and
typed error and excludes `stream_prefix_changed`. The strict path clamps
retries to at most one at lines 291-296.

Streaming retry remains limited to the pre-visible preamble. The current
implementation stops inspection at the first ordinary data frame, including
a role-only frame, not only at the first text or tool payload
(`UnavailableToolRetryService.cs:408-432`). The live stack sends such a frame
before its rejection, so the SSE rejection was passed through without retry.
Non-streaming retries can override the backend's `x-should-retry:false` hint.
This is a deliberate configured proxy policy, not transparent forwarding.

## 4. Findings and required follow-up

### F1. High: loop-guard streaming drops the terminal SSE framing

**Confirmed in the deployed binary and current source.**

`OAIPreRouter.Cli/Services/SseWedgeProcessor.cs:51-55` returns immediately after
writing a chunk with a nonempty `finish_reason`. It does not forward the
following blank event separator or `data: [DONE]`.
`OAIPreRouter.Cli/Program.cs:842-869` selects this path when the wedge is enabled
and the tool-retry path is not in use. This includes the tested no-tools
request. The tested tool-enabled path retains `[DONE]`.

- Initial controlled repeats: direct **3/3 valid**, proxy **3/3 missing DONE**.
- Final repeats after the interruption: the same **3/3 versus 0/3** result.
- Independent raw HTTP captures confirm that the proxy body has neither DONE
  nor a final blank separator. This is not a probe JSON parsing artifact.
- Some clients accept EOF. Others can miss the final event or report an
  incomplete stream. Client-specific impact was not measured.

Evidence: [first repeats](sse-terminator/results.json),
[final repeats](sse-terminator-resume/results.json),
[raw direct](sse-terminator-resume/raw-direct.sse), and
[raw proxy](sse-terminator-resume/raw-proxy.sse).

**Fix:** stop wedge decision-making after a finish chunk, but preserve the full
terminal event, trailing usage events if present, and DONE. The existing DONE
branch at lines 41-45 also returns before the next blank line. Add raw-byte
regressions for both terminal branches. Existing `LoopGuardTests.cs` fixtures
use `finish_reason=null` before DONE and miss the real finish sequence.

### F2. Medium: corrective retry can require a tool when the request does not

`UnavailableToolRetryService.cs:252-279` tells the model to emit a complete
tool call unless `tool_choice` is explicitly `none`. An omitted or `auto`
choice does not require a call. At lines 530-536, `parallel_tool_calls:false`
also becomes "exactly one", although it only limits parallel calls, and
`max_calls:0` adds a no-call instruction that conflicts with the generic call
instruction.

The live A/B showed a concrete semantic difference. Direct non-streaming
output rejected an unoffered envelope with HTTP 500 and
`x-should-retry:false`. The same request through the proxy returned HTTP 200
and an offered `get_status` call after one corrective retry, despite the
original copier task saying not to perform the action. No tool was executed.
This does not show a guard bypass: the second generation is an offered,
validated call. It shows that schema validity is not task-intent preservation.

Evidence:
[direct rejection](live-ab/unoffered_json.A-direct.json) and
[proxy retry result](live-ab/unoffered_json.B-production-proxy.json).

**Fix:** preserve optional calls under `auto`, use "at most one" for parallel
false, and treat every effective zero-call policy consistently. Add banner
regressions for `auto`, omitted choice, named choice, `none`, parallel false,
and zero maximum. Keep the retry policy explicit to clients. No retry setting
was changed during this audit.

### F3. Medium: audio detour enabled with no STT listener

`~/mediallmproxy/appsettings.proxy-a.json:19,34-37` enables audio detour to
`127.0.0.1:8085`. There was no listener during initial or final inspection.
The current `CONFIG.md:95-96` already notes this limitation.

Uncached audio that reaches transcription will not fall back to the text
backend: `Program.cs:560-595` returns 502 for a backend failure or 504 for a
timeout. Cached transcripts can still succeed, so this is not a claim that
every audio request fails. No real audio was submitted.

**Action:** restore or correct the STT service. Do not disable detour as a
workaround unless native audio support of the primary is established.

### F4. Medium: config verification does not check the new guard gates

`~/scripts/mediallmproxy/config-show.sh:21-39` omits tool-output ownership,
raw curing, and all unavailable-tool retry settings. A clean output from that
script therefore cannot verify the most important gates in this audit.

**Fix:** compare JSON and `/health` for the contract, effective cure flag,
retry enable/count, strict-only gate, request context, and response limit.
Account for effective StrictNative overrides rather than comparing the raw
cure flag alone.

### F5. Medium operational risk: update script has no release-ref gate

The deployed DLL matches the clean source checkout. This is **not** a binary
mismatch. However, that source branch has six commits not on local `main`,
and local `main` has four not on the branch.
`~/scripts/mediallmproxy/update.sh:6-18` builds the current checkout without an
approved branch, tag, or clean-tree requirement.

**Action:** record an approved immutable source ref and DLL hash before an
update, and reject an unintended or dirty checkout. Do not switch branches or
redeploy merely to make this audit look clean.

### F6. Low: historical docs and control behavior disagree

- The old proxy `INSTRUCTIONS.md` describes an absent root JSON and environment
  overrides. Current root JSON exists and the active service has no drop-ins.
- `~/scripts/mediallmproxy/README.md:3-6` still says environment/drop-in config,
  while its later section describes the JSON migration.
- That README describes default actions on both proxies and temperature 0.5.
  Current controls default to A, `update.sh:41-53` restarts active units only,
  B is retired, and the live temperature is 0.4.
- `~/mediallmproxy/CONFIG.md:15` still labels a default restart as both instances.
- `docs/TOOL-OUTPUT.md:7-11` and the display experiment README still call
  `display-kv-58e319cb2866` deployed. That is the earlier recorded image, not
  the current `0dc9c3fefb06`. Its guard hash matches, so this is status drift,
  not evidence that the parser fixes are missing.

**Action:** update operator-facing current-state docs. Preserve old run records
as historical evidence. All control `.sh` files passed `bash -n`.

## 5. Validation results

| Check | Result | Saved evidence |
|---|---|---|
| Real staged installer and installed parser/schema/stream/HTTP suite | 70 tests normal and 70 under `-O`, PASS | [CPU log](cpu-installed.log) |
| Wrong pin, source drift, reapplication | Refused without writes; pristine input unchanged | Same CPU log |
| Probe assessment regressions | 10 normal and 10 under `-O`, PASS | [Assessment log](probe-assessment.log) |
| Full recipe offline gate | 254 discovered, 12 unittest skips, no failed step; host-tuning fixtures pass | [Offline log](offline.log) |
| Proxy Release tests | 287 passed, 0 failed, 0 skipped | [Proxy test log](proxy-tests.log) |
| Real inference gate | 5/5 initially and 5/5 after interruption | [Initial](verify-current.log), [final](verify-resume.log) |
| Identical-payload live direct/proxy A/B | Initial assessor: 10 PASS, 4 INCONCLUSIVE | [A/B results](live-ab/results.json) |
| Deterministic deployed-DLL contract A/B | StrictNative preserves 4/4 fixtures | [Exact replay](exact-contract-ab/results.json) |
| No-tools SSE terminal follow-up | Proxy fails 3/3 in each of two rounds, direct passes 3/3 in each | Linked under F1 |

The initial four inconclusive results are retained, not rewritten as passes:

1. The no-tools proxy completion lacked DONE. Follow-up confirmed F1.
2. Both model-backed four-space literal probes removed the leading spaces.
   They therefore did not induce the exact ownership fixture.
3. The proxy's non-streaming corrective retry produced a valid offered call
   instead of the requested rejection fixture. This is the F2 observation.

The exact synthetic A/B removes model variation. Both sides use isolated
copies of the **deployed DLL**, request raw curing, and receive the same
strict-guard-classified literal responses. Only `ToolOutputContract` differs.

| Fixture | LegacyRaw | StrictNative |
|---|---|---|
| Four-space literal, JSON | Fabricates a call | Preserves literal, no call |
| Four-space literal, SSE | Fabricates a call | Preserves literal, no call |
| Literal plus native call, JSON | Preserves this fixture | Preserves this fixture |
| Literal plus native call, SSE | Emits `mixed_native_raw_calls` after native call | Preserves content and native call |

This confirms the need to keep StrictNative enabled and raw curing disabled in production.
It does not establish broader model reliability or a performance improvement.
The live A/B uses temperature 0.4, top_p 1, seed 0, and 512 maximum tokens on
both sides. Cache and retry work differ, so timing fields are observations,
not a performance A/B. The engine's buffering rollback was not repeated live
because that would require a serving-stack restart.

The A/B generated one configured proxy retry. Its counters changed from
1 retry / 0 successes / 1 failure to 2 / 1 / 1. That increment is test activity,
not an unexplained production incident.

## 6. Next gates

1. Fix and test F1 in proxy source before the next proxy deployment.
2. Fix the F2 retry banner, then repeat auto/none/required/named policy checks.
3. Extend the config verifier and correct operator docs.
4. Resolve the optional STT availability gap separately.
5. Keep early call release deferred until the prefix-stability and global-policy
   requirements in parser plan section 12 are proved.
6. Test the original agent workflow and a real client SDK. Full raw-token
   ownership of model-generated literal/truncated fixtures is still unproved.

No parser source blocker was found in the exercised paths. The SSE proxy bug
means end-to-end streaming cannot be called fully conformant yet. Linters
`ruff` and `shellcheck` were not installed, so they were not run or installed.
No external release-state review or network performance claim is part of this
local audit.

## Reproduce

Run from the recipe root, with the current services already up:

```bash
python3 runs/2026-10-21/parser-proxy-audit/direct_proxy_ab.py --output /path/to/new/live-ab
python3 runs/2026-10-21/parser-proxy-audit/repeat_sse_terminator.py --output /path/to/new/sse-check
python3 runs/2026-10-21/parser-proxy-audit/replay_contract_ab.py --dll "$HOME/mediallmproxy/bin/OAIPreRouter.Cli.dll" --output /path/to/new/exact-replay
./verify.sh
```

The repeat script records regression verdicts but does not exit nonzero for
the known missing-DONE defect. Inspect its `results.json`. The initial live
probe also permits INCONCLUSIVE outcomes. Neither exit 0 is an all-pass claim.
Raw generated output in the JSON and SSE artifacts is untrusted test data.
User-specific home paths in test logs were replaced with `/home/<user>`.
