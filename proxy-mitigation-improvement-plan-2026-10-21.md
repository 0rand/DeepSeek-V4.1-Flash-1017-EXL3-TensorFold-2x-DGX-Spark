# Proxy mitigation and improvement plan

Date: 21 October 2026. Status: proposed, not applied.

Basis: [completed parser/proxy audit](runs/2026-10-21/parser-proxy-audit/README.md),
commit `44f021b`. This plan adds no production change and does not replace the
audit's historical evidence.

## 1. Objective and fixed constraints

Restore complete SSE framing, preserve the original task and tool policy on
retry, and make deployment/config checks reliable. Resolve the separate STT
availability gap without weakening the serving guard.

Keep these settings and boundaries unless a separate approved change requires
otherwise:

- Backend contract `StrictNative`; effective raw-call curing off.
- Strict validation and whole-answer executable atomicity on.
- Current engine image, source pin, display-KV, memory, spill, sampling, tool
  schemas, and model aliases unchanged.
- No early call atoms or post-envelope prose resumption in this work.
- No serving-model restart, second DeepSeek stack, `docker exec` into ranks,
  display-manager stop, or driver/kernel update. Do not start a new GPU model
  service on either serving node as part of this mitigation.
- Proxy B stays retired. Address only `media-llm-proxy.service` in a cutover.
- No generated tool is executed by the validation suite.
- Keep network bindings unchanged. This plan does not silently remove LAN
  access or open a new listener beyond loopback-only test proxies.

The parser itself has no confirmed blocker in the exercised paths. Changes
should start in the proxy, not in the engine guard.

## 2. Recommended order

| Order | Work package | Covers | Production effect |
|---|---|---|---|
| 0 | Capture a release/config baseline and build a strict contract test gate | All findings | None |
| 1 | Canary and optionally apply a temporary safe-mode config | F1, F2 | Proxy-only restart after approval |
| 2 | Fix terminal event handling | F1 | Enable the wedge only after its gate passes |
| 3 | Fix retry policy guidance and document recovery semantics | F2 | Re-enable retry only after its separate gate passes |
| 4 | Improve config verification, release controls, and current-state docs | F4, F5, F6 | Safer subsequent updates |
| 5 | Restore and test STT as an independent change | F3 | Audio service/config only |
| Later | Close SDK, media, and original-agent coverage gaps | Residual audit limits | Separately gated |

Development of packages 2, 3, and 4 can proceed in parallel on separate files.
Production promotion remains sequential. Give each behavioral change its own
focused commit and validation record. Do not mix an engine upgrade or a broad
refactor into these fixes.

## 3. Immediate mitigation, before a fixed binary is available

### 3.1 Recommended temporary profile

Stage this **delta**, not a replacement config, for Proxy A:

```json
{
  "LoopGuardOptions": {
    "WedgeEnabled": false
  },
  "UnavailableToolRetryOptions": {
    "Enabled": false
  }
}
```

Keep `LoopGuardOptions.Enabled` at its current value. This disables the broken
mid-turn wedge, not the existing request-side loop-guard stages. Keep
`ToolOutputContract=StrictNative` and `CureRawToolCalls=false` explicit.

Effects and trade-offs:

- No-tools streams avoid the broken wedge path.
- Tool streams also avoid it when retry is disabled.
- Typed strict rejections pass back to clients instead of being changed by a
  corrective generation.
- Automatic mid-turn loop recovery and automatic strict-error recovery are
  temporarily unavailable. Long reasoning can still run to the existing
  generation limits. This is an explicit reliability trade-off, not a free fix.

**Critical ordering:** do not disable retry while the broken wedge remains
armed. `Program.cs:768-769,842` otherwise moves tool-enabled streams into the
same broken path. If flags are applied separately, disable the wedge first,
validate it, then disable retry. Prefer validating both flags in one isolated
profile and applying that vetted delta together in one approved proxy restart.
Using only `MaxRetries=0` is not equivalent to disabling retry for path
selection: the current `retryEnabled` gate checks `Enabled` and declared tools.

This is the recommended temporary profile, not an instruction to apply it
without approval. If no proxy restart is acceptable, leave config unchanged,
document the known SSE risk, and prioritize the tested binary fix. Do not work
around it by executing raw protocol text or switching to the legacy scanner.

### 3.2 Mitigation gate

1. Save the current config files, selected service environment, active unit
   list, source ref, and all deployed binary-file hashes. Store backups outside
   the active config path. Do not publish keys or machine-specific paths.
2. Start one isolated copy of the deployed proxy DLL on a random loopback port,
   with the proposed delta and a synthetic upstream. Test no-tools and tools,
   JSON and SSE, native calls, literal text, and typed rejections.
3. Against the same model on `:8100`, send short sequential direct/canary
   requests. Match sampling and request body. Do not execute returned calls.
4. Require a complete finish event and exactly one complete DONE event for
   successful SSE. For a complete native rejection fixture, require exactly
   one typed error then DONE, no executable call, and no successful finish.
   Transport-truncated streams must remain visibly incomplete, not be repaired
   into this complete-rejection contract.
5. Confirm zero corrective retries with the temporary profile. Confirm literal
   preservation and incremental prose with tools and without tools.
6. If approved, drain or let in-flight proxy requests finish, apply only the
   vetted config delta, and restart **Proxy A only**. This can briefly interrupt
   clients. Do not claim a zero-downtime cutover.
7. Run the production contract gate immediately. Confirm the serving rank
   identities did not change.

Current `/health` omits `loopGuard.wedgeEnabled`. Until package 4 exposes it,
verify the selected config and environment precedence, plus the behavioral
streaming gate. `loopGuard.enabled=true` alone does not prove the wedge is off.

**Rollback:** prefer the vetted temporary profile if a new binary fails later.
Restoring the original profile reintroduces the known bugs, so that is an
operator-approved last resort, not the default rollback. Keep both profiles
and their hashes clearly named.

## 4. Permanent fixes

### Package 2: complete terminal SSE handling (F1)

**Scope:** `OAIPreRouter.Cli/Services/SseWedgeProcessor.cs`, the real forwarding
route in `Program.cs:892-896`, and focused tests in
`OAIPreRouter.Tester/LoopGuardTests.cs` or a dedicated terminal-framing suite.
The route currently aborts caught stream failures only when raw curing is on,
which is false under StrictNative. Include this adjacent failure-signaling
case so a strict-native read/write failure cannot become ordinary clean EOF.

Implementation contract:

- On `finish_reason`, stop evaluating or launching a wedge. Do not end output
  at that data line.
- Drain the complete finish event, any trailing usage-only events, and the
  complete `[DONE]` event through its blank separator.
- Treat DONE as a terminal event, not just a terminal line. Preserve event
  order and payloads, including LF/CRLF and supported multiline SSE data.
- Apply the same framing contract to the normal path and the replacement
  stream after a successful wedge.
- Do not fabricate DONE, a successful finish, or a replacement call when the
  upstream truncates or fails. Preserve failure/cancellation semantics and
  dispose both attempts correctly.
- Do not buffer the whole answer. Retain cancellation and downstream
  backpressure. Do not add another raw tool-output parser.

Required regressions:

| Case | Expected observation |
|---|---|
| Content, finish, DONE | Complete finish and DONE event separators |
| Finish, usage-only event, DONE | Usage retained in order |
| Role-only stream followed by error and DONE | Error preserved, no successful finish |
| DONE without a prior finish | Forward what was received, do not invent a finish |
| EOF before terminal completion | No fabricated successful termination |
| Post-header upstream read or downstream write failure through real route | Abort or otherwise expose incomplete transport, never normal clean completion |
| Split UTF-8, CRLF, comments, multiline event | Valid unchanged payload and event boundaries |
| Wedge not triggered, judge fails, replacement succeeds/fails | Framing preserved on every path |
| Client cancellation or slow client | Prompt cancellation, bounded streaming memory, no leaked upstream |
| Terminal chunk followed by more reasoning | No new judge/retry is launched after terminal state |

**Acceptance:** deterministic byte/event tests pass, the full proxy suite
passes, and the previously failing repeated no-tools probe passes for all
three direct and all three proxy requests. Also test tool streams with retry
both enabled and disabled. Restore `WedgeEnabled=true` only after this gate.

### Package 3: preserve request policy and task intent on retry (F2)

**Scope:** `UnavailableToolRetryService.cs`, its options/docs, and
`StrictToolRetryTests.cs`. Do not change the offered tools, tool schemas,
thinking mode, sampling, or user/system messages to force apparent success.

Use one normalized policy view for banner generation:

| Effective request policy | Corrective guidance |
|---|---|
| Choice omitted, null, or `auto` | Calls remain optional. Follow the original task. If a call is needed, use only an offered function with valid arguments. |
| `none` | Do not emit an executable call. Preserve requested inert examples as text. |
| `required` | At least one offered valid call, subject to the backend's supported limits. |
| Named function choice | Follow the exact function choice and its supported cardinality. |
| `parallel_tool_calls:false` | At most one call, not exactly one under `auto`. |
| Invalid or incompatible policy | Preserve the backend request error. Do not retry around it or rewrite it into a different policy. |

Remove unconditional instructions to emit a call under `auto`. Never add a
blanket ban on literal protocol examples just because executable calls are
forbidden. Keep `stream_prefix_changed`, non-guard errors, transport failures,
and output already exposed to the client outside corrective retry.

#### Clarification of the audit's `max_calls:0` example

The proxy currently formats a request `max_calls` value into its banner. In
the pinned engine, `tensorfold/server/tool_policy.py:60-71` does **not** read a
wire-level `max_calls` field. Its internal `max_calls` is 1 for parallel false,
otherwise unset. The guard also takes that limit as an internal parse argument.

Thus the contradictory zero-limit banner is a proxy defect, but this audit
does not establish a supported backend zero-call API. Remove unsupported
request-field guidance in this focused fix. Do not silently introduce a new
API limit or claim the engine enforces a field it ignores. Use `tool_choice`
`none` for the established no-call policy. A new negotiated call-limit API,
if wanted, needs its own documented request validation and integration tests.

#### Recovery policy decision

The current configured single retry overrides a backend `x-should-retry:false`
hint for eligible non-streaming strict errors. Separate that decision from
fixing the banner:

- Temporary safe mode: recovery disabled, errors pass through.
- The proposed operational profile does not change shipped `Enabled` defaults
  or existing deployments implicitly. Do not change retry authorization in the
  banner-only patch.
- Fixed release: keep the approved safe-mode config until recovery is explicitly
  approved again. Before a separate policy implementation, choose whether to
  honor `x-should-retry:false` or allow a clearly named opt-in override.
- If recovery is wanted, define a clearly named opt-in policy for this
  override. Retain maximum one, exact typed/source matching, original request
  context, and the current conservative preamble cutoff.
- Do not broaden streaming retry to continue past role-only frames in this
  patch. That is a separate behavioral change with client and ordering tests.
- Never interpret valid schema output as proof that the original task was
  fulfilled. Preserve failure when the task cannot be completed safely.

Required tests include omitted/null/auto/none/required/named choice, parallel null
and false, unsupported call-limit fields, invalid combinations, the verbatim
copier case, a second guard failure, a second non-guard failure, buffer limits,
and cancellation. Verify retry counters and headers, including the documented
handling of `x-should-retry`.

**Acceptance:** banner/unit tests prove that optional calls stay optional and
no contradictory instruction is generated. Deterministic replay proves no
new native call can be fabricated from literal content. Model-backed copier
probes must not be labeled PASS when the model changes the fixture. A real
client/agent check must precede enabling recovery again.

### Package 4A: trustworthy effective-config verification (F4)

**Scope:** proxy `/health`, `config-show.sh`, `restart.sh`, and their tests.

- Add health fields for wedge enablement and the selected retry policy. Keep
  existing fields compatible. Expose values, not secrets.
- Compare contract, effective cure, retry enable/count/raw-abort/context/byte
  limit, wedge gate, primary URL, and existing media/sampling gates.
- Model config precedence correctly: startup content root, shared JSON,
  selected environment JSON, and applicable environment/command overrides.
  Do not use the current shallow dictionary merge for nested sections.
- Prefer the app's resolved options as the runtime authority. For ambiguous
  or unsupported selectors/overrides, fail with a clear diagnostic rather
  than sorting environment names and guessing which one won.
- Validate expected derived behavior: StrictNative forces cure off. Distinguish
  a requested cure flag from its effective value.
- Fail nonzero for a required missing/invalid file, a live mismatch, an
  unreachable active instance, or an unverified safety gate. A deliberately
  retired instance is not a failure and must not be started by a check.
- Test before/after restart detection so editing JSON without restarting does
  not look applied. Print safe field names and redacted values only.

**Acceptance:** fixtures detect each wrong guard gate, nested override,
environment shadow, stale process config, and retired-instance condition.
The production verifier agrees with independent health and streaming checks.

### Package 4B: release provenance and reversible promotion (F5)

**Scope:** `update.sh` and deployment procedure. The current script copies
binaries before checking config and only warns on invalid config. Correct
this along with the missing release-ref gate.

Before replacing any deployed file:

1. Require an approved immutable commit, clean source tree, recorded SDK/runtime,
   passing tests, and explicit target unit. A branch name alone is not provenance.
2. Validate required config for the target instance and effective safety gates.
   Fail closed before copying anything if validation fails.
3. Publish to a disk-backed staging/release directory. Do not modify the dev
   checkout or publish into the live bin directory. Avoid RAM-backed temp space.
4. Record source ref, artifact hashes, config hashes, compatibility fields, and
   test-record locations in a release manifest. Keep secrets out of it.
5. Back up the complete prior artifact set and config snapshot. Promote the
   DLL, deps, and runtimeconfig as one coherent release, with a controlled
   directory switch or another tested transactional method.
6. Restart only approved active Proxy A, then run the real contract gate.
   `/health` alone is not a release gate.
7. On failure, restore the complete previous artifact set and vetted safe-mode
   config, restart A, and run the contract gate again. Do not stop the engine.

Add tests that interrupted builds, bad config, missing files, wrong ref, dirty
tree, failed canary, and failed promotion leave the old release recoverable.
Controls should never awaken retired B or silently pull/merge a branch.

### Package 4C: current-state operator docs (F6)

Update `~/mediallmproxy/CONFIG.md`, its stale `INSTRUCTIONS.md`, and
`~/scripts/mediallmproxy/README.md` from the verified config/control contract:
JSON content root and precedence, A-only defaults, active-only update behavior,
actual sampling, strict ownership, optional retry semantics, wedge gate, and
STT availability. State how to inspect the current image instead of calling an
old image permanently deployed.

Keep audit and run records immutable. Add a new deployment record for each
cutover. Where operational files are not tracked, establish a tracked source
and an explicit sync/install step instead of maintaining divergent copies.

### Package 5: restore STT independently (F3)

- Establish the intended STT service and an approved model/service config.
  Confirm why `:8085` is absent before restoring or repointing it. Check host
  and GPU memory headroom and approved service placement. Do not start a new
  local GPU STT process on the serving pair merely to fill the missing port.
- Restore that service, or update only the audio backend URL to an approved
  working endpoint. Do not point at an unrelated service that merely answers
  `/health`.
- With consent to process test audio, use a small synthetic/local speech sample
  and verify transcription plus downstream answer. Check uncached and cached
  cases, timeout, backend failure, non-speech handling, and URL policy.
- Preserve the current 502/504 failure behavior when transcription is
  unavailable. Do not pretend that raw audio forwarding is a mitigation until
  native support is proved.
- Expose a clear unavailable/degraded audio capability separately from the main
  text model's healthy state. Do not make text-only health depend on an optional
  STT service without an explicit product decision.

**Acceptance:** intended speech works through Proxy A, failure modes are
predictable, and text/native tool regression checks still pass. Roll back only
the audio change if its gate fails.

## 5. Shared release and validation gates

Create a reusable strict contract runner in the proxy/control source, not by
editing the immutable scripts saved in the audit run. It must accept direct
and proxy base URLs so it can target isolated canaries, block external
redirects, run sequential bounded requests, and never execute returned tools.

Unlike the audit snapshot scripts, the release runner must exit nonzero for a
confirmed failure. Mandatory checks must also block promotion when
INCONCLUSIVE. Do not convert a model fixture-induction failure into a pass.
Use deterministic synthetic upstreams for exact ownership/framing cases and
real model/client requests for end-to-end behavior.

| Gate | Pass condition |
|---|---|
| Source and release | Approved ref, clean tree, complete matching artifacts, safe config |
| Proxy unit suite | Existing 287 tests remain green and new regressions pass |
| SSE success | Full events, preserved usage, exactly one DONE, no missing terminal separator |
| Complete native SSE rejection fixture | One native typed error then DONE, no call or successful finish |
| Transport-truncated SSE | Incomplete/failure signal, no fabricated finish or DONE |
| Prose streaming | Multiple content events before terminal, tools and no tools |
| Native call/literal | Native offered call unchanged; literal never converted to executable output |
| Retry policy | None in safe mode, at most one only when explicitly enabled and eligible |
| Cancellation | No orphan attempt or new corrective request after client cancellation |
| Config verifier | Nonzero for wrong/unverified gates, correct retired-instance behavior |
| Real inference | Direct `./verify.sh` 5/5 and bounded proxy contract checks |
| User workflow | Actual SDK plus original agent scenario complete with a recording/stub tool executor, without executing generated calls |
| Rollback drill | Previous artifacts plus safe config restore and pass the same gate |

Save each new canary, production, and rollback record under the UTC-date run
folder and add it to that day's index. Record the treatment (flags, artifact,
retry policy), request hashes, raw SSE, outcomes, and counter baselines. Do not
mix counters across restarts. Do not claim performance improvements from this
functional A/B. Performance claims need the repository's separate fair-A/B
method.

Monitor explicit stream-complete/incomplete, strict rejection, retry attempted/
succeeded/failed, and skipped-after-output counts. Define retry success from a
terminal result, not just the first successful/role-only frame. Audit repeated
attempts without storing raw arguments, reasoning, audio, or credentials in
normal logs. A confirmed framing failure, fabricated literal call, or policy
violation blocks promotion or triggers the documented rollback. Do not invent
numeric error-rate thresholds without traffic baselines.

## 6. Decisions and stopping rules

No live input is needed to prepare or review this plan. Before execution, the
operator must approve:

1. The temporary loss of wedge and corrective retry, plus the proxy restart window.
2. Whether automatic strict-error recovery is wanted after the banner fix, or
   strict pass-through remains the desired operating mode.
3. The intended STT endpoint/service and use of test audio.
4. The exact release ref, target unit, and rollback artifact/config pair.

Work is complete when F1-F6 have their acceptance evidence, production config
and operator docs agree, the real client/agent gate passes, and the rollback
is demonstrated. A healthy endpoint or green unit suite alone is insufficient.

Full hybrid early-call streaming, engine upgrades, expanded schema support,
broader streaming retries, and new tool-limit APIs stay out of this change.
They require separate designs and proofs. This prevents a mitigation from
weakening the already working strict guard.
