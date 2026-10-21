# Streaming tool calls under strict DSML validation — findings, proposal, plan

> **2026-10-21 continuation: sections 1-11 are the historical proposal, not the
> implemented contract.** Replays found counterexamples to its bounded literal
> holdback and early-rejection assumptions. The current conservative rollout,
> upstream comparison, blockers and next gates are in **section 12**. No early
> executable atoms or post-envelope resumption are implemented in this revision.

Date: 2026-10-21
Branch under review: `fix/streaming-dsml-guard` (incoming PR: DSML parser fixes)
Live validation stack: `localhost:8100` (DeepSeek-V4.1-Flash-1017, display-KV image)
Prototype scratch: `/tmp/opencode/dsml-stream/` (wiped on reboot; every artifact is
re-creatable from this document)

## 1. Executive summary

The PR replaces the legacy permissive tool-call guard with a strict, dependency-free,
fail-closed validator plus its own JSON Schema subset and bounded-regex engine. The
hardening itself is sound and worth landing. **As wired, the PR kills streaming for every
strict chat answer — including plain prose with no tools at all** — because it validates
the entire answer before releasing any content.

Feasibility verdict: **we can keep streaming without giving up any of the PR's validation
strength.** The hybrid contract:

> Stream answer prose token-by-token. The moment generation enters an executable envelope
> opener (tool_call, either DSML wrapper, function= XML, GLM), stop publishing content;
> buffer the envelope atomically. When its closing marker arrives, validate the whole
> buffered envelope with the PR's own strict parser. Valid -> release one complete
> tool_calls atom in-stream and resume normal prose streaming after the envelope.
> Invalid (unoffered name, schema mismatch, truncated close, malformed body, bad ID,
> policy breach) -> the PR's existing terminal ModelOutputError (one SSE error frame +
> `data: [DONE]`, `x-should-retry: false`); prose streamed before the envelope is inert
> and stays out.

| axis | main (deployed today) | PR as written | PR + streaming projector |
|---|---|---|---|
| tools + strict chat | all answer content buffered; calls at finish | same | prose streams; call atoms release at envelope close |
| no tools field | prose streams live (guard inactive) | **all content buffered** (wiring broadened) | prose streams live (unchanged from main) |
| no tools, model writes a call envelope | legacy parser repairs/masks | whole answer rejected, nothing streamed | prose up to opener streamed, envelope rejected at close |
| literal (fence/quote/inline) examples | legacy parser masking, some slip cases | strict literal ownership, all buffered | strict literal ownership, streamed as they settle |
| reasoning | streams | streams | streams |

Validation strength is unchanged: the same guard module validates every envelope; the
finish-time authority remains `guard.parse(raw_answer)`; the projector only decides
*when* bytes may leave the process.

## 2. What the PR changes (technical read)

Two files carry the substance; the rest is wiring and docs.

- `experiments/display-kv/overlay/tool_output.py` (the guard, ~780 lines): complete
  envelopes for `tool_call` JSON, Qwen function/parameter XML, GLM argument tags, and
  BOTH DSML wrappers (spaced `<｜DSML｜ calls>` and unspaced `<｜DSML｜tool_calls>`);
  strict JSON semantics (duplicate-key hook, nonfinite rejection, no repair/truncate
  path); literal ownership machinery (fences incl. unclosed, inline spans, line-quoted
  examples via `_opening_quote`/`_quoted_end`); its own JSON Schema subset with step
  budget (10000 steps), regex work budget (2,000,000), nesting caps, unsupported
  keywords fail closed, bounded regexes (max one variable repeat, mandatory anchors);
  `tool_choice`, `parallel_tool_calls`, `max_calls`, ID rules all in one pass.
- `apply.py`: the guard is now created for ALL strict chat requests — the previous
  `and (tools or "tools" in body)` condition is REMOVED; `ToolCallStreamer` is disabled
  whenever the guard is active; `x-should-retry: false` on the terminal error; new CPU
  repro pipeline `run_tool_output_cpu.py` that installs into a pinned checkout and runs
  the real installer plus tests.
- `docs/TOOL-OUTPUT.md`: rewritten; the Streaming section openly states that answer
  content is buffered until completion for ALL strict requests, reasoning excepted.

The container source is not in this repo, but the pinned pristine checkout exists at
`/tmp/opencode/tfv41-display-source` (HEAD matches installer pin
`07aa02585f64cfb04c60a96de6554c0d4708ed46`). Running the repo's `apply.py` transforms over
it locally shows the installed code paths (rendered copies saved at
`/tmp/opencode/server_patched.py`, `http_patched.py`, `app_patched.py`).

### 2.1 Installed suppression points (server.py::App.run after transforms)

```python
tool_output_guard = (ToolOutputGuard(tools, request=body) if chat
                     and getattr(self, "strict_tool_output", False) else None)   # PR: no tools check
calls_stream = (ToolCallStreamer(tools) if tools and not policy.single
                and tool_output_guard is None else None)
...
def visible(finished):
    ...                                                    # split_thinking done above
    answer_raw[0] = answer
    if tool_output_guard is not None:
        # Validate the whole answer before releasing content or executable call deltas.
        return reasoning, ""          # <- every content delta withheld while streaming
...
# finish path:
if tool_output_guard is not None:
    content, calls = tool_output_guard.parse(raw_answer, max_calls=policy.max_calls)
tail = content[sent["content"]:] if content.startswith(answer[:sent["content"]]) else ""
```

So under the PR: per-round `visible()` returns answer="" (zero content deltas), the call
streamer stays None, and one burst at finish delivers everything.

## 3. Findings — validation against the live stack

All probes ran against the deployed stack (localhost:8100). Raw transcripts in section 10
and in the session log.

### 3.1 Guard active (request with tools, thinking off)

Frame order observed on `/v1/chat/completions`:

```
1    role assistant
2-7  reasoning_content deltas          # reasoning streams live
8    tool_calls: one complete call (id call_365afc69aa67..., name get_status)
9    finish_reason "tool_calls"
```

Zero content deltas, one call atom at the end. This is the CURRENT deployed build
(main's overlay: guard active only because tools is present). The PR extends exactly this
suppression to models/requests without tools.

### 3.2 Guard inactive (no tools field)

Same payload minus tools: content streamed live (10 content deltas of 19 chunks,
" The sea stretches", " endlessly", ...). So today's stack streams plain prose fine; the
PR would break this class of traffic.

### 3.3 Tokenization reality — why naive delta-streaming is unsafe

Asked (no tools, thinking off) to reproduce a DSML block verbatim, deltas arrived as:

```
'<｜DSML｜ calls'        # closer bytes + 5 more chars in ONE token
'><｜DSML｜'             # closes prior envelope, opens next, SAME token
' invoke name'
'="x"></｜DSML｜ invoke'
```

Facts:
- one token/delta can contain closer bytes AND opener bytes of the next envelope;
- wrapper runs split across deltas arbitrarily (also `ictions ...<｜` seen live);
- a delta can contain 8+ bytes BEYOND the visible `ictions` prefix (spaced wrapper
  `<｜DSML｜ calls` plus chrome), so a projector must hold whole deltas, not
  slices; the bounded holdback (longest opener prefix ~40 bytes, measured) is safe
  precisely because those tokens exist;
- `StreamDecoder.add` (`tensorfold/cuda/reply_text.py`) already yields exact
  byte-accurate accumulated text (deferred incomplete chars); the projector feeds on
  that string, not on raw deltas.
- the legacy path already proves the pattern safe in production: `hide_tool_calls`
  cuts from DSML open to close (buffering the call), and `_partial_tag` holds back
  opener fragments while streaming. The hybrid generalizes this behind the strict
  validator.

### 3.4 Cost of buffering is small (measured)

Live DSML envelope capture: the whole spaced DSML calls block decoded and hit the wire
in a handful of adjacent deltas; a full envelope (open+invoke+two params+close) is a few
rounds. Per-envelope buffering therefore costs about one decode round of local delay
(client-visible TTFC delta only), nothing more, because tokens stream back-to-back
inside an envelope; no client round-trip happens mid-buffer.

## 4. Design: the streaming projector (ToolOutputStream)

New overlay file (container path `tensorfold.cuda.tool_stream.py`), ~250 lines,
stdlib-only. It is a projector of the PR guard's own classification — never a second
parser: it reuses the guard's regexes/scan helpers to decide the same boundaries and
delegates the final result to `guard.parse`.

### 4.1 Contract and invariants

Wire-level behavior per stream request (strict chat + tools AND no-tools):

1. Reasoning (`reasoning_content`) streams exactly as today. Unchanged.
2. Answer content streams as a monotone prefix of the eventual guard-validated content.
   Prose between envelopes streams live.
3. At the FIRST byte of an executable-context envelope opener (after holdback
   resolution), the projector enters BUFFER mode: nothing flows until the envelope
   closes. Closing markers:
   - `tool_call`: kept until either a decodable JSON body + `</tool_call>` close, or
     the guard's fallback rule finds a later close (matching `parse` exactly),
   - `</｜DSML｜ calls>` / `</｜DSML｜tool_calls>`, `</function>`, GLM close.
4. At close: strict validate via the guard's logic on the buffered span:
   - valid -> release one complete atom (id, name, arguments) as a `tool_calls`
     delta; index = its position in the final calls list; resume prose scanning
     AFTER the envelope (the PR invariant "no state originates in call arguments").
   - invalid -> record terminal `ModelOutputError`; stop publishing; the finish path
     re-raises via `guard.parse` -> existing SSE error frame + `[DONE]`.
5. EOS inside a buffer: if the buffer can still only ever classify as
   literal-or-content, flush it as content (unclosed fences and quoted tails: the PR
   flushes them too). If it has an executable interpretation pending (opener without
   close), the guard's truncated-call rule classifies it rejected at finish
   (`truncated_call` / `malformed_dsml`): the error frame is the contract.

Invariants asserted by the replay harness (section 8, I-1..I-6):

- I-1 namespace: published content bytes are the guard's content prefixes; never call
  bytes (atom release replaces the removed envelope span).
- I-2 monotonicity: `published` never shrinks across feeds (server `sent` counter
  requirement).
- I-3 atom-prefix integrity: released calls == final `guard.parse()[1][:k]` in order;
  final `calls_streamed == k`; the finish chunk releases only calls[k:] with correct
  OpenAI indexes (http.py's existing `i < calls_streamed` skip needs NO change).
- I-4 prefix-safety: any settle-as-content byte range provably can't be reclassified
  by any continuation (all branch decisions derived from the guard's own scanner).
- I-5 reject parity: if the full parse raises, the projector either raised earlier at
  the same envelope or the fault is reachable only through finish state (then finish
  raises; SSE error frame still fires).
- I-6 finish authority: `guard.parse(raw_answer)` decisive; the projector PARSES and
  releases its atoms independently, but the authoritative final content/calls list
  (and its ids) at finish must exactly equal what was streamed (replay asserts).

### 4.2 Envelope buffering states

```
STREAM mode (prose):
  publish {safe_prefix(rest)}         # bounded holdback, see 4.3
  first executable-context marker -> BUFFER
BUFFER mode (envelope in flight):
  capture ONLY bytes from opener start (holdback includes chrome prefix like
  "tions calls"), advance until the guard's own close rule resolves:
    - tool_call + "{"  body: _DECODER.raw_decode from opener; guard's close-match
      rule after a successful raw_decode; fallback re-search if garbage (mirror of
      the guard's two-step logic, byte-for-byte);
    - tool_call non-JSON body / Qwen XML / GLM: scan the same close regex the guard
      uses; NOT the first array-close if the PR's scan would pick a later one;
    - DSML: whole calls block until '</｜DSML｜ calls>' / '</｜DSML｜tool_calls>'.
  on close: run the EXACT guard per-envelope validation (offer, choice, schema, IDs)
    -> STREAM (atom released) or REJECT recorded.
  EOS without close -> re-establish guard classification on the accumulated text
    (identical result by construction).
```

### 4.3 Holdback mechanics (per feed, all bounded)

- e1 partial-marker suffix: a delta tail that is a prefix of any marker/set needs
  >1 delta to classify. Hold it (at most the longest opener prefix ~26-40 bytes,
  measured incl. wrapper forms) and re-scan on the next delta. Live evidence
  (section 3.3) shows these tokens are common enough that the bound is required.
- e0 trailing-whitespace: the eventual `parse` result `strip()`s only at whole-answer
  level; mid-text trailing spaces settle differently only if a container opens after
  them. Track a whitespace-trail boundary; flush when the next non-ws byte arrives
  or the whole answer ends.
- e0g quoted-literal-example pending (the `_opening_quote` decision): while a line
  keeps an odd quote count, the guard may still flip it to a quoted literal example
  at line end. Hold from the quote-open to line end. Bounded by one line.
- envelope classifier pending (the `_next_marker` + `_quoted_end` interplay): when
  the guard's analysis at close-time uses a FUTURE byte, the byte range from marker
  to that byte is held. Ends when the guard's own rule resolves.

Everything else streams immediately. Measured live: the longest holdback NEVER
exceeded one line (est. during the conference re-derivation; harness asserts it).

### 4.4 Why this keeps the PR's validation strength whole

No new parser is written. The projector re-exports the guard's OWN classification:
`_next_marker`, `_opening_quote`, `_quoted_end`, `_block`, `_call`, `_parameters`,
`_schema`, `_pairs`, `_DECODER`, `_invoke`, `_dsparam` etc. are consumed as-is from
`tool_output.py`. Divergence is then structurally impossible; the harness only needs
to prove the projector did not publish a byte range the guard would NOT have put into
`content`, and that all held state resolves to a guard result at finish identical to
a non-streaming run.

## 5. Reasoning-vs-content emphasis (streaming UX)

Today: reasoning streams, content = one burst. Hybrid: reasoning streams; content
streams with pauses only across envelopes. Reference latency ordering: at proto live
times the envelope buffer drains within the NEXT decode round; end users see:

- plain prose answers: no change vs no-tools today (the PR would have stalled them);
- tool-call turns: prose-before-call streams; code path still instantiates the
  guard; the ONE burst shrinks to "the remaining tail".

Emphasis derived from `server.py::run()` analysis: sent counters and tail slicing
need only one guard-specific amendment (the projector holds its own
streamed-prefix position), everything else is shape-preserving.

## 6. Integration plan (server-side wiring)

### 6.1 New wiring (overlay `tool_stream.py`, plus apply.py anchors)

Module surface:

```python
class ToolOutputStream:
    def __init__(self, guard, request, *, tools): ...
    def feed(self, answer: str) -> str:            # prefix published so far
    def finish(self, raw_answer, max_calls): ...   # -> (content, calls, calls_streamed)
    def terminal(self): ...                        # ModelOutputError if rejected
```

server.py `run()` changes (all done via `apply.py` anchored replaces, exactly the
pattern apply.py already uses):

1. After the guard is built, also build the projector (only when the guard is not
   None and policy.allows_streaming — see 6.2):

```python
tool_output_stream = (ToolOutputStream(tool_output_guard, tools, request=body)
                      if tool_output_guard is not None and policy.max_calls != 1
                      and request_allows_streaming(body) else None)
```

2. In `visible()`: replace the PR's guard branch with:

```python
if tool_output_stream is not None:
    return reasoning, tool_output_stream.feed(answer)
if tool_output_guard is not None:
    return reasoning, ""                # whole-answer fallback mode (see 6.2)
```

Note the YIELD contract stays intact because `answer` returned by `visible()` is
ALREADY the streamed-prefix (as opposed to a fresh `""` each time); the existing
`shown = answer.rstrip() if tools else answer` + `delta["content"] = shown[sent:]`
delta slicing then emits exactly the newly certified bytes. The `rstrip` question:
rstripping the streamed prefix would desync `sent` from held positions; the
projector must therefore hand back the exact prefix INCLUDING the trailing
whitespace-trail holdback it uses internally (e0/e0g from 4.3), and simply keep
`shown = answer` with no rstrip change — the same guarantee the no-tools path has
today (the upstream no-tools path has always streamed unstripped; only the with-tools
branch rstripped).

3. At finish: the projector owns the authoritative results once it was active:

```python
if tool_output_stream is not None:
    content, calls, calls_streamed = tool_output_stream.finish(raw_answer, max_calls=policy.max_calls)
elif tool_output_guard is not None:
    content, calls = tool_output_guard.parse(raw_answer, max_calls=policy.max_calls)
    calls_streamed = 0
else:
    ...                                    # legacy, unchanged
```

`ToolCallStreamer` stays disabled (per the PR) — its argument-delta granularity is
incompatible with fail-closed validation; one-atom-after-close replaces it.

4. `calls_streamed` reuses the field http.py ALREADY consumes:
   `if i < result.get("calls_streamed", 0): continue`. The Projector therefore must
   release its atoms to the wire itself while streaming (a `tool_atom_events` callback
   into `visible()`'s emit path OR the projector returns both a content prefix AND
   queued atom events; two options:

   - O1 (projection-into-delta): projector emits atoms itself by pushing them into a
     queue that `on_tokens` drains right after content deltas and BEFORE finish.
     Strict ordering guarantee: reasoning < content-prefix < atom < next content.
   - O2 (batch at finish): atoms only in the `calls` list; `calls_streamed` stays 0.
     Simpler, but the call atom then arrives one round later than needed (a full round
     of latency, about nothing on real traffic since the finish chunk immediately follows
     the last envelope in the common one-call case).

   Recommendation: **O1**. Zero wire-shape change, ordering guaranteed, the
   finish path stays exactly the PR's (only the counter changes).

   Ordering detail (I-3): atoms are released in answer order and each atom released
   BEFORE the next feed() call scans the next envelope, but relative to content the
   rule is: content-prefix(strictly up to envelope start) precedes its atom, and the
   atom precedes any content AFTER the envelope.

5. http.py SSE adapter: with O1 no changes at all. (`calls_streamed` semantics
   preserved; `x-should-retry: false` (PR) untouched; `_stream_error` path unchanged.)
   If the repo later adopts O2, the only adapter change is `calls_streamed` handling
   already present upstream.

### 6.2 The policy fallback (critical semantic correctness)

Streaming can NOT satisfy 3 request shapes under a fail-closed, atomic contract:
(1) `parallel_tool_calls: false`, (2) `tool_choice: {function: name}` / forced modes,
(3) `policy.max_calls == 1` (a second call later in the answer invalidates a first
already released — retraction is impossible on SSE).

For these, keep the PR's whole-answer buffered mode (guard branch above). This
preserves the request's intent exactly and is what the shipped code already does;
it only stops being the default.

`request_allows_streaming(body)`: true for default OpenCode-agent style calls
(auto choice, parallel on/absent), false for the three shapes above. Both flags are
derived exactly from `ToolCallPolicy`'s existing fields (`single`, `max_calls`) plus
one reading of `body.get("parallel_tool_calls")`, so the decision matrix stays in ONE
place (`ToolOutputStream.FORCED_BUFFERED(request, policy)` classmethod; documented
in the overlay and test-matrix'd).

### 6.3 Fallback mode == PR's whole-answer mode (control for the rollout)

The finish-path fallback (`elif tool_output_guard is not None: parse(...)`) is the
PR's raw shape and its SSE behavior (error frame only, nothing streamed). This is
useful as an instant rollback lever (`TF_TOOL_OUTPUT_STREAMING=0` env or skip the
`tool_output_stream` construction entirely) — document in
`docs/TOOL-OUTPUT.md` + `docs/OPERATIONS.md`.

## 7. Latency / metrics expectations

(covered in section 5 — kept short here.) The per-envelope pause shows up only in
`ttft`-to-first-atom latency; TTFT of the first content delta is unchanged. The
`tokens` totals, `finish` shapes, and the `tensorfold` stats block all stay identical
to the PR's values because the final parse is untouched. `answer_raw` and the
`calls_stream` variable names persist for legacy visibility.

## 8. Tests

New file `experiments/display-kv/tests/test_tool_output_stream.py` mirroring the
existing test style (`TF_DISPLAY_SOURCE`-anchored overlay imports):

1. Synthetic fixtures = main's test suite cases PLUS these, replayed through
   `feed()` deltas AND at once:
   - plain prose; prose + offered tool_call JSON/XML/GLM/DSML envelope (spaced and
     unspaced DSML); unoffered call; malformed `tool_call` bodies; truncated
     `tool_call` at EOS; DSML with the invoke-nested-in-JSON-string literal case;
     DSML parameter-literal (existing test); `>`-quoted line copy of a malformed
     envelope; fenced literal before/after a real call; nested JSON closing-tag
     inside string argument; multiple sequential envelopes with prose between.
2. I-1..I-6 replay: for every fixture, feed `t[:k]` for k=0..len(t) step 1 AND the
   tokenized fixture splits; at each k assert
   `published == guard.parse(t)[0][:len(published)]` and
   all released atoms are IDENTICAL (name, arguments bytes, id string) to
   `guard.parse(t)[1][:k']` slices, `k' == len(released so far)`.
3. Live capture fixtures (from section 3 probes, incl. the delta transcripts in 3.3)
   replayed delta-by-delta; assert settle boundaries at the measured offsets.
4. Policy gate test: the three forced-buffered shapes from 6.2 keep `feed()` inert
   (empty prefix behavior — exactly the PR mode) and `finish()` delegate.

## 9. Risks and open items

1. O1 atom ordering needs an `emit` hook inside/on_visible path; make sure the
   queue drains BEFOREthe next `visible(False)` feed, else the wire ordering
   (atom vs following content) can flip. Test asserts it.
2. `sent` bookkeeping: the change to `shown = answer` (drop `rstrip` on guard path)
   has to be validated against thinking separation (`answer` handed to
   `visible()` was always pre-split; no regression for no-tools; with-tools the
   projector's own e0-trail is the new strip authority: document).
3. The projector must never see `answer` text that `split_thinking` later folds
   differently (it only ever feeds the ANSWER part, as today).
4. Cancellation mid-buffer: client disconnect while in BUFFER mode is the same
   RequestCancelled path as today; nothing new.
5. When `guard.parse` at finish REJECTS while earlier atoms were released: already
   validated call #1 IS already on the wire. Per-envelope semantics accept this
   (user-endorsed); document in docs/TOOL-OUTPUT.md ("per-envelope atomicity;
   earlier released atoms stand").
6. `x-should-retry: false` (PR) + `ModelOutputError` already wire the terminal
   frame; nothing else needed in http.py.
7. PR content passes schema-strict `active_tool_specs`; the projector uses
   exactly `tool_output_guard.tools` (same list) so no drift.
8. Open follow-up (out of scope): the PR extends strict validation to no-tools
   requests; if the team wants the old no-tools behavior back (truly raw content
   streaming with no guard), add `strict_tool_output = False` handling for
   `body` without tools — currently the PR wiring makes the guard active, which is
   what keys the projector into rejecting envelopes that appear in prose-only
   replies. Decide whether that strictness is desired; it is the PR's stance; keep.

## 10. Evidence appendix

### 10.1 Live deltas, with tools (today's deployed behavior)

```
delta keys: ['role'] | rc repr: None | content repr: None | tool_calls: None | finish: None
delta keys: ['reasoning_content'] | rc repr: 'The' | ...         (6 reasoning deltas)
delta keys: ['tool_calls'] | rc repr: None | tool_calls: [{"index": 0, "id": "call_365afc69aa674cfe93e137bb", "type": "function", "function": {"name": "get_status", "arguments": ...}]}]
delta keys: [] | finish: tool_calls
```

### 10.2 Live deltas, no tools (plain prose)

```
deltas with content: 10 / 19 chunks
first few: ['The sea stretches', ' endlessly', ' under', ' the', ' sky, its']
```

### 10.3 Live deltas, raw DSML reproduction (tokenization evidence)

```
'<｜DSML｜ calls'
'><｜DSML｜'
' invoke name'
'="x"></｜DSML｜ invoke'</｜DSML｜ invoke'
None
```

Note the fused `><｜` case (close+opener in one token) and the `="x"></｜DSML｜ invoke`
token (argument + closer + opener in one delta). These are the exact patterns the
holdback boundary handles; the replay harness encodes them byte-for-byte.

### 10.4 Guard behavior verified against PR module copy

```
guard.parse('look `x` <｜DSML｜ calls><｜DSML｜ invoke name="get_status"></｜DSML｜ invoke></｜DSML｜ calls> tail')
-> ('look `x`  tail', [{'id': 'call_00b2...', 'function': {'name': 'get_status', ...}}])
guard.parse('<｜DSML｜ calls')   -> ModelOutputError, reason=invalid_executable_tool_output
```

(Unclosed opener at EOS rejects; confirms the projector needs no special
"unfinished envelope at finish" handling beyond delegating to the guard.)

### 10.5 Where the artifacts were

```
/tmp/opencode/dsml-stream/tool_output_strict.py       # PR guard module copy
/tmp/opencode/dsml-stream/stream_projector.py         # prototype skeleton (partial)
/tmp/opencode/tfv41-display-source                    # pristine pinned TF source
/tmp/opencode/server_patched.py / http_patched.py     # installed-source renderings
```

Scratch is volatile; this document + the section-6 anchors are the durable record.

## 11. Recommendation

1. Merge the PR's parser hardening (strict validator + schema + CPU runner): YES,
   it is the foundation.
2. Apply the streaming projector (section 4-6) as a follow-up patch on the same
   branch before merging to main: streaming without losing any validation strength;
   default-path policy presets unchanged.
3. Keep the PR's whole-answer mode for the three forced-buffered request shapes
   (section 6.2) — those semantics genuinely need answer-wide evidence.
4. Keep `TF_TOOL_OUTPUT_STREAMING=0` as a one-line operational rollback to the PR's
   buffered behavior.

## 12. Continuation on 2026-10-21: conservative prose projection

### 12.1 Upstream review at 06:40 UTC

- Recipe upstream `main` is `1b3066d`, after PR #35. It now pins TensorFold
  `730d41abd8f4` (`dsv41-1.1.0`) and defaults `DROP_UNOFFERED_TOOL_CALLS=1`.
  **Dropping** a call is not this fork's explicit terminal-error contract.
- Upstream's display-KV, cache-metrics and memory/top-k experiments still require
  pristine `07aa02585f64`. Changing this fork's engine pin without porting and
  validating the anchored transforms would be incorrect. No engine upgrade or
  blanket recipe merge is part of this parser change.
- Upstream's still-open tool-output PR #32 is at `879b19d`. Its useful ideas are
  raw-opener prose projection, engine/guard classification parity, safe JSON
  decoder failure handling and source-span mapping. It still delegates to the
  permissive engine parser and does not replace the local strict/schema work.
- The strict foundation described earlier was a **local** branch,
  `fix/streaming-dsml-guard` at `d719bca`, not the current upstream PR #32.
  Its two commits (`c65e56f`, `d719bca`) have been merged locally. Existing
  display/spill/memory behavior and the engine input pin are preserved.

Authoritative comparison sources: recipe PRs
[#35](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/pull/35)
and [#32](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/pull/32),
plus fetched commit trees. No upstream changes have been pushed.

### 12.2 Counterexamples to the original early-atom proposal

Let `CALL` be an offered, schema-valid complete `<tool_call>` block.
Durable reproductions are in `tests/test_tool_output_stream.py` under the
display-KV experiment.

1. ``'before `' + CALL`` is executable at EOS, but appending a matching backtick
   several uninterrupted prose lines later makes it literal. The unresolved
   interval is **not bounded to one line**.
2. ``'before `' + CALL + '`'`` is literal at EOS, but appending another backtick
   lengthens that closing run and makes the earlier call executable. A closing
   delimiter at the current end is provisional.
3. `'"' + CALL` is executable at EOS, but appending `\n\n"` makes it literal:
   `_quoted_end` uses `lstrip()` across lines. Line-only holdback is insufficient.
4. `<tool_call>{"name":"task","arguments":{"x":"</tool_call>` rejects as a
   finished answer, but appending ` literal"}}</tool_call>` makes a valid call.
   A JSON decode failure plus the fallback closer is **not** a monotone rejection
   proof. The closer can still be string data in an unfinished JSON value.
5. Generated IDs differ across parser invocations. Exact early/final call equality
   would require stable request-local identity allocation. A later explicit
   duplicate ID can also invalidate calls already released.
6. The strict marker detector includes arbitrary namespace prefixes. A fixed
   26-40 character opener buffer is not sufficient for every recognized marker.

Reusing private regexes does not itself prove prefix stability. The original
claim that divergence is structurally impossible is withdrawn. These cases must
remain regression gates for any later full projector.

### 12.3 Implemented phase 1

- `ToolOutputGuard.stream(cumulative_answer)` returns a monotone, unmodified
  prose prefix. It never invokes `parse`, decodes an envelope or allocates IDs.
- Scan **raw** `_marker` candidates, including candidates in quotes and fences.
  Hold incomplete spellings, including namespace prefixes. At the first candidate
  freeze the rest of the answer until the one authoritative final parse.
- A possible prefix longer than 256 characters also freezes at its `<`, to bound
  repeated scans. This is a conservative latency fallback, not rejection,
  truncation or an output-size limit. Ordinary HTML and comparisons release when
  they can no longer grow into a candidate.
- No-tools prose and tool-enabled prose without candidates stream normally.
  Reasoning is split first and streams independently, using the existing engine
  semantics. Guard-path whitespace is preserved without the legacy `rstrip()`.
- All executable calls remain whole-answer atomic, including forced, named,
  `parallel_tool_calls:false`, call-count and duplicate-ID policies. No separate
  forced-mode projector gate is necessary while **all calls** wait until finish.
- A late failure can leave inert prose already delivered. No call, buffered
  envelope or successful finish is delivered. The existing single SSE error and
  `[DONE]` contract remains. Nonstreaming failure still returns only HTTP 500
  plus the typed error and `x-should-retry:false`.
- Decoder rewrites or final content inconsistent with the published prefix fail
  closed as `reason=stream_prefix_changed`, rather than silently dropping a tail.
- Set `TOOL_OUTPUT_STREAMING=0` in `.env` and restart the display-KV launcher to
  restore whole-answer content buffering. It forwards `TF_TOOL_OUTPUT_STREAMING`
  to both ranks. This does **not** disable strict validation. Invalid launcher
  values refuse before launch. The ordinary image/launcher are unchanged.

### 12.4 Gates and remaining work

- The real staged installer must pass normal and `-O` parser/schema/replay/HTTP
  regressions, including wrong-pin/drift/reapply refusal without writes.
- HTTP tests must observe prose while generation is blocked behind a real
  barrier, both with and without tools. Recovery tests reassemble SSE content,
  not assume that content appears in one frame.
- Keep atomic late rejection, literal preservation, partial markers across every
  character/two-chunk split, fused DSML deltas, Unicode namespaces, bounded
  pending scans, rollback, decoder rewrites, cancellation and request isolation.
- Save a model-backed baseline, build only after both ranks stop, then use
  `./start-display-kv.sh --build`. Run `./verify.sh` and the opt-in
  `tests/live_tool_streaming.py`. A healthy endpoint alone does not prove TP=2.
  A model refusal or failure to induce a fixture remains **inconclusive**.
- Do not stop a healthy stack until the head's sudo authorization needed for
  DRM preparation is available. The current image identity is saved before any
  cutover. No `docker exec`, display-manager stop or second model server is used.
- Future full hybrid: first build a shared prefix-stable literal/envelope scanner
  with explicit pending states, prove JSON closer ownership, settle ID/global
  policy semantics, then test same-feed event ordering. Do not release atoms
  merely because `parse(current_prefix)` succeeds or rejects.

Run records and current deployment status belong under
`runs/2026-10-21/parser-guardrails/`, not in claims copied from old live runs.
