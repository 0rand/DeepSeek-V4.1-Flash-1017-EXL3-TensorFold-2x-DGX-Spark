# Runs of 21 October 2026 (UTC): release gate of engine `dsv41-1.1.0`

## Parser and proxy consistency audit

- [Parser and MediaLLMProxy audit](parser-proxy-audit/README.md): display image `tensorfold-dsv41:display-kv-0dc9c3fefb06`, strict guard and streaming enabled, proxy `StrictNative`, temperature 0.4. Real inference 5/5, installed parser 70 tests normal and 70 under `-O`, proxy 287/287. Functional A/B confirms strict-native literal preservation and reproduces missing SSE terminal framing on the proxy loop-guard path. No service or production config changes were made by the audit.

Same pair of DGX Spark as `runs/2026-10-18/`. The gate ran in a lock window from 02:17 to 02:49 UTC (the node clock). Each step started a fresh engine, then ran the same checks in the same order, with the whole suite sent straight to the engine (`v41_suite.py`):

1. start, then a structured-output smoke test;
2. `v41_suite.py` (32 tests);
3. `v41_grammar_check.py 2` (14 checks);
4. `v41_bench.py` (1 to 4 streams, prefix cache);
5. a long run: 4 streams at the same time with prompts of 20K, 50K, 100K and 200K tokens, 1,500 new tokens each, with `MemAvailable` read each second on both nodes.

Settings, equal in both steps (production settings of the pair): `CONTEXT=1048576`, `PARALLEL=4`, `SPILL_GIB=64` (a fresh spill folder in each step), decode graphs on, no restart inside a step. The gate report does not list the other settings. The data are one pass each.

## Release gate of engine 1.1.0

| Step | Engine image | Engine source |
|---|---|---|
| Baseline (1.0.x production image) | `tensorfold-dsv41:3c45bdaeab7d` | the 1.0.0 code, `07aa02585f64`, plus the prompt indexer top-k change (fork PR #7) |
| 1.1.0 | `tensorfold-dsv41:fc9200a44463` | the same git tree as `730d41abd8f4`, the commit of release [dsv41-1.1.0](https://github.com/Elevatormusic/TensorFold/releases/tag/dsv41-1.1.0) that the recipe now pins |

The 1.1.0 step ran the image `fc9200a44463`, the head of fork PR #24. Release commit `730d41abd8f4` is the merge of that pull request and has an identical git tree, so the engine code is the same. The tool-call check for `TF_DROP_UNOFFERED_TOOL_CALLS=1` ran in an earlier window on the candidate `e2b282c50724` (see section 2 of the gate report).

| Check | Baseline `3c45bdaeab7d` | 1.1.0 `fc9200a44463` |
|---|---|---|
| Suite (`v41_suite.py`) | 32/32 | 32/32 |
| Grammar check, `--parallel` 2 | 14/14 | 14/14 |
| Decode, 1 stream, prose / code | 52.7 / 100.4 tok/s | 51.9 / 99.7 tok/s |
| Decode, 4 streams, prose, per stream / aggregate | 25.3 / 96.1 tok/s | 26.8 / 99.3 tok/s (+5.9% / +3.3%) |
| Decode, 4 streams, code, per stream / aggregate | 37.1 / 141.5 tok/s | 39.7 / 145.7 tok/s (+7.0% / +3.0%) |
| Prefill, 32K prompt | 2,717 tok/s, TTFT 11.78 s | 2,751 tok/s, TTFT 11.63 s |
| Prefix cache, 24K prompt, cold / warm TTFT | 11.43 s / 1.10 s | 11.18 s / 1.05 s |
| Long run, errors | 0 | 0 |
| Long run, lowest head `MemAvailable` | 1,654 MiB | 1,450 MiB |
| Long run, lowest worker `MemAvailable` | 2,146 MiB | 2,507 MiB |
| Late graph captures while serving | not counted | 0 |
| earlyoom events | none | none |

- The token hashes of the 4 long prompts are the same in both steps (`d58f653b2ce8`, `f42ae2e092ab`, `0ed6e9e70041`, `0d2a3aa664b6`).
- The 1-stream decode difference is inside the run-to-run noise. The 4-stream gain is mostly from fork PR #20 (drafts of all streams ranked in one round). The image also holds fork PRs #8 and #24, so the gain is not from #20 alone.
- The long runs started with different head `MemAvailable` (2,113 MiB for the baseline and 3,190 MiB for 1.1.0), so the two lowest values are not strictly comparable. The head margin against the 1 GiB earlyoom limit is small in both steps. A first window on the candidate `e2b282c50724` (no admission fix yet) had one long run in which earlyoom stopped the head engine; three repeats and both steps of this window did not. The gate report gives the details. The admission fix (fork PR #24) was added after that run.
- The prefill and decode rows are single passes. They are release-gate checks, not an A/B with a median of 3 runs.

Files:

| File | Content |
|---|---|
| `gate-1.1.0/gate-report.md` | the gate report (host names, home paths and router details removed) |
| `gate-1.1.0/data/b-*` | baseline step: suite, grammar check, bench, long run, lowest memory, earlyoom check |
| `gate-1.1.0/data/f-*` | 1.1.0 step: the same files |
| `gate-1.1.0/data/idle-b.log`, `idle-f.log` | idle `MemAvailable` before the start, after the start and before the long run |
| `gate-1.1.0/data/mem-b-full-*`, `mem-f-full-*` | `MemAvailable` in kB each second, head and worker, over the whole step |

## Parser isolation and conservative prose streaming

[Parser guardrails record](parser-guardrails/README.md) documents the upstream
review, integrated strict-validator foundation, conservative prose projector,
installed-source CPU checks and old-image model-backed baseline.

The code is deployed as `tensorfold-dsv41:display-kv-58e319cb2866` on both nodes
and live-validated: `./verify.sh` 5/5, parser probe 12 PASS / 2 INCONCLUSIVE /
0 FAIL. The two inconclusive cases are fixture-induction limits, not parser
failures. The `TOOL_OUTPUT_STREAMING=0` rollback was also validated live with a
negative control, and the streaming default was restored and re-validated.

## Display-KV unified 1.0 serving run

- Serving image: `tensorfold-dsv41:display-kv-0dc9c3fefb06` (ID `4dd9d9a7f234`),
  built from the 1.0.0 base `tensorfold-dsv41:07aa02585f64`.
- Run files: `runall-07aa02585f64-083516Z-*`. The filename uses the root
  `.env` `IMAGE` tag; the serving container was the display variant above.
- `./verify.sh`: 5/5. `v41_suite.py`: 30/30 (the 2 router tests were skipped
  without a router key). `v41_grammar_check.py`: 14/14.
- `v41_conformance.py`: 3/5. The 3 API-path cases pass. The 2 raw-template
  cases fail on special-token/`<action>` rendering differences, the documented
  diagnostic cases; the API serving path passes. See
  [What I changed](../../docs/WHAT-I-CHANGED.md) (3/3 API roles).
- Differential, bench and Hermes were skipped (`ENCODING_DIR`, `BENCH`,
  `HERMES` not set).
- The suite log's repository path is sanitized to `<repo>`.

## Proxy mitigation source staging

[Proxy mitigation staging record](proxy-mitigation-staging/README.md): work
packages 0, 2, 3, 4A, 4C and the stage-only part of 4B. Proxy commit
`13b8561` passed 308/308 unit tests, 25/25 control tests in normal and optimized
Python, 36/36 synthetic route cases and `./verify.sh` 5/5. Proxy A now runs that
DLL with wedge and corrective retry enabled. Live direct/proxy checks had 12
PASS, 2 model-induction INCONCLUSIVE and 0 FAIL. The retry case preserved the
typed rejection after one retry. A controlled NUDGE exercised the wedge and
produced a banner, one finish and one DONE. The engine and rank image did not
change. The safe profile remains a rollback fallback. SDK/original-agent, STT,
recovery-policy and rollback-drill acceptance remain open. No generated tool was
executed.
