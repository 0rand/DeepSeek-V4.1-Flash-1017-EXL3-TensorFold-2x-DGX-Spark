# TensorFold dsv41-1.1.0 release candidate: gate report

Date: 2026-10-20 (UTC 00:05 to 00:38 on 2026-10-21 on the nodes).
Candidate: fork branch `deepseek-v41-tp2`, commit `e2b282c50724` (0c7614c4938d plus fork PR #20).
Image: `tensorfold-dsv41:e2b282c50724`, image ID `901c0199dcd2`, the same ID on the head and the worker.
The first candidate, `0c7614c4938d`, was also built (image ID `e2e2f4b1207a`, both nodes). It was NOT gated.
Nothing was tagged, published, merged or pushed. The production unit `dgx-v41-1017-serving` was not touched and stays down.

## UPDATE: final candidate and window 2 (read this first)

- Final 1.1.0 candidate: `deepseek-v41-tp2` at `730d41abd8f4eb1745f3cef0a2fc9d6ba6039c51` (merge of fork PR #24). Its git tree (`5ef672fac2d0`) is identical to the tree of `fc9200a444635815c6b90ddd7e9b1ca70c6a2dab` (PR #24 head). So the image `tensorfold-dsv41:fc9200a44463` (image ID `bbf4f4485278`, same on both nodes, git HEAD `fc9200a4...`) IS the candidate. It includes fork PR #8 (dense overrides) and PR #24 (admission gate counts none of the cache while the baseline is unknown) on top of `e2b282c50724`.
- Window 2 (2026-10-21 02:17 to 02:49 UTC) ran the whole suite straight to the engine on the NORMAL model view `deepseek-v41-1017-exl3-2.9bpw-afc7aa0a`, an engine URL on the head (requests from the head itself), for both images. The router was not used and its config was not changed.
- Result: step 1 baseline `3c45bdaeab7d` and step 2 candidate `fc9200a44463` both pass 32 of 32, grammar 14 of 14, and both survive the long 4-stream run in the run-A order (no restart). No earlyoom event. See section 8.
- The earlier sections (1 to 7) describe window 1 on `e2b282c50724` and are kept as they were.

## Result in one line (window 1, e2b282c50724)

32 of 32 tests pass in effect (30 of 32 through the router, which points at a different stack; the 2 router tests pass 2 of 2 when sent straight to the engine).
Grammar check 14 of 14. Structured-output smoke passes. Graphs-off suite: same 30 of 32 (same 2 router tests).
Zero "late capture" warnings and zero WARNING lines in all rank logs.
ONE PROBLEM: in one run, earlyoom stopped the head engine during the long 4-stream test (details below). Three repeats did not fail.

## 1. Build

- Method: the recipe Dockerfile with `TF_COMMIT` build argument, built on the head, then `docker save | ssh worker docker load`. This is the same method as the 3c45bdaeab7d build.
- The head and the worker image IDs match (`901c0199dcd2`). `BUILD-OK`.
- The image has `git HEAD` = `e2b282c50724a51182ae9e1545efe8468e41c599`.
- Pip chose `transformers 5.20.2` in this build. The 3c45bdaeab7d image has `5.20.1`. All other pinned packages are equal (`xgrammar 0.2.8`, `huggingface-hub 1.33.0`, same NGC torch).
- The CUDA extensions were built before the weights loaded (one `docker run` per node, in parallel, about 2.3 minutes). They went to their own directory `torch_extensions-e2b282c50724` (82 MiB), so the production extension build was not changed.
- The candidate used its own spill directory `tf-ds-spill-rel110` (the content is owned by root and could not be removed by the user; it is almost empty). The rank cache, triton cache and `nv` cache are shared with production (they are keyed by content).

## 2. Gate checks (production settings, graphs on)

| Check | Result |
| --- | --- |
| Start (warm rank cache) | serving after 63 seconds |
| Structured-output smoke (json_schema and plain) | 200 and 200, pass |
| `v41_suite.py` | 30 of 32. The 2 failed tests are `router/tools_stream` and `router/default`: HTTP 500 |
| Router cause | The router in the test setup pointed to a different model stack, not to this engine. It answers "connection reset". It is not a candidate fault. |
| The same 2 tests sent straight to the engine | 2 of 2 pass (`v41_suite_direct.py router`, router URL replaced by the engine URL) |
| `v41_grammar_check.py 2` | 14 of 14 |
| Late-capture warnings (`captured while serving`) | 0 on r0, 0 on r1, after start, suite, grammar, bench and every long run |
| WARNING lines in the rank logs | 0 |
| Tool-call and image checks (`val_tests.py`, with `TF_DROP_UNOFFERED_TOOL_CALLS=1` and `--vision-max-images 32`) | T1 12 images: 200. T2 33 images: 400 "at most 32". T3 overlapping image requests: all 200, server alive. T4 unoffered tool call, with and without tools offered, streamed and not streamed (12 requests): all 200, no tool markup in the content, no tool call returned |
| Graphs off (`TF_DS_GRAPHS=0`), `v41_suite.py` | 30 of 32. Same 2 router tests fail with HTTP 500. All other tests pass. |

No `val_checks.py` exists in the repo. I used `val_tests.py` from `~/deploy/validate-pr13-15-20261020/` (it checks PR 13, 14 and 15 behaviour live). The repo also has the unit test `tests/test_unoffered_tool_calls.py`; I did not run it.

Note on the graphs-off prefill rows: 6,647 and 25,515 tok/s are cache hits (the prompts were in the spill from the earlier run). Do not use them.

## 3. Speed

### Suite rows (cold prompts, first run on a fresh spill directory)

| Row | 3c45bdaeab7d (suite-3c45bdaeab7d.log) | e2b282c50724 |
| --- | --- | --- |
| Decode prose, thinking off, tok/s | 57.5 | 57.7 |
| Decode code, thinking off, tok/s | 93.3 | 93.5 |
| Decode prose, thinking on, tok/s | 59.1 | 59.4 |
| Decode code, thinking on, tok/s | 76.9 | 77.2 |
| Prefill 8K, tok/s (TTFT) | 812 (9.86 s) | 1,658 (4.83 s) |
| Prefill 32K, tok/s (TTFT) | 2,717 (11.78 s) | 2,696 (11.87 s) |

The 8K row includes the first-request warm-up, so it is not a clean prefill figure. The 32K prefill is equal within 1 percent.

### Bench (`v41_bench.py`), candidate e2b282c50724

| Streams | Prose per stream (aggregate) tok/s | Code per stream (aggregate) tok/s | TTFT prose / code |
| --- | --- | --- | --- |
| 1 | 53.0 (52.5) | 100.7 (98.8) | 0.09 s / 0.10 s |
| 2 | 39.1 (75.8) | 63.2 (118.2) | 0.13 s / 0.15 s |
| 3 | 33.3 (91.6) | 48.2 (136.7) | 0.18 s / 0.21 s |
| 4 | 27.0 (100.1) | 40.0 (147.0) | 0.22 s / 0.26 s |

Prefix cache: cold 24K prompt TTFT 11.47 s (cached 0); same prompt again TTFT 1.17 s (cached 22,528 tokens); new 24K prompt TTFT 11.50 s.

### Comparison with 3c45bdaeab7d: LIMIT OF THIS REPORT

There is no `bench-results-3c45bdaeab7d.json` on the head. I did not run the bench on 3c45bdaeab7d in the same session. So the +3.0 to +6.9 percent claim of PR #20 at 4 streams is NOT confirmed here.
The closest older bench file is `bench-results-fd2365042f56.json` (17 Oct, before PRs 4 to 7 and before #20):

| 4 streams | fd2365042f56 (older) | e2b282c50724 | Change |
| --- | --- | --- | --- |
| Prose per stream tok/s | 24.9 | 27.0 | +8.4 percent |
| Prose aggregate tok/s | 94.4 | 100.1 | +6.0 percent |
| Code per stream tok/s | 36.8 | 40.0 | +8.7 percent |
| Code aggregate tok/s | 140.3 | 147.0 | +4.8 percent |

This change includes all fork work since fd2365042f56, not only PR #20. A same-session A/B of 3c45bdaeab7d and e2b282c50724 needs one more lock window (about 10 minutes of GPU time).

## 4. Long 4-stream run (20K, 50K, 100K and 200K token prompts, 1,500 new tokens each, `capdrive.py long`)

Memory was sampled from `/proc/meminfo` once every second on both nodes.

| Run | Server state before the run | Result | Lowest MemAvailable, head | Lowest MemAvailable, worker |
| --- | --- | --- | --- | --- |
| A (candidate) | after smoke, suite, grammar check and bench, no restart | FAILED: earlyoom sent SIGTERM to the head engine at 1,018 MiB; 0 tokens; engine and worker rank down | 1,030 MiB (22 s into the run) | 3,109 MiB |
| D1 (candidate) | fresh start | pass: 4 of 4 requests, 1,500 tokens each, 0 errors, 282 s | 1,505 MiB | 2,579 MiB |
| D3 (candidate, `SPILL_GIB=0`) | fresh start | pass, 0 errors, 262 s | 1,421 MiB | 2,829 MiB |
| E (3c45bdaeab7d baseline) | fresh start | pass, 0 errors, 264 s | 1,340 MiB | 2,269 MiB |

- Run A cause: the head had only 2,339 MiB free when the run started and it fell to 1,030 MiB in the first 22 seconds (the four prompts start together). The host earlyoom limit for SIGTERM is 1 GiB. The engine floor `TF_DS_MEM_FLOOR_GIB` is 1.5 GiB. The head log ends with `terminate called without an active exception`. The worker log shows `NCCL error 4` after that (a result of the head stop, not a separate fault).
- Fresh-start runs pass with 1.3 to 1.5 GiB of margin on the head. The result for the candidate (1,505 and 1,421 MiB) is not worse than the baseline (1,340 MiB).
- I did NOT run the baseline in the same sequence as run A (suite and bench first, then the long run). So I cannot say if 3c45bdaeab7d also fails in that state. Treat this as an open risk, not as a proven regression.
- Token hashes are identical in D1, D3 and E for all four prompts (`d58f653b2ce8`, `f42ae2e092ab`, `0ed6e9e70041`, `0d2a3aa664b6`). Decode per stream: 19.8, 30.3, 41.0, 114.4 tok/s (D1).
- Late-capture warnings in these runs: 0. WARNING lines: 0. Request errors: 0 (runs D1, D3, E).

## 5. Cleanup

- Both rank containers were stopped and removed after every window. No `rel110` container is left on the head or the worker.
- Both lock markers were removed. The shared GPU locks on both nodes were released.
- No `systemctl stop gdm`, no sysctl, driver or firmware change, no `docker exec`, no IBGDA code.

## 6. Log and file paths

The logs of window 1 stay on the head and are not in this folder. The summary tables above hold their numbers. The logs of window 2 are in [data/](data/) (see section 8). The rank logs are not saved here.

## 7. Next steps from window 1 (done in window 2)

Both steps were done in window 2 (section 8).

## 8. Window 2: baseline 3c45bdaeab7d and final candidate fc9200a44463, run-A order, normal model

Setup, equal for both steps: model view `deepseek-v41-1017-exl3-2.9bpw-afc7aa0a`; the whole `v41_suite.py` was sent straight to the engine (`v41_suite_direct.py`, an engine URL on the head, requests from the head itself); production settings (graphs on, spill 64 GiB, each step with its own fresh spill directory); no restart inside a step. Order in each step: start, smoke, suite, grammar check, bench, long 4-stream run (20K, 50K, 100K, 200K tokens, 1,500 new tokens each). MemAvailable was sampled every 1 s on both nodes. Step 2 had `TF_DS_MEMTRACE=/cache/memtrace/rel110b-fix` (allocator trace on; a path prefix, not `1`).

### Results

| Item | Step 1: 3c45bdaeab7d | Step 2: fc9200a44463 (tree = 730d41abd8f4) |
| --- | --- | --- |
| Suite, whole, direct to the engine | 32 of 32 | 32 of 32 |
| Grammar check | 14 of 14 | 14 of 14 |
| Idle MemAvailable before the start (head / worker) | 116,512 / 115,362 MiB | 113,838 / 112,687 MiB |
| MemAvailable after the start (head / worker) | 3,635 / 4,403 MiB | 2,941 / 3,669 MiB |
| MemAvailable at the start of the long run (head / worker) | 2,113 / 2,625 MiB | 3,190 / 3,759 MiB |
| Long run: errors | 0 | 0 |
| Long run: lowest MemAvailable, head | 1,654 MiB at 02:29:09 (3 min 45 s after the long run start 02:25:24) | 1,450 MiB at 02:47:39 (3 min 21 s after the start 02:44:18) |
| Long run: lowest MemAvailable, worker | 2,146 MiB at 02:28:19 | 2,507 MiB at 02:47:16 |
| earlyoom events (head and worker) | none | none |
| "admission waits" lines | 0 | 0 |
| Late-capture warnings / WARNING lines | 0 / 0 | 0 / 0 |
| Token hashes of the 4 long prompts | `d58f653b2ce8`, `f42ae2e092ab`, `0ed6e9e70041`, `0d2a3aa664b6` | the same 4 hashes |
| Decode tok/s per stream (20K, 50K, 100K, 200K) | 18.6, 28.2, 40.0, 113.6 | 19.0, 29.4, 40.1, 113.6 |
| TTFT (20K, 50K, 100K, 200K) | 9.3, 56.4, 132.0, 254.9 s | 9.1, 53.7, 126.6, 245.9 s |

- The memtrace of step 2 gives the same lowest value as the 1 s sampler: rank 0 1,450 MiB (phase `fill`), rank 1 2,507 MiB (phase `fill`). The CSV files are not saved here.
- Run A of window 1 (`e2b282c50724`, killed by earlyoom at 1,018 MiB) did NOT repeat in window 2 with the baseline or with the candidate. In run A the head had 2,339 MiB at the long run start and fell to 1,307 MiB in 1 s. In window 2 the low point came 3 to 4 minutes into the run, not in the first seconds. So run A is a marginal event: the head sits near the 1 GiB earlyoom limit. The fix did not need its admission wait in this window (0 lines), so this window does not show the fix working; it shows no regression.
- The head low point of the candidate (1,450 MiB) is 204 MiB lower than the baseline (1,654 MiB). The two long runs started with different MemAvailable (2,113 MiB and 3,190 MiB), so they are not strictly comparable.

### Same-session bench for the PR #20 claim (4 streams, step 1 vs step 2)

| Case | 3c45bdaeab7d | fc9200a44463 | Change |
| --- | --- | --- | --- |
| Prose x4, per stream tok/s | 25.3 | 26.8 | +5.9 percent |
| Prose x4, aggregate tok/s | 96.1 | 99.3 | +3.3 percent |
| Code x4, per stream tok/s | 37.1 | 39.7 | +7.0 percent |
| Code x4, aggregate tok/s | 141.5 | 145.7 | +3.0 percent |
| Prose x3, per stream (aggregate) | 31.4 (88.4) | 32.9 (90.6) | +4.8 (+2.5) percent |
| Code x3, per stream (aggregate) | 44.4 (127.6) | 47.9 (135.8) | +7.9 (+6.4) percent |
| Prose x2 / code x2, per stream | 37.8 / 62.1 | 38.5 / 62.7 | +1.9 / +1.0 percent |
| Prose x1 / code x1 decode tok/s | 52.7 / 100.4 | 51.9 / 99.7 | -1.5 / -0.7 percent |

- The 4-stream gain (+3.0 to +7.0 percent) is in the range that PR #20 claims (+3.0 to +6.9 percent). One stream is flat (within 1.5 percent, inside the run-to-run noise).
- The candidate also contains fork PR #8 and PR #24, so the gain is not from PR #20 alone. TTFT is unchanged (0.09 to 0.28 s at every level).
- Caution: the step 1 bench ran after the suite and the grammar check; the step 2 bench ran after the same steps in step 2 too (same order). Each run is one pass.
- Prefix cache: cold 24K prompt TTFT 11.43 s (step 1) and 11.18 s (step 2); warm 1.10 s and 1.05 s with 22,528 cached tokens; a new 24K prompt 11.51 s and 11.33 s.
- Suite speed rows (cold): decode prose / code, thinking off, 56.9 / 92.8 tok/s (step 1) and 56.4 / 92.2 (step 2); thinking on 58.7 / 76.4 and 58.5 / 76.2; prefill 32K 2,717 tok/s (TTFT 11.78 s) and 2,751 tok/s (TTFT 11.63 s); prefill 8K 1,499 and 1,824 tok/s (includes the first-request warm-up).

### Window 2 housekeeping

- The fix image was built on both nodes before the window with no lock (`build-fix.sh`, started when both nodes had 25 GiB or more free). Image ID `bbf4f4485278` on both nodes. The CUDA extensions were prebuilt under the lock in `torch_extensions-fc9200a44463` (about 2.4 minutes). The first start of the candidate took 5.5 minutes (cold extension and rank caches); the baseline start took 1 minute.
- Both rank containers were removed. Both lock markers (`rel110b.run` and `rel110b.held`) and the sampler markers were removed. No `rel110` container is left on either node. No `systemctl stop gdm`, no host change, no `docker exec`, no production unit change. Nothing was tagged or published.
- Logs of window 2 are in [data/](data/). Step 1 files have the prefix `b-` and step 2 files have the prefix `f-`: the `suite`, `grammar`, `bench`, `long4`, `long-minmem` and `earlyoom` logs, the suite and bench result files (`.json`), and the memory samples `mem-b-full-*` and `mem-f-full-*` (one line each second; `MemAvailable` in kB). `idle-b.log` and `idle-f.log` hold the idle memory values. One change from the original files: in the suite logs, the path of the result file is replaced by its name.

## 9. Verdict for the release

The final candidate (tree `5ef672fac2d0`, commit `730d41abd8f4`, tested as image `fc9200a44463`) passes 32 of 32, 14 of 14, has no late-capture warning, and survived the long 4-stream run in the run-A order with no earlyoom event. The baseline behaved the same. The run-A kill of window 1 did not repeat, but the head still has only about 1.4 to 1.7 GiB of margin against the 1 GiB earlyoom limit at the low point.
