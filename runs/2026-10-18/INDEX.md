# Runs of 18 October 2026 (UTC): long prompts and the 1M context

Same pair of DGX Spark as `runs/2026-10-17/`. All prefill rows use bertholomus's `kit_bench.py prefill` (random-word prompts, so no prompt-cache hit; `cached_tokens` is 0 in the server log) through `scripts/src127.py` (it connects from 127.0.0.1). `scripts/pf.sh TAG` runs one 8K warm prompt, then 8K two times, then 32K, 64K and 128K once each. Speed is `prompt_tokens / time to first token`.

## Final setup (the recipe as published)

Image `tensorfold-dsv41:92aa710b533d` (fork PRs #1-#4), `TF_DS_ENGRAM_DONTNEED=1`, `--oom-score-adj 1000`, and the host settings of `scripts/host_tuning.sh`: no swap for the engine, `vm.swappiness=10`, `vm.compaction_proactiveness=0`, `vm.extfrag_threshold=500`, `vm.min_free_kbytes=1048576`, `vm.watermark_scale_factor=50`, and earlyoom set to stop the engine below 1 GiB available.

| Run | 8K (best of 2) | 32K | 64K | 128K | File |
|---|---:|---:|---:|---:|---|
| hw5 | 2,214 | 2,201 | 2,119 | 1,986 | `kitbench-prefill/pf-hw5-*.json` |
| hw6 (right after hw5) | 2,174 | 2,180 | 2,109 | 1,981 | `kitbench-prefill/pf-hw6-*.json` |
| ht1 (after `scripts/host_tuning.sh` and an engine restart) | 2,223 | 2,065 | 2,037 | 1,914 | `kitbench-prefill/pf-ht1-*.json` |
| ht2 | 2,179 | 2,165 | 1,836 | 1,961 | `kitbench-prefill/pf-ht2-*.json` |
| ht3 | 2,177 | 2,100 | 1,889 | 1,851 | `kitbench-prefill/pf-ht3-*.json` |
| **Median** | **2,179** | **2,165** | **2,037** | **1,961** | |

hw5 and hw6 ran with the same host settings, set by hand before the engine restart. The 128K prompt is now steady (1,851-1,986). The 64K prompt still varies (1,836-2,119). No other request reached the server during these runs.

## Release gate (final setup)

- `gate/suite-ht.log`: 32/32 in 264 s. The 64K and 200K needles filled from new (31.1 s and 94.9 s). Before the memory changes, the 64K needle froze the head.
- `gate/grammar-check-ht.log`: 14/14.
- Lowest MemAvailable during the bench and the gate: 2,206 MiB on the head, 2,712 MiB on the worker. During the 1M needle (below): 1,779 MiB on the head. earlyoom acted at 1,536 MiB in these runs (it did not act); after the 1M run the limit went down to 1,024 MiB.

## Decode check (final setup)

`kitbench-decode/`: greedy decode, 3 replies per prompt, same text and same number of steps as on 17 October (`runs/2026-10-17`). Set a: code 84.7 (median of `decode-a-ht2.json`; `decode-a-ht.json` gave 71.0), prose 59.0, structured 104.3. Set b: code 98.8, prose 57.4, structured 139.5. In each set-a run, 1 or 2 of the 3 code replies ran at 69-71 tok/s instead of 85: decode has the same slow state as the prompts. Engine counters for the 512-token replies of set a (`logs/r0-decode-done-ht.log`: every rank 0 `done` line of the decode check; `rounds` = decode steps, `accepted` = accepted/drafted tokens; tokens per decode step = (rounds + accepted) / rounds): prose 1.98 (rounds 258, 253 of 436 drafted tokens accepted), code 3.60 (rounds 142, 369 of 542), structured 4.73 (rounds 108, 403 of 502); the slow code replies (69-71 tok/s) had the same 142 steps.

## Memory settings: A/B with 0.5 s samples

`scripts/hwsample.sh` sampled both nodes every 0.5 s (GPU clock, power and throttle reasons; CPU; PSI; reclaim, compaction and swap counters) during two `pf.sh` runs (248 prompt chunks) for each setting. `scripts/hwcorr.py` puts each chunk of the rank 0 chunk log (`logs/r?-chunks-*.log`) next to the samples.

| Setting | Runs | Slow chunks (over 1.3 s) | 128K tok/s |
|---|---|---:|---:|
| First version of `scripts/host_tuning.sh` (`extfrag_threshold=1000`, kernel watermarks) | hw1, hw2 | 5 of 248 | 1,962 and 1,967 |
| Background compaction on (`compaction_proactiveness=20`), `min_free_kbytes=1048576`, `watermark_scale_factor=50` | hw3, hw4 | 46 of 248 | 1,586 and 1,736 |
| Final settings (above) | hw5, hw6 | 2 of 248 (both 1.31 s) | 1,986 and 1,981 |

- The GPU clock stayed at 2,370-2,460 MHz in slow chunks, and no throttle reason was set. The GPU did not slow down.
- With the first version, slow chunks had 57 times (head) and 9 times (worker) the page scanning of normal chunks. With background compaction on, slow chunks had no extra page scanning; that cause is not known.
- The worker's swap-outs came from other processes, not the engine (polkitd held 1.7 GB of swap).

## First version of the host settings

Same image, `vm.extfrag_threshold=1000` and kernel watermarks. The spread was large: slow runs had chunks of 1.5-2.6 s among the normal 0.9-1.1 s ones (`scripts/chunksum.py` over `logs/r0-final-92aa710b533d.log`).

| Run | 8K (best of 2) | 32K | 64K | 128K | File |
|---|---:|---:|---:|---:|---|
| A (same code, mounted test copy) | 2,200 | 2,188 | 2,113 | 1,988 | `kitbench-prefill/pf-dn-async-*.json` |
| B (final image) | 2,160 | 2,180 | 2,113 | 1,202 | `kitbench-prefill/pf-final1-*.json` |
| C (final image, right after B) | 1,690 | 1,865 | 1,879 | 1,540 | `kitbench-prefill/pf-final2-*.json` |
| Median | 2,160 | 2,180 | 2,113 | 1,540 | |

## 1M context

`needle-1m.json` (`scripts/needle_1m.py 1035000 0.63`): a 1,034,753-token prompt (limit 1,048,576), random filler lines, one needle at 63% depth. Found (`ORCHID-8071`), first token after 830.4 s (1,246 tok/s), `cached_tokens` 0. Image `fd2365042f56` with swap off; chunk log in `logs/r0-20261018T015205Z.log`.

`needle-1m-final.json`: the same test on the final setup. A 1,035,020-token prompt, found (`ORCHID-6456`), first token after 817.7 s (1,266 tok/s), `cached_tokens` 0. Lowest MemAvailable during the fill: 1,779 MiB on the head, 2,542 MiB on the worker. That was only 243 MiB above earlyoom's first limit (1,536 MiB), so `scripts/host_tuning.sh` now stops the engine below 1,024 MiB (SIGKILL below 512 MiB). MemAvailable already leaves out the kernel's high watermark (about 2.3 GiB with these settings), so the kernel still has its reserve at those limits.

## How each change was found (in order)

| Step | Change | 32K | 64K | 128K | File |
|---|---|---:|---:|---:|---|
| 0 | 17 October, image `fd2365042f56` (`runs/2026-10-17/16_*`) | 1,707 | 1,457 | 1,420 | |
| 1 | Swap off for the engine, swappiness 10 | 1,872 | 1,805-1,902 | 1,672 | `e5-prefill-*.json`, `e5-counters.txt` |
| 2 | Prompt chunks of 4096 (rejected) | 1,710 | 1,619 | 1,553 | `pf-c4096-*.json` |
| 3 | Context 262K instead of 1M (rejected: no gain) | 1,676 | 1,272 | 1,280 | `pf-ctx262k-*.json` |
| 4 | Compaction down (`compaction_proactiveness=0`, `extfrag_threshold=1000`) | 2,185 | 2,038 | 1,507 | `pf-compact0-*.json` |
| 5 | + Engram page drop on the chunk's thread | 2,082 | 2,009 | 1,903 | `pf-dontneed-*.json` |
| 6 | + spill off (rejected: no gain, spill stays on) | 2,087 | 1,988 | 1,901 | `pf-dn-nospill-*.json` |
| 7 | + page drop on its own thread (fork PR #4) | 2,188 | 2,113 | 1,988 | `pf-dn-async-*.json` |
| 8 | + 32 Engram reader threads (rejected) | 2,100 | 2,050 | 1,953 | `pf-dn-t32-*.json` |
| 9 | + final memory settings (`extfrag_threshold=500`, `min_free_kbytes=1048576`, `watermark_scale_factor=50`), median of 5 runs | 2,165 | 2,037 | 1,961 | `pf-hw5-*` to `pf-ht3-*` |

What the measurements showed:

- **Per-section timing** (`TF_DS_TIMING=1`, both ranks: `logs/r?-20261018T021147Z.log`, `logs/r0-20261018T020727Z.log`). A normal 2048-token chunk: MoE ~370 ms, attention on ratio-1 layers ~172 ms, hc ~113 ms, gather ~105 ms, Engram compute ~52 ms, Engram read wait ~1.7 ms. Attention on ratio-2 layers grows with position (68 ms at the start, 195 ms at 61K). In a slow chunk every section of one rank grows (MoE 614-702 ms), and the other rank waits in the gather.
- **Kernel counters** for one 64K prompt (`scripts/vmd.sh`): with default compaction, 548 direct compaction stalls on the head; after step 4, 7. Without the page drop the worker reclaimed ~2.2M pages a 64K prompt; with it, ~0.5M.
- **Drive** (`scripts/randread.c` on the worker, idle server): 4 KiB random reads at 514K/s buffered (128 threads) and 0.9-1.6M/s with O_DIRECT. A chunk needs ~49K of them, so the drive is not the limit.
- **GPU clock and rails**: 2,360-2,500 MHz under load, no throttle reason set; both RoCE rails carry equal traffic.
