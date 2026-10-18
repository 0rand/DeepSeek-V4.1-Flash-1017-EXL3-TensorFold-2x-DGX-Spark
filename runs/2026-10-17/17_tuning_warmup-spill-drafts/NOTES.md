# Tuning runs, 2026-10-17 23:25Z to 23:45Z (image fd2365042f56)

All runs used bertholomus's client (`kit_bench.py`, see `../16_*`) or `prefix_probe.py` (this folder). One change at a time, one restart for each.

| Test | Setting | Result | Kept? |
|---|---|---|---|
| Cold 8K prefill, 2 fresh prompts right after a start (`tune-prefill-8k.json`) | 24 warm-up lengths (engine default list + 4161, 8209) | 10.98 s, then 4.83 s TTFT | No: the first long prompt still pays ~6 s. A longer warm-up list does not remove it, so it is not a kernel build. The likely cause is cold Engram reads from NVMe (not proven). |
| 18K-token document, then a second question (`prefix_probe.py 7 0`, `7 1`) | spill on | cold 8.95 s; warm 1.09 s (16,384 cached) | - |
| The same document after a restart (`prefix_probe.py 7 0`) and a new one (`8 0`) | spill on (`TF_DS_SPILL_GIB=64`) | **2.66 s** (16,384 tokens read back from disk) vs 16.8 s for the new document | **Yes** |
| Draft depth 4 (`tune-d4-*.json`) vs 5 (`../16_*`) | `--mtp-drafts 4` | set a code/prose/structured 86.7/58.1/98.7 (5: 84.7/58.7/103.8); set b 88.8/57.2/123.9 (5: 95.1/57.7/140.6); 4 streams 113.2 (5: 116.7) | No: 5 stays. Depth 3 not run. |

Gate on the final setting (depth 5, spill 64 GiB, short warm-up list): `grammar-check-fd23-spill.log` 14/14, `suite-fd23-spill.log` 32/32.
That suite ran right after an earlier suite run, so its prefill and needle rows are prefix-cache hits (0.7-1.5 s). Do not read them as speed numbers.
The `prefix_probe.py` lines above were printed to the console; they are copied here.
