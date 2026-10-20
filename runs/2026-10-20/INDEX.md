# Runs of 20 October 2026 (UTC): memory under concurrent load (#21)

Same pair of DGX Spark as `runs/2026-10-18/`. Engine image `tensorfold-dsv41:val-90c9cf71c74e`: fork branch `deepseek-v41-tp2` at 434301e with fork PRs #13, #14 and #15, built on `tensorfold-dsv41:3c45bdaeab7d`. Settings: `CONTEXT=1048576`, `TF_DS_PREFILL_CHUNK=2048`, `SPILL_GIB=64`, `MEM_FLOOR_GIB=1.5`, `--vision-max-images 32`, host tuning from `scripts/host_tuning.sh` (earlyoom stops a process at about 1 GiB available).

`scripts/oom_trace.py` read `MemAvailable` on both nodes each second while it sent the load. Each image request had PNG images of 1024x768 and about 20K tokens of random-word text (about 34.5K prompt tokens with 8 images). The `started` field in the JSON files is head local time (UTC-7). `oom/earlyoom-head.txt` is in UTC.

## Results

| Run | `PARALLEL` | Load | Head lowest `MemAvailable` | Worker lowest | Result |
|---|---|---|---:|---:|---|
| 1 | 4 | 4 long prompts at the same time (99K, 152K, 229K, 305K prompt tokens), then 4 x 8 images | 1,009 MiB (earlyoom `mem avail` line) | n/a | long prompts 4 x 200; images: earlyoom stopped rank 0 at 06:36:51Z |
| 2 | 4 | new engine: 1 x 8, then 2 x 8, then 4 x 8 images | 1,379 MiB | 2,009 MiB | all 200 |
| 3 | 2 | 4 long prompts, then 4 x 8, then 4 x 16 images | 1,356 MiB (4 x 8); 4 x 16: 1,064 MiB in the samples, 901 MiB in the earlyoom `mem avail` line | 2,351 MiB | 4 x 8: all 200; 4 x 16: earlyoom stopped rank 0 at 06:49:50Z |

Files:

- Run 1: `oom/run1-p4-long-then-images.out` (the long-prompt results). This first version of the script stopped on the error and lost its samples, so this run has no CSV.
- Run 2: `oom/run2-p4-images.json` (requests and lowest values for each phase) and `.csv` (one row each second).
- Run 3: `oom/run3-p2-long-then-images.json` and `.csv`.
- The earlyoom events: `oom/earlyoom-head.txt`, with the `mem avail` line that earlyoom read just before each stop (the 1 s samples can miss the lowest point). The worker had no earlyoom event in any run.

## What the runs show

- The head runs out first. Rank 0 runs the vision tower (0.95 GiB in the engine log, `oom/run2-engine-vision.txt`) and decodes the images. At idle, the head had 0.4 to 0.6 GiB less available than the worker.
- With 4 x 8 images (8 is the recipe's limit), `PARALLEL=2` passed (run 3) where `PARALLEL=4` failed (run 1). The cache state differs (see the caveat), so this is not a clean A/B. The head's lowest point for 4 x 8 images was 1,356 MiB in run 3 and 1,379 MiB in run 2 (a new engine with no long prompts). With 4 x 16 images, which needs `--vision-max-images`, `PARALLEL=2` also failed.
- The worker went down to 2,009 MiB (run 2). The lowest values on the worker were 2.0 to 2.4 GiB.
- **Caveat for run 3:** its long prompts took 5 s to 17 s, against 55 s to 439 s in run 1. The prefix-cache spill (`SPILL_GIB=64`) kept them on disk after the restart, so run 3 loaded them and did not fill them again. Runs 1 and 3 are therefore not a clean A/B for the long prompts.
- `scripts/` holds the two scripts exactly as they ran (sha256 `a95537dc...89e7cbe` for `oom_trace.py`, `5f0f23be...e7c7c89` for `val_checks.py`). Runs 2 and 3 used these files. Run 1 used an earlier `oom_trace.py` without the `try` around the image request; that copy stopped on the first failed request and lost its samples. They are a record, not a general tool: they send requests to `172.17.0.1:8888` from 127.0.0.1 (that pair's docker bridge address) and read the worker at `10.100.80.2`. Their earlyoom filter keeps only the `low memory` and `sending` lines, so `oom/earlyoom-head.txt` was taken again from the journal with the `mem avail` lines.
