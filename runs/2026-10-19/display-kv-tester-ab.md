# Display-KV test and A/B report by Sora39831 (pull request #6)

Source: the author's comments on pull request #6 ([first report](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/pull/6#issuecomment-6275425639), [A/B report](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/pull/6#issuecomment-6276483069)). The maintainers did not measure these numbers. This file keeps them as a record, with the author's caveats. The raw samples stay with the author.

## Setup

- Two DGX Spark nodes (head and worker over ConnectX-7), driver 580.178.04, `PARALLEL=2`, `CONTEXT=1048576`, temperature 0, streaming, a fresh engine start before each block.
- Mainline: `tensorfold-dsv41:07aa02585f64-imgbudget1`, commit `3b6d540`, `SPILL_GIB=64`. This image carries a local vision patch.
- Display: image built from pull request head `935fdb2` over a pristine `07aa0258` base, `SPILL_GIB=32`, bundle default `PREFILL_CHUNK=512`. Mainline used 2048.
- Thinking on, `reasoning_effort=high`, for every request.

## Prefill (cold prompts, one stream)

| Prompt | Mainline | Display | Change |
|---|---|---|---|
| 32,014 tokens | TTFT 15.08 s, 2,124 tok/s | TTFT 19.05 s, 1,680 tok/s | -20.9% |
| 54,008 tokens | 26.5 s, 2,038 tok/s | 33.3 s, 1,623 tok/s | -20.4% |
| 127,966 tokens | 66.55 s, 1,923 tok/s | 83.42 s, 1,534 tok/s | -20.2% |
| 1,029,544 tokens | 792.5 s, 1,299 tok/s | 942.4 s, 1,092 tok/s | -15.9% |

Decode with one stream: 51 to 62 tok/s mainline, 48 to 59 tok/s display. Prefix cache hit on a 54K prompt: TTFT 0.73 s (53,248 cached) mainline, 0.46 s (53,760 cached) display. A needle at about 1M tokens was found on both sides. Structured output, a forced tool call and the over-context HTTP 400 gave the same result on both sides. Minimum host MemAvailable during the 1M prefill: 1.96 GiB mainline, 2.01 GiB display.

## Caveats of the author

- The display image of this test is the combined build of pull request #6: display KV, a patched SSD spill, the tool-output guard and a memory guard. Pull request #30 has only the display KV part (spill off, no guard). The gap in prefill tok/s may differ for that smaller build, and nobody measured it.
- The two blocks ran one after the other (mainline first), with one run per size on one pair.
- The mainline image has a local vision patch. This is not a comparison of two identical binaries.
- The display build used a 512-token prefill chunk and mainline used 2048. The test does not separate the effect of the chunk size from the effect of the display allocation.
- The 2 x 30K concurrency test on mainline hit a prefix-cache carry-over and is not in the table.

## Other results of the first report

- 84 offline tests passed on a pristine source. The launcher `--build` passed the DRM preparation and two full 1792 MiB sweeps on both nodes. `verify.sh` gave 5 of 5.
- The spill tier started and stayed at 0 entries. With a 2.1M-position pool and two streams nothing is evicted, so write, evict and restore were not exercised. Five prompted attempts did not produce an unoffered `task` call, so the tool-output guard was not reached.
- The author reported two issues: the head `sudo -v` needs a terminal (fixed in the launcher), and the drift error did not name the hashes (fixed in `apply.py`).
