# Display-KV position budget and overload test by Sora39831 (pull request #6)

Source: the author's [comment](https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/pull/6#issuecomment-6282611548) on pull request #6. The maintainers did not measure these numbers. This file keeps them as a record, with the author's caveats. The raw samples stay with the author.

## Setup

- One pair of DGX Spark nodes. Stock lane: image `tensorfold-dsv41:07aa02585f64-imgbudget1` (a local vision patch), commit `3b6d540`. Display lane: the display image of pull request #6 at head `935fdb2` over a pristine `07aa0258` base. `PARALLEL=2`, `CONTEXT=1048576`, `MEM_FLOOR_GIB=1.5`.
- The display build of that time had the hard allocator cap on by default (`MEM_ALLOCATOR_CAP=1` in that build). In the memory/top-k experiment the matching switch is `MEM_CAP`, and it is `0` by default (see `docs/MEMORY-TOPK.md`).
- earlyoom: SIGTERM below 1 GiB MemAvailable, SIGKILL below 512 MiB.
- Every request extends one cached prefix of about 1.03M tokens and has 1,048,432 tokens in total. The generator sends N requests at once. Only about 19K tokens of each prompt are prefilled, so this is not a prefill throughput test.

## Position budget

| | Mainline | Display |
|---|---|---|
| Positions | 2,097,152 (two windows of 1,048,576) | 2,109,440 (1792 MiB mapping, ordinary KV 0) |
| 2 streams at maximum context: minimum host MemAvailable | 2.33 GiB | about 2.0 GiB |

The display path adds 12,288 positions (+0.59%).

## Overload (N streams of 1,048,432 tokens at once)

| Streams | Mainline | Display |
|---|---|---|
| 3 | all HTTP 200, minimum MemAvailable 1.22 GiB | not run |
| 4 | earlyoom SIGTERM at 1019 MiB MemAvailable, container exit 133 | all HTTP 200, minimum 1.54 GiB |
| 5 | not run | all HTTP 200, minimum 1.68 GiB |
| 6 | not run | CUDA out of memory, a rank lost, exit 75 |

## Caveats of the author

- With the cap off (`MEM_ALLOCATOR_CAP=0`) and a positive floor the author saw no lane failure with 3 to 5 long streams (minimum 2.35 GiB). The test did not produce a spike large enough to make earlyoom act. It is not proof that the cap is safe to remove.
- All requests extend one cached prefix. A fully cold 1M prefill has a higher transient peak.
- The 5-stream display run overlapped one unrelated client request. The 4- and 6-stream runs were clean.
- Values are from the head. The author says the worker has the same shape.
