# DeepSeek-V4.1-Flash-1017 EXL3 on 2x DGX Spark, with TensorFold

**2.1x the decode speed of the vLLM kit on the same two Sparks, the same weights and the same test suite, with all 32 agent tests passing.**

This recipe serves [DeepSeek-V4.1-Flash-1017](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017) (EXL3 2.9 bpw, 1M context, Engram tables) on two NVIDIA DGX Spark (GB10) with tensor parallelism over the ConnectX-7 link. The engine is [TensorFold](https://github.com/ashhart/TensorFold) with the DeepSeek-V4.1 TP2 engine by [bertholomus](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) (its Python line, v0.5.1), plus the fixes in [What I changed](#what-i-changed-to-make-it-work). The speed is bertholomus's engine. My changes make it ready for agents, with no loss of speed.

> **Status, 2026-10-17.** Measured on one pair of DGX Spark. Engine image `tensorfold-dsv41:fd2365042f56`. Every image went through the deploy gate (live request, 32-test suite, `--parallel` grammar check) and the Hermes agent gate. The numbers below are measured values only; this page states no number that a run in [`runs/`](runs/2026-10-17/) did not produce.

## Speed: TensorFold vs vLLM, same pair, same day

The vLLM baseline is the [MiaAI-Lab kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) at commit `6f7d1590`. Both engines ran the same suite ([`tests/v41_suite.py`](tests/v41_suite.py)) on the same pair, from the same weight files. vLLM: 1 run. TensorFold: 6 runs (6 engine builds of the same day, the last one is the released image); the table gives the median and the range.

| Single stream | vLLM kit | TensorFold (median, range) | Speedup |
|---|---:|---:|---:|
| Decode, prose | 27.0 tok/s | **56.8** (52.5-57.7) tok/s | **2.1x** |
| Decode, code | 39.9 tok/s | **92.2** (86.0-93.3) tok/s | **2.3x** |
| Decode, prose, thinking on | 25.7 tok/s | **54.2** (47.2-58.5) tok/s | **2.1x** |
| Decode, code, thinking on | 34.5 tok/s | **71.6** (59.6-77.8) tok/s | **2.1x** |
| Needle in 64K, total time | 80.4 s | **35.9** (34.0-40.0) s | **2.2x** |
| Prefill, 32K prompt (\*) | 925 tok/s | 2,100 (1,597-2,260) tok/s | 2.3x |
| Needle in 200K, total time (\*) | 216.9 s | 109.8 (103.6-126.9) s | 2.0x |
| Prefill, 8K prompt (first long prompt after a start) | 789 tok/s | 562 (427-958) tok/s | 0.7x |

- **Decode** is completion tokens per second of generation, from the server's own `usage` token counts. Prose and code are reported apart, because code drafts better with speculative decoding.
- **(\*) These two rows can include prefix-cache hits, on both engines.** The suite's 32K prompt starts with its 8K prompt, and the 200K needle prompt starts like the 64K one. The suite does not record cached tokens. For prefill with no cache hits, see the next table (TensorFold only).
- **The 8K row is slower on TensorFold.** It is the first long prompt after each start. A cold 8K prompt takes ~11 s right after a start and ~4-5 s later. A longer warm-up list does not remove this ([runs/.../17_*](runs/2026-10-17/17_tuning_warmup-spill-drafts/NOTES.md)), so it is not a kernel build; the likely cause is cold Engram reads from NVMe.
- **Both engines use speculative decoding** (vLLM: DSpark k=3; TensorFold: DSpark, 5 drafts; 4 drafts measured slower).
- Method, definitions and every run: [docs/BENCHMARKS.md](docs/BENCHMARKS.md).

### Compared with bertholomus's own numbers (same client)

bertholomus's benchmark client ([`kit_bench.py`](runs/2026-10-17/16_kit-bench-bertholomus_tensorfold-fd2365042f56/)) ran against this recipe's server with their published settings. Their numbers come from [their README](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10) (original V4.1-Flash weights, their nodes); ours use the 1017 weights.

| kit_bench row | Their Zig engine v0.6.0 | Their Python engine v0.5.1 | **This recipe** |
|---|---:|---:|---:|
| Decode set a: code / prose / structured | 89.8 / 59.0 / 106.6 | 89.4 / 57.7 / 107.2 | 84.7 / 58.7 / 103.8 |
| Decode set b: code / prose / structured | 103.1 / 64.1 / 143.2 | 101.5 / 63.4 / 143.4 | 95.1 / 57.7 / 140.6 |
| 2 / 4 streams, aggregate | 83.1 / 118.9 | 79.8 / 113.0 | 81.5 / 116.7 |
| 4 streams sustained 90 s | 142.0 | 125.3 | 120.0 |
| Prefill, no cache hits: 8K / 32K / 64K / 128K | 2,433 / 2,495 / 2,461 / 2,354 | 2,387 / 2,421 / 2,363 / 2,242 | 2,190 / 1,707 / 1,457 / 1,420 |
| Decode after a 128K prompt | 100.1 | 49.8 | 61.8 (52-73, 3 runs) |

- Decode and concurrency are within ~10% of their engine. Our long-prompt prefill is ~30-37% slower than their published numbers; the cause is not known yet ([KNOWN-LIMITS](docs/KNOWN-LIMITS.md)).
- Their Zig engine does not have the agent features below (`tool_choice` required/named and structured output return HTTP 400 there).

### Concurrency and the prefix cache (TensorFold, released image)

TensorFold serves 4 requests at once on one lane (`--parallel 4`).

| Streams | Prose, aggregate | Code, aggregate | Code, per stream |
|---:|---:|---:|---:|
| 1 | 29.6 tok/s (includes a 7.3 s first-request TTFT) | 94.1 tok/s | 95.8 tok/s |
| 2 | 73.5 tok/s | 106.6 tok/s | 57.2 tok/s |
| 3 | 82.3 tok/s | 124.3 tok/s | 43.5 tok/s |
| 4 | **94.4 tok/s** | **140.3 tok/s** | 36.8 tok/s |

- **Prefix cache.** A second question about the same 24K-token document: time to first token **12.8 s cold, 1.1 s warm** (22,528 of 24,210 prompt tokens came from the cache).
- **Prompts survive a restart** (`SPILL_GIB=64`, on by default). Kept prompts also go to disk. After a restart, an 18K-token document came back in **2.7 s** instead of 16.8 s cold.
- **No vLLM concurrency baseline.** The vLLM kit runs with `MAX_NUM_SEQS=1` (one request at a time). This recipe did not measure vLLM at 2-4 streams.

## What I changed to make it work

bertholomus's V4.1 engine was fast but not ready for agents: on the first run it passed **25 of 32** serving tests. I fixed the engine in the public fork [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold) and took it to **32/32**, without loss of speed.

| Step | Change | Suite |
|---|---|---:|
| 0 | bertholomus's V4.1 engine (Python line + willjriley's fixes) as found | 25/32 |
| 1 | **Forced tool calls on DeepSeek's DSML format** (`tool_choice` required and named) ([#1](https://github.com/Elevatormusic/TensorFold/pull/1)) | 27/32 |
| 2 | **Prompts byte-identical to DeepSeek's own encoder** (12/12 differential, golden fixtures) + a small chat template patch ([#1](https://github.com/Elevatormusic/TensorFold/pull/1)) | 28/32 |
| 3 | **Structured output on the speculative, two-node decoder**: json_schema, json_object, regex, choice, also 4 at a time ([#2](https://github.com/Elevatormusic/TensorFold/pull/2)) | 32/32 |
| 4 | **Fault isolation**: one bad request no longer takes the server down; a broken lane exits with code 75 for a supervisor ([#3](https://github.com/Elevatormusic/TensorFold/pull/3)) | 32/32 |

- **Deploy gate.** [`scripts/deploy_gate.sh`](scripts/deploy_gate.sh) deploys a new image and rolls back on its own if the live request, the suite or the grammar check fails. It caught a real bug in a build that passed all unit tests.
- **Agent gate.** The final images passed Hermes Agent's own client code at every thinking level, streamed and not streamed (35/35).
- **Review.** The OpenAI Codex review bot reviewed every engine change; all 8 findings are fixed, each with a test.

The full story, with the reason for each change and each launcher setting: **[docs/WHAT-I-CHANGED.md](docs/WHAT-I-CHANGED.md)**.

## Requirements

- **2x DGX Spark** (GB10, 128 GB each) with a QSFP cable between the ConnectX-7 ports. Measured on DGX OS (Ubuntu 24.04.5), driver 580.178.04, CUDA 13.0, Docker 29.6.2.
- **Disk per node:** ~385 GiB for the weights and Engram tables, plus ~110 GiB for the rank cache on the first start. An NVMe drive: the Engram tables are read at random offsets during decode.
- **Memory per node:** ~100 GiB free. Stop other GPU programs and other model servers first.
- **Network:** static IPv4 addresses on the CX7 link, RoCE v2, key-based ssh from the head to the worker. See [docs/NETWORK.md](docs/NETWORK.md).
- **Hugging Face CLI** (`hf`) on the head, logged in if your account needs it for the downloads.

## Quick start

Run every command on the head node, in this repo.

1. Copy the settings file, then edit the addresses, device names and paths:

```bash
cp .env.example .env
```

2. Download the weights and the Engram tables, verify their sha256, build the model folder, and copy it to the worker (~400 GB; hours on a slow link):

```bash
scripts/download.sh
```

3. Get the kit chat template at its pinned commit and apply the patch:

```bash
scripts/patch_template.py --worker
```

4. Build the engine image at the pinned commit and copy it to the worker:

```bash
docker/build.sh
```

5. Start both ranks (preflight checks run first; the first start takes ~15 min, later starts ~2-4 min):

```bash
./start.sh
```

6. Send five real requests (plain, thinking, strict json_schema, required tool call, regex):

```bash
./verify.sh
```

The server is now on `http://127.0.0.1:8888/v1` (OpenAI-compatible) with the model id `DeepSeek-V4.1-Flash-1017`.

## Test it yourself

Run the full test set against your server. The results go to `runs/<date>/`:

```bash
tests/run_all.sh
```

Add `BENCH=1` for the concurrency benchmark. See [tests/README.md](tests/README.md) for each test and its settings.

## Day-to-day

| Task | Command |
|---|---|
| Status of both ranks | `./status.sh` |
| Stop (saves both logs to `logs/`) | `./stop.sh` |
| Deploy a new engine image with rollback | `scripts/deploy_gate.sh <new-tag> <old-tag>` |

[docs/OPERATIONS.md](docs/OPERATIONS.md) covers restarts, exit code 75, boot, logs and the things that hurt on DGX Spark.

## Go back to the vLLM baseline

1. Stop TensorFold:

```bash
./stop.sh
```

2. Start the MiaAI kit at commit `6f7d1590` as its own README tells you. It uses the same weight revision. Both servers use port 8888 and ~100 GiB per node, so run only one at a time.

## Pinned versions

| What | Pin |
|---|---|
| Weights | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` @ `afc7aa0a` (43 shards) |
| Engram tables | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, shards 47 and 48 |
| Engine | [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold) @ `fd2365042f56` (TensorFold 0.6.6 + V4.1 port + PRs #1-#3) |
| Base image | `nvcr.io/nvidia/pytorch:26.07-py3` |
| Chat template | MiaAI kit `files/chat_template.jinja` @ `6f7d1590` + [our patch](runs/2026-10-17/template-patch.diff) |
| vLLM baseline | MiaAI kit @ `6f7d1590`, image `sha256:2f0cf3ad...` |

Every sha256, file size and node fact: [runs/2026-10-17/PROVENANCE.md](runs/2026-10-17/PROVENANCE.md). What is not measured or not supported: [docs/KNOWN-LIMITS.md](docs/KNOWN-LIMITS.md).

## Repo layout

| Path | What |
|---|---|
| `start.sh`, `stop.sh`, `status.sh`, `verify.sh` | run the server |
| `scripts/` | download, template patch, preflight, deploy gate (`_raw_*` files: the scripts as they ran on the measured pair) |
| `docker/` | the engine image (`Dockerfile`, `build.sh`) |
| `tests/` | the suite, the grammar check, the benchmark, the encoding tests, the Hermes gate |
| `runs/2026-10-17/` | every log and result behind the numbers on this page |
| `docs/` | what I changed, benchmarks, network, operations, known limits |

## Credits and licence

This recipe stands on the work of DeepSeek, LibertAIDAI (the EXL3 quantization), turboderp (ExLlamaV3), ashhart and the TensorFold contributors, bertholomus (the V4.1 TP2 engine this recipe runs), willjriley and Capicua25x (engine fixes), MiaAI-Lab (the vLLM kit and the template), and tonyd2wild (the benchmark method). See [CREDITS.md](CREDITS.md).

The scripts, tests and documents in this repo are Apache-2.0 ([LICENSE](LICENSE)). The weights, the template and the engine keep their own licences.
