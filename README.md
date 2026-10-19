# DeepSeek V4.1 Flash on two DGX Spark

**Run DeepSeek's V4.1 Flash model on two NVIDIA DGX Spark computers with about 2× the decode and prefill speed of the standard vLLM setup, ready for AI agents.**

Also on Hugging Face: [Elevatormusic/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark-TensorFold-2x-Faster](https://huggingface.co/Elevatormusic/DeepSeek-V4.1-Flash-1017-2x-DGX-Spark-TensorFold-2x-Faster). The engine is [my TensorFold fork](https://github.com/Elevatormusic/TensorFold) (branch `deepseek-v41-tp2`).

Measured on 17 and 18 October 2026 on one pair of DGX Spark. Every number on this page comes from a saved run in [`runs/`](runs/2026-10-17/INDEX.md) ([prefill and 1M context](runs/2026-10-18/INDEX.md)).

---

## At a glance

| | Standard setup (vLLM) | This recipe | |
|---|---:|---:|---|
| Decode, prose (1 request) | 27 tok/s | **57 tok/s** | **2.1× faster** |
| Decode, code (1 request) | 40 tok/s | **92 tok/s** | **2.3× faster** |
| Prefill, 32K-token prompt | 925 tok/s | **2,179 tok/s** | **2.4× faster** |
| Needle-in-a-haystack, 64K-token prompt (request time) | 80 s | **36 s** | **2.2× faster** |
| Max context length, as each setup ships | 262,144 tokens | **1,048,576 tokens** | **4× longer** |

*Prefill* is the speed at which the server reads the prompt; it sets the time to first token (TTFT). *Decode* is the speed at which it writes the reply. A *token* is a piece of a word: 100 tokens are about 75 English words.

Both setups ran on the same two computers, with the same model files. The prefill speed of this recipe is the median of 3 runs with cold prompts (new random prompts, no prefix-cache hit); the range is 2,174 to 2,195 tok/s. The vLLM prefill speed comes from the test suite's 32K prompt, which can include a prefix-cache hit; that would favor vLLM. In a needle test with a 1,035,020-token prompt, the model found the needle; the TTFT was 13.6 minutes.

Prefill with cold prompts (bertholomus's kit_bench client, median of 3 runs):

| Prompt length | 8K tokens | 32K | 64K | 128K | 1,035,020 |
|---|---:|---:|---:|---:|---:|
| Prefill (tok/s) | 2,166 | 2,179 | 2,110 | 1,989 | 1,266 (1 run) |

---

## What you get

- **About 2× the decode speed** of the standard vLLM setup: 57 vs 27 tok/s on prose, 92 vs 40 tok/s on code.
- **Fast prefill.** About 2,000 to 2,200 tok/s for prompts up to 64K tokens.
- **Ready for AI agents.** Forced tool calls (`tool_choice` set to `required` or to one tool) and structured output (a JSON schema, a regex or a list of choices).
- **4 concurrent requests.** Up to 4 requests share the two computers. Aggregate decode reaches 140 tok/s on code.
- **Prefix cache.** A second question about the same long document has a TTFT of about 1 s instead of 13 s. Cached prompts also go to disk, so after a restart the TTFT is about 3 s.
- **Display-backed KV and SSD spill.** The current baseline maps 1792 MiB of the display reservation per node into a 2.1M-position shared pool and spills kept prompts to a 32 GiB per-rank SSD tier, so long agent sessions survive eviction and restarts.
- **Fault isolation.** One failed request ends only that request, not the whole server.
- **1M-token context.** The full 1,048,576-token context window. A needle test with a 1,035,020-token prompt passed.

---

## Before you start

You need:

- **Two NVIDIA DGX Spark**, joined by one QSFP cable between their ConnectX-7 network ports. NVIDIA's [Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks) guide shows how to cable them, set fixed addresses, and set up ssh.
- **About 500 GB of free disk on each computer.** The model files use 385 GB. The first start uses 110 GB more.
- **About 100 GB of free memory on each computer.** Close other AI programs first.
- **A network setup** with fixed addresses on the cable link and ssh from the first computer to the second. Do NVIDIA's guide first, then read [our network notes](docs/NETWORK.md).
- **Software on the head:** Docker with GPU support (DGX OS has it), Python 3, `curl`, `rsync`, and the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli) (`hf`). The worker needs Docker and `rsync`.
- **Time.** Most of the first setup is the 400 GB download. It takes a few hours on a fast internet link.

In this guide, the *head* is the computer where you type the commands. The *worker* is the second computer.

---

## Set up

Run each command on the head, in this folder.

**1. Make your settings file.** Then open `.env` and type your addresses and folder paths.

```bash
cp .env.example .env
```

**2. Download the model.** This step gets 400 GB, checks every file, and copies it to the worker.

```bash
scripts/download.sh
```

**3. Prepare the chat template.** This step gets the template and applies a small fix.

```bash
scripts/patch_template.py --worker
```

**4. Build the engine.** This step builds the server software from [my TensorFold fork](https://github.com/Elevatormusic/TensorFold) and copies it to the worker. The fork is bertholomus's engine plus the changes in [What I changed](#what-i-changed-to-make-it-work).

```bash
docker/build.sh
```

**5. Tune the memory settings (once).** This keeps the model out of swap, turns transparent huge pages off and stops memory stalls during long prefills, on both computers. If memory runs out, it stops the model, not the computer (this needs the earlyoom package, which DGX OS has). It asks for your password on each computer. `scripts/host_tuning.sh --undo` removes it.

```bash
scripts/host_tuning.sh
```

**6. Start the server.** It checks your setup first. The first start takes about 15 minutes. Later starts take 2 to 4 minutes.

```bash
./start.sh
```

**7. Check that it works.** This sends five real requests and shows the result of each.

```bash
./verify.sh
```

Your server is ready at `http://127.0.0.1:8888/v1`. It works with any app that supports the OpenAI API. The model name is `DeepSeek-V4.1-Flash-1017`.

### Display-Backed KV (current baseline)

On DGX Spark, the display-backed variant uses 1792 MiB of the NVIDIA display
reservation per node for compressed KV and indexer storage. With this model it
provides a 2,109,440-position shared pool while keeping the 1,048,576-token
per-request context limit. It also exposes live pool and SSD spill metrics to
EngineMetrics.

Stop the current engine before switching, then run the self-contained launcher:

```bash
./stop.sh
./start-display-kv.sh --build
./verify.sh
```

With `--build`, the launcher prepares a separate image on both nodes. Later
`./start-display-kv.sh` runs use that saved image without building or copying,
even if staged source files have changed. It safely reloads `nvidia_drm` on
both nodes, checks the full display mapping, and starts the variant. It uses this
repo's scripts and sources only; no other model repo is needed. The head may ask
for sudo and the worker needs passwordless sudo. Active NVIDIA scanout, GPU
compute processes and DRM clients prevent preparation; an unrelated framebuffer
TTY or a connected-but-disabled output does not.

The normal `.env` and `start.sh` are not changed. Disk prompt spill is enabled
through the reused GLM tier when `SPILL_GIB` is nonzero (32 GiB per rank in the
current baseline); in-memory prefix reuse remains on. DRM changes are
runtime-only and the launcher prepares them again after a reboot. Full-window
and near-full-pool validation remain open. See [Display KV](docs/DISPLAY-KV.md)
for checks, rollback and tests.

#### How the display cache works

The display cache maps the compressed KV and indexer planes into the NVIDIA
display reservation, which normal compute does not use. The engine then serves
from a shared pool of 2,109,440 positions; each request still has the model's
1,048,576-token context limit. The reservation is outside PyTorch's allocator
accounting and does not appear in `nvidia-smi`'s memory total. Mapped rows move
through bounded GPU-kernel copies, never through CPU reads or copy engines.

The launcher prepares `nvidia_drm` at runtime on both nodes, runs two full
integrity sweeps over the mapping, and starts the variant. Preparation refuses
if an NVIDIA card is scanning out, a GPU compute process is active, or a DRM
client holds the card. It never stops GDM or the firmware framebuffer console.
DRM changes are runtime-only; after a reboot, run the launcher again.

#### Display-RAM and SSD spill settings

| Setting | Default | What it does |
|---|---|---|
| `TF_DS_DISPLAY_KV_MIB` | 1792 | Display reservation mapped for KV per node (set by the launcher). |
| `PARALLEL` | 4 | Concurrent streams sharing the pool; display KV requires more than 1. |
| `CONTEXT` | 1048576 | Per-request context limit; the shared pool is larger. |
| `PREFILL_CHUNK` | 512 | Smaller chunks reduce temporary prefill memory. |
| `MEM_FLOOR_GIB` | 1.25 | Host memory kept free; display images raise positive floors below 1.5 GiB to 1.5 GiB. |
| `SPILL_GIB` | 32 | SSD spill capacity per rank; 0 disables the tier. |
| `TF_DS_SPILL_DIR` | `/cache/tf-ds-spill` | Spill directory (set by `start.sh`). |
| `TF_DS_SPILL_HIGHWATER` | 0.70 | Write kept prompts early past this fraction of memory. |
| `TF_DS_SPILL_MIN` | 4096 | Prompts shorter than this are not spilled. |
| `TF_DS_SPILL_MIN_FREE_GIB` | 50 | No write leaves less than this much disk free. |
| `TF_DS_SPILL_KEEP_BUILDS` | 1 | Other builds' spill directories kept at start. |
| `TF_SPILL_WRITERS` | 2 | Background writer threads per rank. |
| `TF_SPILL_DIRECT` | 1 | Use O_DIRECT; 0 falls back to buffered writes. |
| `TF_SPILL_FLUSH_S` | 60 | Shutdown flush budget; the drain waits up to 55 s inside `docker stop -t 60`. |

#### Use patterns

- **Long agent sessions.** The first request fills the prompt; later turns reuse
  the in-memory prefix. When a prompt leaves the window, the spill tier writes it
  to disk; a later request that extends it restores the rows instead of
  prefilling.
- **Restart persistence.** Prompts whose writes reached disk survive a restart.
  A validated restart restored a 237568-token prompt in 0.05 s.
- **Concurrency.** Four streams share the pool; each stream keeps its own request
  context. The spill tier writes and restores per rank, so both ranks must hold
  the same prompt.
- **Monitoring.** `/health["spill"]` and the `spill_*` Prometheus families show
  entries, bytes, write/load counters, restore latency, failures and free disk.
  EngineMetrics renders them in its Live Cache panel.

#### Precautions

- Display-RAM is the current baseline, but full-window and near-full-pool
  operation are not yet validated. Watch host memory, not only CUDA free memory.
- Do not run another model server, GPU workload or DRM client during
  preparation or probes. Stop both ranks before `--build` or `--build-only`.
- Never stop GDM or the firmware framebuffer console to make DRM checks pass.
- The spill signature includes the model, TensorFold sources, CUDA/torch
  version, GPU name, KV layout and display size, but ignores the driver version.
  A source change starts a new spill directory; old stores remain on disk but
  are not reused. `TF_DS_SPILL_KEEP_BUILDS` controls how many old directories
  are kept.
- Spill storage is per rank. A prompt is restored only when every rank has it;
  otherwise the request fills fresh. Corrupt entries are dropped after a failed
  restore.
- `docker stop` sends SIGTERM; the shutdown hook drains queued writes for up to
  55 s. A SIGKILL or crash can lose queued writes.

---

## Connect your app

Use these three settings in any app that supports a custom OpenAI endpoint:

| Setting | Value |
|---|---|
| Base URL | `http://127.0.0.1:8888/v1` |
| Model | `DeepSeek-V4.1-Flash-1017` |
| API key | any text (or your key, if you set one) |

Thinking is on by default at high effort. Set `THINKING=1` and
`REASONING_EFFORT=high` in `.env` to configure these server defaults; requests can
override them. Use `THINKING=0` for a non-thinking default.

[The connection guide](docs/CONNECT.md) covers thinking levels, tool calls, structured output, and use from other computers.

---

## Everyday use

| To do this | Run this |
|---|---|
| See if the server is running | `./status.sh` |
| Stop the server | `./stop.sh` |
| Run all the tests | `tests/run_all.sh` |
| Install a new engine version safely | `scripts/deploy_gate.sh <new> <old>` |

If something goes wrong, read [Operations](docs/OPERATIONS.md). It covers restarts, logs, and what to avoid on DGX Spark.

---

## EngineMetrics

[EngineMetrics](https://github.com/0rand/EngineMetrics) is a zero-dependency
terminal dashboard for TensorFold (and vLLM, mlx-serve and oMLX). Its TensorFold
adapter supports this recipe's display-backed pool and SSD spill tier, and this
recipe's `/health` and `/metrics` endpoints were extended to match the fields
EngineMetrics reads.

```bash
git clone https://github.com/0rand/EngineMetrics.git
cd EngineMetrics
python3 engine_metrics.py --host 127.0.0.1 --port 8100 --interval 2
```

Use `--port 8888` for the ordinary recipe. The dashboard shows live prefill and
decode rates, stream states, pool occupancy, kept prompts, MTP acceptance,
TTFT/latency averages, cache hits and the SSD spill tier (stored/written/loaded,
restore latency, failures and disk free). Current engines report their configured
capacity as `spill.gib` in `/health`; `--spill-cap GIB` is a fallback for older
engines.

---

## What I changed to make it work

The fast engine for this model was written by [bertholomus](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10). It was fast, but agent apps could not use it fully. I added what they need, with no loss of speed.

1. **Forced tool calls.** `tool_choice` now works with `required` and with one named tool: apps can tell the model that it *must* call a tool, or one specific tool. Agent apps use this to plan their steps.
2. **Prompt encoding identical to DeepSeek's.** Every prompt now matches DeepSeek's own reference encoder, byte for byte.
3. **Structured output.** The model can now follow a JSON schema, a regex, or a list of choices. This works with speculative decoding, and with 4 concurrent requests.
4. **Fault isolation.** Before, one failed request stopped all requests until a manual restart. Now it ends only that request.
5. **Faster long prefills.** The server now drops the page cache of the Engram tables after each prompt chunk. Before, the page cache filled the memory, and some chunks stalled for 1 to 4 s in reclaim and compaction.
6. **Steady long prefills.** The server writes its spill files with O_DIRECT, so they skip the page cache, and the host setup turns transparent huge pages off. Page compaction stalls the GPU on these computers; these two changes remove the last compaction stalls, and prefill speed now changes by less than 2% from run to run.
7. **Display-backed KV and SSD spill.** The current baseline maps the NVIDIA display reservation into a shared KV pool and spills kept prompts to a per-rank SSD tier, so long agent sessions survive eviction and restarts.

[Read the full story](docs/WHAT-I-CHANGED.md), with the reason for each change. The engine code is in [my fork](https://github.com/Elevatormusic/TensorFold), in pull requests [#1](https://github.com/Elevatormusic/TensorFold/pull/1), [#2](https://github.com/Elevatormusic/TensorFold/pull/2), [#3](https://github.com/Elevatormusic/TensorFold/pull/3), [#4](https://github.com/Elevatormusic/TensorFold/pull/4) and [#6](https://github.com/Elevatormusic/TensorFold/pull/6).

---

## Good to know

- **The first long prompt after a start has a higher TTFT,** about 1 to 2 s more. Later prompts do not.
- **Prefill is a little slower than bertholomus reports.** At 32K tokens this setup prefills at 2,179 tok/s; they publish 2,421 on different computers and the original V4.1 weights. Prefill speed is now steady from run to run (64K: 2,110 to 2,115 tok/s), except for the first prompt after a start.
- **A 1M-token prompt has a TTFT of about 14 minutes** (prefill at 1,266 tok/s).
- **vLLM was tested with 1 request at a time only.** So the concurrency numbers have no vLLM comparison.
- **Living Weights is not available.** That TensorFold feature works only with a different model on one computer.

[All known limits](docs/KNOWN-LIMITS.md)

---

## Go back to the standard setup

1. Stop this server:

```bash
./stop.sh
```

2. Start the [MiaAI-Lab vLLM kit](https://github.com/MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks) as its guide tells you. It uses the same model files. Run only one server at a time.

---

## For experts

<details>
<summary><b>Full speed results</b></summary>

#### This recipe vs the vLLM kit (same computers, same day, same prompts)

vLLM: 1 run. This recipe: 6 runs (6 engine builds of the same day, the last is the released one). The table shows the median and the range.

| 1 request | vLLM kit | This recipe (median, range) | Faster by |
|---|---:|---:|---:|
| Decode, prose | 27.0 tok/s | **56.8** (52.5-57.7) tok/s | **2.1×** |
| Decode, code | 39.9 tok/s | **92.2** (86.0-93.3) tok/s | **2.3×** |
| Decode, prose, thinking on | 25.7 tok/s | **54.2** (47.2-58.5) tok/s | **2.1×** |
| Decode, code, thinking on | 34.5 tok/s | **71.6** (59.6-77.8) tok/s | **2.1×** |
| Needle test, 64K tokens (request time) | 80.4 s | **35.9** (34.0-40.0) s | **2.2×** |
| Prefill, 32K prompt (\*) | 925 tok/s | 2,100 (1,597-2,260) tok/s | 2.3× |
| Needle test, 200K tokens (\*) | 216.9 s | 109.8 (103.6-126.9) s | 2.0× |
| Prefill, 8K prompt, first after a start | 789 tok/s | 562 (427-958) tok/s | 0.7× |

- Decode speed counts the completion tokens the server reports, per second after the first token. Prose and code are shown apart, because code has a higher draft acceptance rate with speculative decoding (greedy: 3.6 tokens per step on code, 2.0 on prose; [counters](runs/2026-10-18/kitbench-decode/r0-done-ht.log)).
- (\*) These two rows can include prefix-cache hits, on both engines. The suite's 32K prompt starts with its 8K prompt, and the 200K prompt starts like the 64K one.
- The 8K row is slow because it is the first long prompt after each start. A longer warm-up list does not fix it ([tuning notes](runs/2026-10-17/17_tuning_warmup-spill-drafts/NOTES.md)). The likely cause is cold reads of the Engram tables from disk.
- Both engines use speculative decoding (vLLM: DSpark, 3 drafts; this recipe: DSpark, 5 drafts; 4 drafts measured slower).

#### Same test client as bertholomus

bertholomus's own benchmark client ([`kit_bench.py`](runs/2026-10-17/16_kit-bench-bertholomus_tensorfold-fd2365042f56/)) ran against this server with their settings. Their numbers are from [their README](https://github.com/bertholomus/deepseek-v4.1-tensorfold-tp2-2xgb10): original V4.1 Flash weights, their computers. This recipe uses the 1017 weights.

| kit_bench row | Their Zig engine v0.6.0 | Their Python engine v0.5.1 | **This recipe** |
|---|---:|---:|---:|
| Decode, set a: code / prose / structured | 89.8 / 59.0 / 106.6 | 89.4 / 57.7 / 107.2 | 84.8 / 58.6 / 103.9 (\*\*\*) |
| Decode, set b: code / prose / structured | 103.1 / 64.1 / 143.2 | 101.5 / 63.4 / 143.4 | 98.6 / 56.8 / 139.7 (\*\*\*) |
| Aggregate decode, 2 / 4 concurrent requests | 83.1 / 118.9 | 79.8 / 113.0 | 81.5 / 116.7 |
| Aggregate decode, 4 concurrent requests for 90 s | 142.0 | 125.3 | 120.0 |
| Prefill, cold prompt: 8K / 32K / 64K / 128K | 2,433 / 2,495 / 2,461 / 2,354 | 2,387 / 2,421 / 2,363 / 2,242 | 2,166 / 2,179 / 2,110 / 1,989 (\*\*) |
| Decode after a 128K prompt | 100.1 | 49.8 | 61.8 (52-73, 3 runs) |

Their Zig engine does not have the agent features above (forced tool calls and structured output return an error there).

(\*\*) Median of 3 runs on 18 October with the final setup (fork PRs #4 and #6, `scripts/host_tuning.sh` with transparent huge pages off); ranges 2,163-2,189 / 2,174-2,195 / 2,110-2,115 / 1,976-1,991. Before the O_DIRECT spill and THP change (median of 5): 2,179 / 2,165 / 2,037 / 1,961, with 64K from 1,836 to 2,119. On 17 October: 2,190 / 1,707 / 1,457 / 1,420. How each change was found: [runs/2026-10-18](runs/2026-10-18/INDEX.md).

(\*\*\*) Median of 3 greedy runs on 18 October with transparent huge pages off; each set of 3 is within 0.6%. Before (THP madvise), 1 of 3 code replies ran at 69-71 tok/s instead of 85 (set a: 84.7 / 58.7 / 103.8, set b: 95.1 / 57.7 / 140.6). Files: [runs/2026-10-18/kitbench-decode](runs/2026-10-18/kitbench-decode/).

#### Concurrency (released engine)

| Concurrent requests | Prose, aggregate decode | Code, aggregate decode | Code, per request |
|---:|---:|---:|---:|
| 1 | 29.6 tok/s (includes a 7.3 s TTFT on the first request) | 94.1 tok/s | 95.8 tok/s |
| 2 | 73.5 tok/s | 106.6 tok/s | 57.2 tok/s |
| 3 | 82.3 tok/s | 124.3 tok/s | 43.5 tok/s |
| 4 | **94.4 tok/s** | **140.3 tok/s** | 36.8 tok/s |

- **Prefix cache.** A second question about the same 24K-token document: TTFT **12.8 s cold, 1.1 s warm** (22,528 of 24,210 prompt tokens cached).
- **After a restart** (`SPILL_GIB=64`, on by default, keeps cached prompts on disk): TTFT for an 18K-token prompt **2.7 s** instead of 16.8 s.

Method, definitions and every run: [Benchmarks](docs/BENCHMARKS.md).

</details>

<details>
<summary><b>Exact versions</b></summary>

| What | Version |
|---|---|
| Model weights | `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` @ `afc7aa0a` (43 files) |
| Engram tables | `deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`, files 47 and 48 |
| Engine | [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold) @ `07aa02585f64` (TensorFold 0.6.6 + bertholomus's V4.1 engine + PRs #1-#4 and #6) |
| Base image | `nvcr.io/nvidia/pytorch:26.07-py3` |
| Chat template | MiaAI kit `files/chat_template.jinja` @ `6f7d1590` + [our patch](runs/2026-10-17/template-patch.diff) |
| vLLM comparison | MiaAI kit @ `6f7d1590`, image `sha256:2f0cf3ad...` |
| Tested on | DGX OS (Ubuntu 24.04.5), driver 580.178.04, CUDA 13.0, Docker 29.6.2 |

Every checksum, file size and computer detail: [PROVENANCE](runs/2026-10-17/PROVENANCE.md).

</details>

<details>
<summary><b>What is in this folder</b></summary>

| Path | What |
|---|---|
| `start.sh`, `stop.sh`, `status.sh`, `verify.sh` | run the server |
| `scripts/` | download, template fix, setup checks, safe engine install (`_raw_*`: the scripts as they ran on the tested pair) |
| `docker/` | the engine build |
| `tests/` | the 32 agent tests, the format check, the benchmark, the prompt tests, the Hermes test ([guide](tests/README.md)) |
| `runs/2026-10-17/` | every log and result behind this page |
| `docs/` | how to connect your app, the full story, benchmarks, network, operations, known limits |

</details>

---

## Credits and licence

The display-backed KV cache and SSD spill tier are modifications built on
bertholomus's V4.1 engine. The display cache maps the compressed KV and indexer
planes into the NVIDIA display reservation, adds a shared pool with a per-request
context limit, and exposes pool and spill telemetry through `/health` and
`/metrics`. The SSD spill tier reuses the GLM spill implementation with
display-safe staging copies, CRC-checked restores, eviction, shutdown draining
and restart persistence. These changes were developed and validated by
**opencode** (the DeepSeek V4.1 Flash coding agent) on top of bertholomus's
engine. **GPT Sol** fixed the improper pre-OOM SIGTERM handling, so earlyoom
stops the engine cleanly instead of leaving the lane broken.

This recipe stands on the work of DeepSeek (the model), LibertAIDAI (the compressed weights), turboderp (ExLlamaV3), ashhart and the TensorFold contributors, **bertholomus (the fast V4.1 engine this recipe runs)**, willjriley and Capicua25x (engine fixes), MiaAI-Lab (the vLLM kit and the template) and tonyd2wild (the benchmark method). [Full credits](CREDITS.md)

The scripts, tests and documents here are Apache-2.0 ([LICENSE](LICENSE)). The model, the template and the engine keep their own licences.
