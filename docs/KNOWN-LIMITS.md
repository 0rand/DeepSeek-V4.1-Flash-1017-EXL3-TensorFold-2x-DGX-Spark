# Known limits

This page lists what this recipe does not measure, does not support, and does not ship.
Each item was checked against the files in this repo. The file that supports the item is named.

## Not measured

1. **Image input.**
   - The server starts with `--vision` by default (`VISION=1` in `.env.example`).
   - No test in `tests/`, no check in `verify.sh` and no Hermes case sends an image to the server. The conformance test only renders the image placeholder in a prompt.
   - Not measured: image decode speed, two requests with images at the same time, many images in one request.
2. **vLLM under concurrency.** The vLLM kit ran with `MAX_NUM_SEQS=1` (`runs/2026-10-17/PROVENANCE.md`, section 4). It serves one request at a time. No vLLM result exists for 2 to 4 streams, so the TensorFold concurrency tables have no baseline.
3. **KV cache bytes per token.** The run files do not hold the KV pool size in tokens, for either engine. See [BENCHMARKS.md](BENCHMARKS.md).
4. **Context above 200K tokens.**
   - The server is set to 1,048,576 tokens (`--context 1048576`).
   - The longest measured prompt is the 200K needle test (200,030 tokens). No test reaches 1M tokens.
   - The vLLM baseline was set to 262,144 tokens (`MAX_MODEL_LEN`).
5. **Output quality.** The repo has no perplexity or accuracy test. The suite checks a few fixed tasks. No test compares TensorFold output with vLLM output. The perplexity numbers on the model card are for the weights, not for this engine.

More that is not measured:

- **Replies under load.** bertholomus's release gates check that a reply under `--parallel 4` equals its solo reply. No test in this repo repeats that check.
- **Long-prompt prefill is slower than bertholomus's published numbers.** With their own client, this recipe gives 1,707 / 1,457 / 1,420 tok/s at 32K / 64K / 128K; their Python engine published 2,421 / 2,363 / 2,242 ([runs/.../16_*](../runs/2026-10-17/16_kit-bench-bertholomus_tensorfold-fd2365042f56/)). The weights differ (1017 vs the original V4.1), and so do the nodes. The cause is not known.
- **More than four streams.** The benchmark stops at four streams, which is the `--parallel` setting.
- **Prefix-cache use in the speed rows.** The suite does not record `cached_tokens`, so a cache hit in the 32K prefill and the 200K needle rows cannot be ruled out. See "Prefill" in [BENCHMARKS.md](BENCHMARKS.md).
- **GPU clock and link state.** No run file records the GPU clock or the link bandwidth before a run.
- **The conformance and differential tests.** No stored output of `tests/v41_conformance.py` or `tests/v41_differential.py` exists in `runs/` (`PROVENANCE.md`, section 7). The "12/12" and "3/3" results in `WHAT-I-CHANGED.md` have no log in this repo.
- **Measured on one pair only.** All results come from one pair of DGX Spark, on one day.

## Not supported and not shipped

1. **No boot unit.** The server starts by hand (`./start.sh`). The repo has no systemd unit. [OPERATIONS.md](OPERATIONS.md) describes the rules for one.
2. **No automatic restart.** A device fault (CUDA error, out of memory) or a lost rank breaks the lane. Both ranks then exit with code 75 after 5 s. They stay down until you run `./stop.sh && ./start.sh` or a supervisor does it.
3. **No API key in the tests.** The test scripts send no `Authorization` header to the engine. With `API_KEY_FILE` set, the engine tests get HTTP 401.
4. **Living Weights is not available.**
   - It is the `--slide` mode of TensorFold. It ships in upstream TensorFold 1.0.3 and later (the Zig engine).
   - It learns only into Nemotron 3.5 Lightning, MLX 4-bit.
   - It needs about 26 GiB of extra GPU memory on a GB10.
   - It rewrites a single-node model folder.
   - This recipe runs the Python 0.6.6 line, with EXL3 weights, across two nodes.
5. **Only TP=2 on two nodes.** The launcher has no setting for one, three or four nodes.

## Costs and requirements

1. **The engine is a fork, pinned at a commit.**
   - The recipe builds [Elevatormusic/TensorFold](https://github.com/Elevatormusic/TensorFold) at `fd2365042f56`. The upstream project is `ashhart/TensorFold`.
   - The changes are pull requests #1 to #3 on the fork. The files in this repo do not show that upstream accepted them.
2. **The first start is slow and large.** The first start writes the weight cache of each rank to `CACHE_DIR`. This takes about 15 minutes and about 100 GiB per node (`.env.example`; `scripts/preflight.sh` asks for 110 GiB free). Later starts take 2 to 4 minutes.
3. **The first long prompt pays a warm-up.** The first prompt of 8K tokens or more after a start takes about 10 s longer ([OPERATIONS.md](OPERATIONS.md)). The 8K prefill row in [BENCHMARKS.md](BENCHMARKS.md) includes this cost.
4. **The Engram tables are large.** The two 1017 Engram shards are about 101 GB each. The model folder on each node holds about 196 GiB of weights and about 189 GiB of Engram tables (`PROVENANCE.md`, section 1).
5. **The weights revision is not the newest.**
   - The measured revision of `LibertAIDAI/DeepSeek-V4.1-Flash-1017-EXL3-2.9bpw` is `afc7aa0a`. The files were downloaded before 17:47Z on 2026-10-17.
   - The model card had revision `c178a012` at 18:33Z the same day.
   - This repo did not compare the two revisions. It did not run the tests on the newer one.

## Tests that need other systems

- Two suite tests (group `router`) need a router in front of the engine. Without a router key, `tests/run_all.sh` skips them and runs 30 tests.
- The Hermes agent gate needs a Hermes Agent install and a router. The results in `runs/2026-10-17/` came from the router of the author. The repo does not ship that router.
- The differential test needs DeepSeek's `encoding.py`, which this repo does not hold. See [tests/README.md](../tests/README.md).
