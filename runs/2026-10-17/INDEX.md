# Run evidence of 2026-10-17

All times are UTC. All runs used the same model view (see `PROVENANCE.md`) on the 2x DGX Spark pair, TP=2.
The test tools are in `../../tests/`. `v41_suite.py` made the `*_suite*` files. `v41_grammar_check.py` made the grammar checks.
`v41_bench.py` made the benchmark files. `hermes_thinking_modes.py` made the Hermes files.

The TensorFold image tag is the first 12 hex digits of the TensorFold commit that was built.
For `02`, `03` and `04` the image is inferred from the image build time and the server start time (the gate logs started with run `05`).
The commit subjects agree with the results (for example, the schema tests pass only after the structured-output commits).

## Files

| File | What it is | Image / engine | Time (UTC) |
| --- | --- | --- | --- |
| `01_vllm-miaai-kit_suite_32of32.log` | Console output of the 32-test suite, vLLM baseline. 32/32 passed in 602 s. | vLLM, MiaAI kit image `sha256:2f0cf3ad...` (commit `6f7d1590`) | about 18:46:35Z to 18:56:37Z |
| `01_vllm-miaai-kit_suite-results_32of32.json` | Per-test results of the same run. | same | same |
| `02_tensorfold-469171515cd3_suite_run1_25of32.log` | First TensorFold run. 25/32 passed in 291 s. Failed: `tools/required`, `tools/named`, and the 5 `schema/*` tests (no structured output yet). | TensorFold `469171515cd3` | 20:14:27Z to 20:19:18Z |
| `02_tensorfold-469171515cd3_suite-results_run1_25of32.json` | Per-test results of the same run. | same | same |
| `03_tensorfold-4a413d9c732b_suite_run2_27of32.log` | Second run. 27/32 passed in 294 s. `tools/required` and `tools/named` pass now. Failed: the 5 `schema/*` tests. | TensorFold `4a413d9c732b` | 20:35:34Z to 20:40:28Z |
| `03_tensorfold-4a413d9c732b_suite-results_run2_27of32.json` | Per-test results of the same run. | same | same |
| `04_tensorfold-90c2d59c8179_suite_run3_28of32.log` | Third run. 28/32 passed in 278 s. `schema/json_object` passes now. Failed: `schema/strict_off`, `schema/strict_think`, `schema/choice`, `schema/regex`. | TensorFold `90c2d59c8179` | 20:50:16Z to 20:54:54Z |
| `04_tensorfold-90c2d59c8179_suite-results_run3_28of32.json` | Per-test results of the same run. | same | same |
| `05_tensorfold-7f80c3d267da_suite_32of32.log` | Suite on the grammar image. 32/32 passed in 285 s. | TensorFold `7f80c3d267da` | 21:39:28Z to 21:44:13Z |
| `05_tensorfold-7f80c3d267da_suite-results_32of32.json` | Per-test results of the same run. | same | same |
| `06_tensorfold-7f80c3d267da_grammar-check_14of14.log` | `--parallel` grammar check, 2 rounds of 7 requests (json_schema off/think, json_object, regex, choice, plain prose, plain code). 14/14 passed. | TensorFold `7f80c3d267da` | 21:44:13Z to 21:44:38Z |
| `07_tensorfold-7f80c3d267da_bench.log` | Concurrency benchmark (prose and code, 1 to 4 streams) and prefix-cache check (24 k-token prompt: cold 13.02 s, warm 1.87 s with 22,528 cached tokens). | TensorFold `7f80c3d267da` | 21:44:38Z to 21:47:09Z |
| `07_tensorfold-7f80c3d267da_bench-results.json` | Numbers of the same benchmark. | same | same |
| `08_tensorfold-7f80c3d267da_hermes-agent-gate_35of35.txt` | Console output of the Hermes agent gate (the Hermes `AIAgent` at every thinking level, stream and quiet, math and tool tasks, title task). 35/35 passed. | TensorFold `7f80c3d267da`, through the router | files copied to the head node at 21:54:25Z (run time not recorded; the image served from 21:39Z to 22:19Z) |
| `08_tensorfold-7f80c3d267da_hermes-agent-gate_35of35.json` | Per-case rows of the same run. | same | same |
| `09_tensorfold-d5c9d3cdf4fa_suite_32of32.log` | Suite on the image with the `--parallel` lane fix. 32/32 passed in 324 s. | TensorFold `d5c9d3cdf4fa` | 22:24:17Z to 22:29:41Z |
| `09_tensorfold-d5c9d3cdf4fa_suite-results_32of32.json` | Per-test results of the same run. | same | same |
| `10_tensorfold-d5c9d3cdf4fa_grammar-check_14of14.log` | `--parallel` grammar check, same as `06`. 14/14 passed. | TensorFold `d5c9d3cdf4fa` | 22:29:41Z to 22:30:07Z |
| `11_tensorfold-fd2365042f56_suite_32of32.log` | Suite on the final image. 32/32 passed in 295 s. | TensorFold `fd2365042f56` | 22:58:23Z to 23:03:18Z |
| `11_tensorfold-fd2365042f56_suite-results_32of32.json` | Per-test results of the same run. | same | same |
| `12_tensorfold-fd2365042f56_grammar-check_14of14.log` | `--parallel` grammar check, same as `06`. 14/14 passed. | TensorFold `fd2365042f56` | 23:03:18Z to 23:03:43Z |
| `13_tensorfold-fd2365042f56_bench.log` | Concurrency benchmark (prose and code, 1 to 4 streams) and prefix-cache check (24 k-token prompt: cold 12.76 s, warm 1.07 s with 22,528 cached tokens). | TensorFold `fd2365042f56` | 23:03:43Z to 23:06:06Z |
| `13_tensorfold-fd2365042f56_bench-results.json` | Numbers of the same benchmark. | same | same |
| `14_tensorfold-fd2365042f56_hermes-agent-gate_35of35.txt` | Console output of the Hermes agent gate, same as `08`. 35/35 passed. | TensorFold `fd2365042f56`, through the router | 23:06:27Z to 23:08:23Z |
| `14_tensorfold-fd2365042f56_hermes-agent-gate_35of35.json` | Per-case rows of the same run. | same | same |
| `15_tensorfold-fd2365042f56_deploy-gate.log` | Log of the gate run for `11`, `12` and `13` (`deploy_gate.sh`): the smoke test, and the result line of each step. | TensorFold `fd2365042f56` | 22:58:14Z to 23:06:06Z |

Run `05`, `06` and `07` are one gate run (`deploy_gate.sh`, log lines `SMOKE_OK` 21:39:28Z, `SUITE_EXIT=0` 21:44:13Z, `CHECK_EXIT=0` 21:44:38Z, `BENCH_EXIT=0` 21:47:09Z).
Run `09` and `10` are the gate run for `d5c9d3cdf4fa`. That gate skipped the benchmark (`GATE_DONE (bench skipped)` at 22:30:07Z).
Run `11`, `12` and `13` are the gate run for `fd2365042f56`, the final image (`deploy_gate.sh`, log `15`: `serving` 22:58:14Z, `SMOKE_OK` 22:58:23Z, `SUITE_EXIT=0` 23:03:18Z, `CHECK_EXIT=0` 23:03:43Z, `BENCH_EXIT=0` 23:06:06Z, `GATE_DONE` 23:06:06Z). Run `14` started after `GATE_DONE`, with the image still serving (start and end time from the shell that ran it).
The time of run `01` is the end time of the original `suite.log` on the head node (the `01` files on the head node are later copies), minus the 602 s run time.

## Runs after the release gate (image `fd2365042f56`)

| File | What it is | Time (UTC) |
| --- | --- | --- |
| `16_kit-bench-bertholomus_tensorfold-fd2365042f56/` | bertholomus's `kit_bench.py` (from `tools/dsv41/` of the engine source) against this server, with their published settings: decode sets a and b, 2 and 4 streams (9 reps), 4 streams sustained 90 s, cold prefill 8K-128K, decode after a 128K prompt (3 runs). `kb.log` is the console; one JSON file per step; `run_kitbench.sh` and `src127.py` are the runner. | 23:08Z to 23:24:41Z |
| `17_tuning_warmup-spill-drafts/` | One-change-at-a-time tuning: longer warm-up list (no gain), spill across a restart (kept), draft depth 4 (slower, not kept); then the suite (32/32) and the grammar check (14/14) on the kept setting. Read `NOTES.md` first. | 23:25Z to 23:45Z |
| `18_tensorfold-fd2365042f56_conformance_3of3-api-cases.log` | `tests/v41_conformance.py` in an offline container of the image. 3 of 5 fixtures match; the 2 that differ (cases 3 and 4) use app-only parts (`latest_reminder`, `<｜action｜>`) that the API does not send. All 3 API-role cases match. | about 23:50Z |
| `18_tensorfold-fd2365042f56_differential_12of12.log` | `tests/v41_differential.py` in the same container: 12 of 12 API-shaped requests render byte-identical to DeepSeek's `encoding.py` (0 skipped). | about 23:51Z |

## Other files in this folder

| File | What it is |
| --- | --- |
| `PROVENANCE.md` | Where the weights, the Engram tables, the template, the images and the node facts came from. |
| `weights.sha256` | sha256 sums of the view files (`sha256sum -c` format, paths relative to the view root). |
| `template-patch.diff` | Unified diff of our change to the kit chat template (the template file itself is not in this repo). |
| `encoding-fixtures/` | The five DeepSeek conformance fixtures (`test_input_N.json`, `test_output_N.txt`) for `tests/v41_conformance.py`. Copied unchanged from `encoding/tests/` of `deepseek-ai/DeepSeek-V4.1-Flash-1017` (MIT license). |

## Changes made to the copied files

- `/home/<user>` paths were replaced with `~` in the logs. In `08_..._hermes-agent-gate_35of35.txt` the router host name was replaced with `<router-host>`.
- The two Hermes files (`08_*`) had Windows line ends. They now have Unix line ends. No other content was changed.
- The suite prompt in `tests/v41_suite.py` (a calendar-tool test) contains the placeholder addresses `ana@example.com` and `raj@example.com`. They are example.com addresses, not real ones. They also appear in the results files because they are model output. They were left unchanged.
