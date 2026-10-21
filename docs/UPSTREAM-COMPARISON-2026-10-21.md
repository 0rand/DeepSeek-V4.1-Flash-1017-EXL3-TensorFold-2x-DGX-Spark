# Upstream comparison: current repo vs `recipe/main`

Date: 2026-10-21
Scope: local `main` (`02d53ac`) and `origin/main` (`76cc1b2`) against the original upstream `recipe/main` (`1b3066d`), with the fork point at `3b6d540` (2026-10-18, merge of PR #5 `thp-spill-direct`).

## 1. Executive summary

The original repo has moved substantially since our fork point. It is **96 commits ahead** of the fork point, while our local `main` is **32 commits ahead** (23 on `origin/main` plus 9 unpushed local commits). The two histories diverged at `3b6d540`.

Upstream has done two things:

1. **Merged six of our split PRs** (#28-#33) in refactored, deliberately narrowed form.
2. **Upgraded the core engine to release `dsv41-1.1.0`** (`730d41abd8f4`) and added a large amount of repo hygiene, CI, tests and documentation.

The improvements are real and worth taking, but **a single `git merge recipe/main` is a bad idea**: it produces **23 conflicted files**, mostly because upstream's display-KV experiment is a different, display-only implementation than our monolithic one.

**Recommendation:** create a separate branch from pristine `recipe/main`, rewire our monolithic display-KV functionality onto it, and keep `main` untouched as the known-working rollback. Do not merge `origin/PR-branch`; it is an older reunification attempt that lacks our latest streaming tool-validation work.

## 2. Repository topology

| Ref | Meaning | Tip | Position |
|---|---|---|---|
| `3b6d540` | fork point (v1.0.0, PR #5) | 18 Oct | common ancestor |
| `origin/main` | our GitHub fork | `76cc1b2` | 23 commits ahead of fork point |
| local `main` | working directory | `02d53ac` | 9 commits ahead of `origin/main` (unpushed) |
| `recipe/main` | original upstream | `1b3066d` | 96 commits ahead of fork point |
| `origin/PR-branch` | old reunification branch | `251b0ee` | 32 ahead of `recipe/main`, 10 behind local `main` |
| `origin/split/display-ssd-spill` | in-flight upstream branch | `c9151e8` | 26 ahead of `recipe/main` |
| `origin/split/tool-output` | in-flight upstream branch | `879d943` | 31 ahead of `recipe/main` |

Upstream tags: `v1.0.0` only. The engine release tag lives on the TensorFold repo, not this recipe repo.

## 3. Upstream improvements

### 3.1 Core engine upgrade: `dsv41-1.1.0` (`730d41abd8f4`)

This is the single most valuable change. It replaces our pinned `07aa02585f64` (engine 1.0.0).

| Engine PR | Change | Why it matters to us |
|---|---|---|
| #19 | No decode graph is captured while serving; warm-up captures every graph the window can use; `TF_DS_CAPTURE_MODE=thread_local` | Fixes `cudaErrorStreamCaptureInvalidated` / CUDA allocation failures in long parallel sessions. Relevant to our `PARALLEL=8` production setting. |
| #21 | Lower host-memory peak: Engram cross-rank sum in 512-row blocks (`TF_DS_ENGRAM_BLOCK`), trim free torch blocks over 256 MiB when idle (`TF_DS_TRIM_MIB`), admission wait under floor + 1 GiB (`TF_DS_ADMIT_GIB`) | Addresses recipe issue #21 (earlyoom stopping the head). Our production `MEM_FLOOR_GIB=1.0` is below earlyoom's 1 GiB limit. |
| #24 | Memory gate counts only cached bytes a trim can give back | Fixes a gate-run earlyoom stop; part of the 1.1.0 release. |
| #9 | RDMA queue pairs take path MTU from the port, not fixed 4096 | Fixes `transport retry counter exceeded` at netdev MTU 1500. |
| #13 | Lane streams found by identity | Overlapping image requests no longer crash the lane. |
| #14 | `--vision-max-images` applies to V4.1 image input | Vision limit now enforced. |
| #20 | Draft ranking across streams (`TF_DS_DEPTH_POLICY=rank`, new default) | +3.0-7.0% decode with 4 streams in the gate run; 1 stream unchanged. |
| #15 | Opt-in `TF_DROP_UNOFFERED_TOOL_CALLS=1` | Drops a call to a tool the request did not offer, also while streaming. Upstream turns it on by default in the recipe. |
| #8 | Dense BF16 overrides | Not used by the default recipe. |
| #18 | 8-key tile search for prompt indexer top-k | Prefill +0.6 / +0.8 / +2.0 / +3.6% at 8K / 32K / 64K / 128K. |
| #12 | CI: CPU tests, lint, workflow checks, secret scan | Repo hygiene. |
| #16 | TLS 1.2 minimum for image URL fetches | Security hardening. |

**Gate run (2026-10-21, image `fc9200a44463`, same tree as `730d41abd8f4`):**

| Check | 1.1.0 | Baseline `3c45bdaeab7d` |
|---|---|---|
| Suite | 32/32 | 32/32 |
| Grammar, `--parallel 2` | 14/14 | 14/14 |
| Decode 1 stream, prose / code | 51.9 / 99.7 tok/s | 52.7 / 100.4 tok/s (noise) |
| Decode 4 streams, prose per stream / aggregate | 26.8 / 99.3 tok/s (+5.9% / +3.3%) | 25.3 / 96.1 tok/s |
| Decode 4 streams, code per stream / aggregate | 39.7 / 145.7 tok/s (+7.0% / +3.0%) | 37.1 / 141.5 tok/s |
| Prefill 32K | 2,751 tok/s, TTFT 11.63 s | 2,717 tok/s, TTFT 11.78 s |
| Prefix cache 24K cold / warm TTFT | 11.18 s / 1.05 s | 11.43 s / 1.10 s |
| Long 4-stream run (20K/50K/100K/200K) | 0 errors, identical tokens | 0 errors, identical tokens |
| Late graph captures while serving | 0 | not counted |

**Known issue:** head `MemAvailable` under 4 long streams was 1.45 GiB (1.1.0) vs 1.65 GiB (baseline), against the 1 GiB earlyoom limit. Upstream advises `PREFILL_CHUNK=1024` or `PARALLEL=2` when the head has little free memory. This matters for our production `PARALLEL=8`.

### 3.2 Our PRs that upstream merged (in narrowed form)

| Upstream PR | Branch | What was merged | What was deliberately left out |
|---|---|---|---|
| #28 | `split/launcher-config` | `PREFILL_CHUNK` validation in `start.sh` | Nothing major |
| #29 | `split/nccl-bootstrap` | Opt-in `PREFLIGHT_NCCL_BOOTSTRAP=1` preflight check with timeout/KILL | Our always-on NCCL bootstrap check |
| #30 | `split/display-kv` | Display-only KV experiment: `apply.py` patches only `model.py`, `engine.py`, `multi.py`; spill forced off; prefill 512 | Our spill tier, tool-output guard, memory guard, cache metrics, bounded top-k |
| #31 | `split/cache-metrics` | Standalone opt-in cache-metrics experiment | Our bundled cache metrics inside display-KV |
| #33 | `split/memory-topk` | Standalone opt-in memory/top-k experiment (`MEM_CAP=0` default, opt-in allocator policy) | Our default-on memory guard and 1.5 GiB floor enforcement |

Upstream accepted the ideas but split our monolithic experiment into separate, opt-in pieces. That is the central architectural difference.

### 3.3 New launcher and preflight features

- **`THINKING` and `REASONING_EFFORT`** in `.env`, default **off** (`THINKING=0`, `REASONING_EFFORT=` empty). Our local default is `THINKING=1`, `REASONING_EFFORT=high`.
- **`DROP_UNOFFERED_TOOL_CALLS=1`** default in `.env.example` and `start.sh`; passes `TF_DROP_UNOFFERED_TOOL_CALLS` to both ranks. Only 0/1 valid.
- **`PREFILL_CHUNK` validation**: positive integer, stops `start.sh` before preflight/containers on bad values.
- **`MEM_CAP`** replaces our `MEM_ALLOCATOR_CAP`; only forwards `TF_DS_MEM_CAP=1` when the selected image carries the `tensorfold.memory-topk` label on both nodes.
- **Preflight driver warning**: `[warn]` when the NVIDIA driver is older than the tested `580.178.04` (issue #24 node-hang signature).
- **Preflight RoCE MTU check**: fails when `active_mtu` is not 4096 or netdev MTU is below ~4200 (issue #15).
- **Preflight NCCL bootstrap** is now opt-in (`PREFLIGHT_NCCL_BOOTSTRAP=1`), with `timeout --kill-after=5s 20s` and `--init`.
- **`scripts/build_extensions.sh` executable bit** fixed (issue #15).
- **`scripts/display_drm.sh`**: `rmmod nvidia_drm` before restore when the card does not reappear.
- **`start-display-kv.sh`**: `sudo -n true 2>/dev/null || sudo -v` fix for NOPASSWD head sudo.

### 3.4 Repo hygiene, CI and tests

Upstream added a full open-source repo structure that our fork does not have:

- `.github/`: `ci.yml` (shellcheck, ruff, offline tests, markdown links, actionlint, zizmor), `secret-scan.yml`, `issue-labels.yml`, `pr-template.yml`, `links-external.yml`, issue forms, PR template, CODEOWNERS, FUNDING, dependabot, `requirements-ci.txt`.
- `CONTRIBUTING.md`, `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CHANGELOG.md`, `ruff.toml`.
- New offline tests: `tests/test_defaults.py` (protects `.env.example` defaults and `start.sh` forwarding), `tests/test_machine_values.py` (home paths, public IPs, Windows paths), `tests/test_nccl_bootstrap.py`, `tests/test_prefill_chunk.py`, `tests/test_repo_rules.py` (runs layout, executable bits, `.env.example` sources cleanly, no secrets), `tests/run_offline.sh`.
- `tests/README.md` rewritten for the offline runner and the split experiments.

**Important:** `tests/test_defaults.py` hard-codes upstream defaults (`THINKING=0`, `TF_COMMIT=730d41abd8f4`, `IMAGE=tensorfold-dsv41:730d41abd8f4`, `PARALLEL=4`, `SPILL_GIB=64`, `BIND_HOST=127.0.0.1`). It will fail against our `.env.example` unless we adopt upstream defaults or update `PROTECTED`.

`tests/test_repo_rules.py` requires:

- `runs/` tracked by git, each entry a `runs/YYYY-MM-DD/` folder with `INDEX.md`;
- no `runs/*` in `.gitignore`;
- scripts executable;
- `.env.example` sources cleanly and holds no token.

Our local repo violates this: `.gitignore` still has `runs/*` and `/data/benchmarks.sqlite*`, and we have `runs/2026/10/*` (old layout).

### 3.5 Documentation and saved runs

- `docs/OPERATIONS.md`: new "Prefill configuration" section, "Engine switches of release 1.1.0" (decode graphs, host memory, draft ranking), expanded out-of-memory section with measured traces from `runs/2026-10-20/`, node-hang signature from issue #24, `PARALLEL=2` / `PREFILL_CHUNK=1024` advice, "do not raise `MEM_FLOOR_GIB`".
- `docs/NETWORK.md`: MTU 9000 section with netplan example, `ping -M do -s 8972`, `ibv_devinfo` check.
- `docs/CONNECT.md`: API key examples, thinking-off default, "set thinking before comparing benchmark scores".
- `README.md`: disk need corrected to ~650 GB (414 GB model + 110 GiB cache + 64 GiB default spill), engine pin updated, display-KV section removed (moved to opt-in docs), EngineMetrics section removed.
- `docs/KNOWN-LIMITS.md`, `docs/BENCHMARKS.md`, `docs/WHAT-I-CHANGED.md` section 7.
- Saved runs: `runs/2026-10-19/display-kv-tester-ab.md`, `runs/2026-10-20/` OOM traces, `runs/2026-10-21/gate-1.1.0/` gate report.

### 3.6 In-flight upstream branches (not merged yet)

| Branch | Tip | Content |
|---|---|---|
| `split/display-ssd-spill` | `c9151e8` (21 Oct) | New `experiments/display-spill/` with `spill_tier.py` **format 4** (1578 lines): header CRC (`hcrc`), base64 int64 token ids, max header 64 MiB, writer readiness, start-up agreement, trim kept builds, shared display lock, `docs/SPILL.md`, `start-display-spill.sh`. A more robust version of our local spill tier (format 2, 1240 lines). Two later fixes: read the driver from a second source and create a missing root for its owner; count an old build folder that cannot be read. |
| `split/tool-output` | `879d943` (21 Oct) | New `experiments/tool-output/` with a simpler 241-line guard that rejects complete unoffered calls. Our local `tool_output.py` is 729 lines and much more advanced (atomic validation, schema checks, literal containers, streaming isolation). A later fix counts open markup in source order, in each block. |
| `PR-branch` | `251b0ee` (20 Oct) | Old reunification branch: upstream main + our local work + review fixes, but 10 commits behind local `main` (lacks our latest streaming tool-validation work), keeps engine 1.0.0, `THINKING=0`, deletes the `runs/2026/10` layout. Not a good merge target. |

## 4. What our repo has that upstream does not

| Area | Our local state | Upstream state |
|---|---|---|
| Display-KV | Monolithic experiment: display KV + spill tier + tool-output guard + memory guard + cache metrics + bounded top-k, all in one image | Display-only experiment (spill forced off, no tool guard, no memory policy, no cache metrics, no bounded top-k); separate opt-in experiments for the rest |
| Tool-output guard | 729-line atomic/fail-closed validator with schema checks, literal containers, streaming isolation; `docs/TOOL-OUTPUT.md`; live streaming probe `tests/live_tool_streaming.py` | 241-line simpler unoffered-call guard in the in-flight `split/tool-output` branch; no live streaming probe |
| Spill tier | 1240-line format 2; inherits `SPILL_GIB` from `.env`; display-safe staging | 1578-line format 4 in the in-flight `split/display-ssd-spill` branch; header CRC, base64 ids, writer readiness; forces `SPILL_GIB=0` in the display-only experiment |
| Memory guard | Default `TF_DS_MEM_CAP=1`, floor `max(1.5, floor_gib)` | Opt-in `MEM_CAP=0`, floor as given (no 1.5 enforcement) |
| Cache metrics | Bundled in display-KV | Separate opt-in `experiments/cache-metrics` |
| Bounded top-k | Bundled in display-KV | Separate opt-in `experiments/memory-topk` |
| Local tests | `tests/live_tool_streaming.py`, `live_tool_streaming_test.py`, `thinking_settings_test.py`, `tool_call_probe.py`, `live_tool_hardening.py` | No equivalents; upstream has `test_defaults.py`, `test_machine_values.py`, `test_nccl_bootstrap.py`, `test_prefill_chunk.py`, `test_repo_rules.py`, `run_offline.sh` |
| Local docs | `docs/TOOL-OUTPUT.md`, display-KV spill settings in `docs/DISPLAY-KV.md` and `README.md` | `docs/SPILL.md` (in-flight), `docs/MEMORY-TOPK.md`, `docs/DISPLAY-KV.md` (display-only) |
| Local runs | `runs/2026-10-21/parser-guardrails/`, `runs/2026/10/*`, `runs/tool-hardening-live-*.json` | `runs/2026-10-19/display-kv-tester-ab.md`, `runs/2026-10-20/`, `runs/2026-10-21/gate-1.1.0/`; requires `runs/YYYY-MM-DD/` layout |
| Production `.env` | `TF_COMMIT=07aa02585f64`, `IMAGE=tensorfold-dsv41:07aa02585f64`, `PARALLEL=8`, `SPILL_GIB=32`, `MEM_FLOOR_GIB=1.0`, `MEM_ALLOCATOR_CAP=0`, `TEMPERATURE=0.4`, `PORT=8100`, `BIND_HOST=0.0.0.0`, `RESTART_POLICY=unless-stopped`, `THINKING=1` | `.env.example`: `TF_COMMIT=730d41abd8f4`, `PARALLEL=4`, `SPILL_GIB=64`, `MEM_FLOOR_GIB=1.5`, `MEM_CAP=0`, `TEMPERATURE=1.0`, `PORT=8888`, `BIND_HOST=127.0.0.1`, `RESTART_POLICY=no`, `THINKING=0` |

## 5. Conflict analysis

A trial merge of `recipe/main` into local `main` (`git merge-tree --write-tree`) produces 23 conflicted files:

```text
.env.example
.gitignore
CREDITS.md
README.md
docs/CONNECT.md
docs/DISPLAY-KV.md
docs/OPERATIONS.md
experiments/display-kv/NOTICE.md
experiments/display-kv/README.md
experiments/display-kv/apply.py
experiments/display-kv/overlay/display_kv.py
experiments/display-kv/prepare.py
experiments/display-kv/probe.py
experiments/display-kv/tests/test_bundle.py
experiments/display-kv/tests/test_display_kv.py
experiments/display-kv/tests/test_integration.py
runs/2026-10-21/INDEX.md
scripts/display_drm.sh
scripts/preflight.sh
start-display-kv.sh
start.sh
tests/display_drm_test.py
tests/start_display_kv_test.py
```

The conflicts fall into four groups:

### 5.1 Configuration and launcher

- `.env.example`: engine pin, `THINKING`/`REASONING_EFFORT`/`TOOL_OUTPUT_STREAMING` vs upstream's `THINKING=0`/empty effort/`DROP_UNOFFERED_TOOL_CALLS`, `MEM_ALLOCATOR_CAP` vs `MEM_CAP`, duplicate `PREFILL_CHUNK` line.
- `start.sh`: validation blocks, `TF_DS_MEM_CAP` handling, `TF_DROP_UNOFFERED_TOOL_CALLS`, thinking default, conditional `--reasoning-effort`.
- `scripts/preflight.sh`: NCCL bootstrap always-on vs opt-in, MTU check, driver warning.
- `.gitignore`: `runs/*` and `/data/benchmarks.sqlite*` vs upstream's removal.
- `scripts/display_drm.sh`: upstream's `rmmod nvidia_drm` fix.
- `start-display-kv.sh`: upstream's sudo fix.

### 5.2 Display-KV experiment

Add/add conflicts because upstream's `experiments/display-kv` is a different, narrower implementation:

- `apply.py` patches only `model.py`, `engine.py`, `multi.py`; ours patches those plus `kernels.py`, `health.py`, `metrics.py`, `app.py`, `server.py`, `http.py`.
- `prepare.py` forces `SPILL_GIB=0` upstream; ours inherits `SPILL_GIB`.
- `README.md`, `NOTICE.md`, `probe.py`, `tests/*` differ.

Resolution: keep our monolithic experiment. Upstream's version would remove spill, tool-output validation, memory guard, cache metrics and bounded top-k from the display lane.

### 5.3 Documentation

- `README.md`: upstream removes the display-KV and EngineMetrics sections, updates disk need and thinking text.
- `docs/DISPLAY-KV.md`: upstream describes the display-only variant; ours describes the full variant.
- `docs/OPERATIONS.md`: upstream removes the display-KV section, adds engine switches and the expanded OOM section.
- `docs/CONNECT.md`: thinking default and API key examples.
- `CREDITS.md`: display cache/spill credit vs display-mapping credit.

Resolution: merge both, but rewrite the display-KV docs for our variant and keep a separate section for upstream's display-only variant if it stays in the tree.

### 5.4 Runs

- `runs/2026-10-21/INDEX.md`: add/add conflict (ours: parser-guardrails; upstream: gate-1.1.0). Merge both entries.
- Our `runs/2026/10/*` layout violates upstream's `runs/YYYY-MM-DD/` rule. Move to `runs/2026-10-18/display-kv-bench/` and add an `INDEX.md` entry.

## 6. Recommendation: rewire branch, keep main untouched

### 6.1 Lane A (chosen)

Keep our monolithic display-KV lane and add upstream's split experiments as optional extras. Our production lane depends on the monolithic image (`tensorfold-dsv41:display-kv-58e319cb2866`), and upstream's `display-spill` and `tool-output` branches are not merged yet.

### 6.2 Branch strategy

- Keep `main` as it is now (`02d53ac`) for rollback.
- Create a new branch from **pristine `recipe/main`** (not a merge commit), for example `rewire/upstream-1.1.0-monolithic`.
- Port our monolithic functionality onto it file by file. This avoids the 23-file merge conflict and makes every ported change reviewable.
- Do not touch `main` until the new branch passes `tests/run_offline.sh` and a live two-node test.

### 6.3 Stage plan

1. **Branch and base:** create `rewire/upstream-1.1.0-monolithic` from `recipe/main`. Confirm `git status` is clean and `TF_COMMIT=730d41abd8f4`.
2. **Port the monolithic experiment:** copy our `experiments/display-kv/` tree (apply.py, overlay/*, prepare.py, probe.py, Dockerfile, tests, README, NOTICE). Keep `REVISION=07aa02585f64` because the transforms are validated only for the 1.0.0 base.
3. **Port the local tests and docs:** `tests/live_tool_streaming.py`, `live_tool_streaming_test.py`, `thinking_settings_test.py`, `tool_call_probe.py`, `live_tool_hardening.py`, `docs/TOOL-OUTPUT.md`, our `docs/DISPLAY-KV.md` content.
4. **Rewire the launcher:** our `start-display-kv.sh` and `prepare.py` must build from an explicit `DISPLAY_KV_BASE` (default `tensorfold-dsv41:07aa02585f64`), not from `.$IMAGE` (which is 1.1.0). The ordinary `start.sh` keeps upstream's 1.1.0 `$IMAGE`.
5. **Rewire the settings:** add `MEM_ALLOCATOR_CAP` and `TOOL_OUTPUT_STREAMING` as display-KV-only settings; pass `TF_DS_MEM_CAP` and `TF_TOOL_OUTPUT_STREAMING` in the bundle. Keep upstream's `MEM_CAP` for the ordinary lane.
6. **Port the runs:** move `runs/2026/10/*` to `runs/2026-10-18/display-kv-bench/`; add `runs/2026-10-21/parser-guardrails/` and the `INDEX.md` entries.
7. **Docs:** add the display-KV and tool-output sections to `README.md`, `docs/OPERATIONS.md`, `docs/DISPLAY-KV.md`, `CREDITS.md`, `AGENTS.md`, `tests/README.md`.
8. **Offline validation:** `tests/run_offline.sh` must pass. Update `tests/test_defaults.py` only if the branch changes a protected default.
9. **Live validation:** `docker/build.sh`, `./stop.sh && ./start.sh`, `./verify.sh`, `tests/run_all.sh`; then `./stop.sh && ./start-display-kv.sh --build`, `./verify.sh`, `tests/live_tool_streaming.py`.
10. **Only then** merge the branch to `main` and push.

## 7. Rewiring checklist

| Item | Current | After rewire | Why |
|---|---|---|---|
| `.env` `TF_COMMIT` / `IMAGE` | `07aa02585f64` | `730d41abd8f4` for ordinary | Engine 1.1.0 fixes graph capture, host memory, RDMA MTU, vision, draft ranking |
| `start-display-kv.sh` base image | `$IMAGE` | explicit `DISPLAY_KV_BASE=07aa02585f64` | `apply.py` only accepts the 1.0.0 revision |
| `experiments/display-kv/apply.py` `REVISION` | `07aa02585f64` | keep `07aa02585f64` | Do not port transforms to 1.1.0 without revalidating every anchor and hash |
| `experiments/display-kv/prepare.py` | inherits `SPILL_GIB` | keep inheritance for our spill tier | Our lane supports spill; upstream's display-only lane does not |
| `start.sh` `MEM_ALLOCATOR_CAP` | passes `TF_DS_MEM_CAP` always | keep for the bundle; upstream `MEM_CAP` for the ordinary lane | The display image reads `TF_DS_MEM_CAP`; the ordinary 1.1.0 image does not |
| `.env.example` `THINKING` | `1` | decide: upstream `0` or keep `1` and update `test_defaults.py` | Upstream's default is off; production `.env` can keep `1` |
| `.env.example` `REASONING_EFFORT` | `high` | decide: upstream empty or keep `high` | Same |
| `.env.example` `TOOL_OUTPUT_STREAMING` | present | keep as display-KV-only setting | Upstream's `STRICT_TOOL_OUTPUT` is a different switch |
| `.gitignore` | `runs/*`, `/data/benchmarks.sqlite*` | remove both | Upstream's `test_repo_rules.py` requires `runs/` tracked |
| `runs/2026/10/*` | old layout | move to `runs/2026-10-18/display-kv-bench/` and add `INDEX.md` | Upstream requires `runs/YYYY-MM-DD/` |
| `tests/test_defaults.py` | absent | add; update `PROTECTED` if our defaults differ | Protects `.env.example` and `start.sh` forwarding |
| `tests/run_offline.sh` | absent | add; include our local tests that skip without GPU/server | CI parity |
| `scripts/preflight.sh` NCCL bootstrap | always on | opt-in `PREFLIGHT_NCCL_BOOTSTRAP=1` | Upstream made it opt-in because it starts a container on each node |
| `scripts/display_drm.sh` | no `rmmod` | add `rmmod nvidia_drm` before restore | Upstream fix for a card that does not reappear |
| `start-display-kv.sh` sudo | `sudo -v` | `sudo -n true 2>/dev/null || sudo -v` | Upstream fix for NOPASSWD head sudo |
| `docs/DISPLAY-KV.md` | full variant | rewrite for our variant; add a section for upstream's display-only variant | Avoid claiming spill/tool output for the display-only image |
| `docs/OPERATIONS.md` | display-KV section | merge upstream's OOM/engine-switches sections with our display-KV section | Both are useful |
| `README.md` | display-KV section | merge upstream's disk/engine updates with our display-KV section | Both are useful |
| `CREDITS.md` | display cache/spill credit | merge upstream's display-mapping credit with ours | Both are accurate |

## 8. Risks and caveats

1. **Engine 1.1.0 is not validated with our overlays.** The display-KV `apply.py` hashes are for 1.0.0. Porting the transforms to 1.1.0 requires revalidating every anchor and hash, then a live two-node test. Upstream's docs keep the experiments on 1.0.0.
2. **Upstream's display-KV is display-only.** It forces `SPILL_GIB=0` and does not include tool output, memory guard, cache metrics or bounded top-k. Adopting it would remove features our production lane uses.
3. **Upstream's tool-output branch is simpler than ours.** Adopting it would downgrade our guard. Keep ours.
4. **Upstream's memory guard is opt-in and does not enforce the 1.5 GiB floor.** Our production uses `MEM_ALLOCATOR_CAP=0`, so the guard is off anyway; but the display bundle must set `MEM_ALLOCATOR_CAP` explicitly.
5. **Upstream's `THINKING=0` default conflicts with our production `THINKING=1`.** `.env` is not tracked, so the server behavior stays the same; but `tests/test_defaults.py` will fail unless we update `PROTECTED` or adopt upstream's default.
6. **Upstream's `PARALLEL=4` default conflicts with our production `PARALLEL=8`.** Engine 1.1.0's head memory margin issue makes 8 risky. Consider 4 or 2.
7. **Upstream's `MEM_FLOOR_GIB=1.5` vs our production 1.0.** 1.0 is below earlyoom's 1 GiB limit; consider 1.5.
8. **Upstream's `SPILL_GIB=64` vs our production 32.** Keep 32 if disk is constrained.
9. **Upstream's `TEMPERATURE=1.0` vs our production 0.4.** Keep 0.4 if that is the desired sampling.
10. **Upstream's `BIND_HOST=127.0.0.1` vs our production 0.0.0.0.** Keep 0.0.0.0 with `API_KEY_FILE` if network access is intended.
11. **Upstream's `PORT=8888` vs our production 8100.** Keep 8100 if the router expects it.
12. **Upstream's `RESTART_POLICY=no` vs our production `unless-stopped`.** Keep `unless-stopped` if desired, but note the restart-order warning.
13. **The rewire branch is a large port, not a merge.** Budget a full day for the port and a second day for live validation.
14. **`origin/PR-branch` is not a shortcut.** It is 10 commits behind local `main`, lacks our latest streaming tool-validation work, keeps engine 1.0.0, and deletes our `runs/2026/10` layout.

## 9. Is the new engine Zig or Python?

**The recipe's engine is still Python, just newer.** The 1.1.0 release is the Python engine line (`deepseek-v41-tp2`) built from the TensorFold fork at `730d41abd8f4`.

- The recipe builds `Elevatormusic/TensorFold` on branch `deepseek-v41-tp2` at `730d41abd8f4` (`docs/WHAT-I-CHANGED.md`, `README.md`, `.env.example`).
- The release notes say "Commit `730d41abd8f4` on `deepseek-v41-tp2` (same git tree as `fc9200a44463`, the tested image)".
- The README's kit_bench table compares bertholomus's **Zig engine v0.6.0** and **Python engine v0.5.1** against this recipe. The Zig engine is a separate implementation and does not have the agent features (forced tool calls and structured output return an error there). This recipe does not build it.
- The 1.1.0 changes (graph capture, host memory, RDMA MTU, vision, draft ranking, tool-call drop, dense BF16, top-k tile) are Python-engine pull requests #8-#24 on the fork.

So: **Python, newer.** No Zig migration happened in 1.1.0.

## 10. Bottom line

- **Upstream is ahead in the engine, launcher safety, CI and docs.** The 1.1.0 engine is a clear win for the ordinary lane: graph-capture fix, host-memory fix, RDMA MTU fix, vision fix, draft ranking (+3-7% at 4 streams), and `DROP_UNOFFERED_TOOL_CALLS` default-on.
- **Upstream is behind in the display-KV lane.** Its merged display-KV is display-only and spill-disabled; its spill and tool-output branches are still in flight. Our monolithic lane is the only one that currently combines display KV, spill, tool output, memory guard, cache metrics and bounded top-k in one image.
- **The rewire branch is the right move.** Start from pristine `recipe/main`, port the monolithic experiment and local tests, keep `main` untouched. This avoids the 23-file merge and gives a clean rollback.
- **Do not adopt upstream's tool-output guard.** Ours is more advanced. Do adopt upstream's spill tier when `split/display-ssd-spill` merges; it is a genuine robustness upgrade (format 4, header CRC, base64 ids, writer readiness).
- **Do not merge `origin/PR-branch`.** It is an older reunification attempt, 10 commits behind local `main`, and lacks our latest streaming tool-validation work.
- **Production `.env` needs manual review after the rewire.** The tracked `.env.example` will follow upstream; `.env` is not tracked, so the server keeps running with the old pin until you edit it. Update `TF_COMMIT`/`IMAGE` to 1.1.0 for the ordinary lane, keep a separate 1.0.0 image for display-KV, and reconsider `PARALLEL=8` and `MEM_FLOOR_GIB=1.0` against the 1.1.0 head-memory margin.

## Appendix A: commands

```bash
# Keep main as the rollback
git branch backup/main-pre-rewire-20261021
git push origin backup/main-pre-rewire-20261021

# Create the rewire branch from pristine upstream
git fetch recipe
git checkout -b rewire/upstream-1.1.0-monolithic recipe/main

# Port our monolithic functionality (see sections 6.3 and 7)

# Offline validation
tests/run_offline.sh

# Live validation
docker/build.sh
./stop.sh && ./start.sh
./verify.sh
tests/run_all.sh

# Display-KV lane on the 1.0.0 base
./stop.sh
./start-display-kv.sh --build
./verify.sh
python3 -B tests/live_tool_streaming.py --base-url http://127.0.0.1:8100 --output /tmp/probes.json
```

## Appendix B: commit inventory

Upstream since the fork point, by area:

| Area | Commits |
|---|---:|
| `.github` | 25 |
| `docs` | 29 |
| `tests` | 28 |
| `README.md` | 15 |
| `experiments` | 13 |
| `runs` | 12 |
| `scripts` | 9 |
| `start.sh` | 5 |

Key upstream commits:

- `1b3066d` Merge PR #35 `engine-pin-dsv41-1.1.0`
- `d485dc9` feat: pin engine `dsv41-1.1.0` and turn `DROP_UNOFFERED_TOOL_CALLS` on
- `d925805` Merge PR #33 `split/memory-topk`
- `a4f852a` Merge PR #31 `split/cache-metrics`
- `c0d1c06` Merge PR #30 `split/display-kv`
- `fa15ff6` Merge PR #29 `split/nccl-bootstrap`
- `d002ccc` Merge PR #28 `split/launcher-config`
- `a15e671` Open-source repo files: contributing rules, templates, CI
- `9ecbc2c` fix: executable `build_extensions.sh`; preflight checks the RoCE MTU
- `148e91d` preflight: warn on a driver older than `580.178.04`
- `a106b6b` feat: `THINKING` and `REASONING_EFFORT` settings (default off)
- `ecb0bd5` feat: isolate opt-in display KV with spill disabled
- `50276a0` feat: add opt-in standalone cache metrics experiment
- `349408f` feat: add opt-in memory and bounded top-k experiment
- `ade79d5` fix: check NCCL socket bootstrap in preflight
- `083ba92` feat: expose validated prefill chunk configuration

In-flight upstream branches:

- `origin/split/display-ssd-spill` at `c9151e8` (26 commits ahead of `recipe/main`)
- `origin/split/tool-output` at `879d943` (31 commits ahead of `recipe/main`)

Upstream is active: re-fetch `recipe` and the split branches before a merge. The tips above are from 2026-10-21 08:10 UTC.
- `origin/PR-branch` at `251b0ee` (32 commits ahead of `recipe/main`, 10 behind local `main`)
