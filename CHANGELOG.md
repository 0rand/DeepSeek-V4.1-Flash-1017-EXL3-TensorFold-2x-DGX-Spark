# Changelog

This file lists the notable changes of each release. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/). The versions follow [Semantic Versioning](https://semver.org/):

- **Major**: a change that needs action from users, for example a new engine image, new model files or a changed default.
- **Minor**: a new opt-in feature or setting.
- **Patch**: a fix or a docs change.

Each release is a git tag. To use a release, run `git checkout vX.Y.Z` before `./start.sh`.

## [Unreleased]

### Added

- Optional, experimental display-backed KV variant (pull request #6, by 0rand): `./start-display-kv.sh` starts a derived image that maps the NVIDIA display reservation into a 2.1M-position KV pool. `./start.sh` does not change. The launcher has its own settings (`MEM_ALLOCATOR_CAP`, `STRICT_TOOL_OUTPUT`). `docs/DISPLAY-KV.md` has the measured cost: the testers saw a lower prefill tok/s.
- A tool-output guard in the display image: a model answer that calls a tool the request did not offer gets an explicit error (HTTP 422, or an error frame in a stream). It is on in the display launcher. `STRICT_TOOL_OUTPUT=0` in `.env` turns it off. See `docs/TOOL-OUTPUT.md`.
- A live memory guard, a patched SSD spill with a shutdown drain on both ranks, and cache telemetry in `/health` and `/metrics` (display image only).
- `THINKING` and `REASONING_EFFORT` in `.env`: turn thinking on for requests that do not say, and set their effort. The default stays off. `docs/CONNECT.md` says to set thinking before you compare benchmark scores.

### Changed

- `scripts/preflight.sh` shows a `[warn]` line when a node has an NVIDIA driver older than the tested 580.178.04 (#24). `docs/OPERATIONS.md` gives the node-hang log signature from #24 and the recovery steps.
- `docs/OPERATIONS.md`: what to do when earlyoom stops the engine under concurrent load (#21): `PARALLEL=2`, and do not raise `MEM_FLOOR_GIB`.
- `docs/OPERATIONS.md`: the out-of-memory steps now use the measured traces from #21 (`runs/2026-10-20/`). With 8 images in each request, `PARALLEL=2` passed the load that failed with `PARALLEL=4` (not a clean A/B); with 16 it also failed. The steps now also say to send fewer images at the same time, or to set `VISION=0` when you do not need images.

### Fixed

- `./start.sh` stopped with "Permission denied" on a new clone: `scripts/build_extensions.sh` had no executable bit (#15).
- Preflight now checks that RoCE has MTU 4096 on both nodes. At the default MTU 1500, the RDMA gather failed with "transport retry counter exceeded" (#15). `docs/NETWORK.md` tells you to set MTU 9000.

### Repo

- `runs/` keeps the `runs/YYYY-MM-DD/` layout with an `INDEX.md` for each day. The tester reports of pull request #6 are saved there. A 5,963-line tool-call transcript of the pull request is not kept.
- The experiment tests of `experiments/display-kv/` skip when torch or the pinned TensorFold source is absent, so the offline CI job passes without them.
- CI: a defaults guard test, a home-path and public-IP scan, workflow checks (actionlint, zizmor) and a secret scan (gitleaks).
- CI: the secret scan refuses a `.gitleaksignore` in the checkout, because gitleaks loads it in any case.
- CI: the secret scan runs from the base branch (`secret-scan.yml`, `pull_request_target`), so a pull request cannot change it.
- AGENTS.md: rules for AI agents that open issues and pull requests, with links to CONTRIBUTING.md, the templates and SECURITY.md.
- Issues: the `issue-labels` workflow adds a type label from the title prefix (`[bug]`, `[feature]`, `[perf]`), also for an issue made without a form. An issue with no prefix gets `needs-triage`.

## [1.0.0] - 2026-10-19

The first release: the setup that the README measured on 17 and 18 October 2026.

### Serving

- DeepSeek-V4.1-Flash-1017 (EXL3, 2.9 bpw) on two DGX Spark with TensorFold, tensor-parallel over the QSFP link.
- About 2x the decode speed of the standard vLLM setup (57 vs 27 tok/s on prose, 92 vs 40 tok/s on code), 1,048,576-token context and 4 concurrent requests.
- Prefill of about 2,000 to 2,200 tok/s up to 64K tokens: fork PRs 1-4, host memory settings and warm-up cache (#3), O_DIRECT prompt spill and transparent huge pages off (#5).
- Prefix cache in memory and on disk (`SPILL_GIB`), so a restart keeps cached prompts.
- Forced tool calls and structured output for AI agents.

### Docs

- README in plain language, connect guide and notes for agents (#1, #2).
- Standard serving terms: prefill, decode, TTFT, tok/s (#4).

### Repo

- Contributing rules, issue forms, pull request template, CI (shellcheck, ruff, offline tests, Markdown links), security policy and code of conduct (#7).
- AI disclosure and label rules for contributors (#8).
- A pull request check for the type and the AI disclosure, and this changelog (#9).

[Unreleased]: https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/releases/tag/v1.0.0
