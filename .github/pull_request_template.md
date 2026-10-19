<!--
Thank you for your pull request. Fill in each part. Write "n/a" if a part does not apply.
Read CONTRIBUTING.md first: https://github.com/Elevatormusic/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark/blob/main/CONTRIBUTING.md
-->

## Summary

<!-- What does this pull request change, and why? One focused change only. Link the issue, if there is one. -->

## Default behaviour

<!-- Does a user who runs ./start.sh with .env.example get a different server after this change? -->

- [ ] This pull request changes no default.
- [ ] This pull request changes a default. The title says so, and the table below gives each change.

| Setting | Old value | New value | Reason | Evidence (run under `runs/YYYY-MM-DD/`) |
|---|---|---|---|---|
| | | | | |

## Test evidence

### Offline tests

<!-- Paste the last lines of `tests/run_offline.sh`. CI runs it too. -->

```text

```

### Live checks

<!-- Only if the change touches the serving path. Write "n/a" if not. -->

- `./verify.sh`: <!-- for example "5/5 passed" -->
- `tests/run_all.sh` summary: <!-- the path of the summary file under runs/YYYY-MM-DD/ -->
- Hardware: <!-- for example "2x DGX Spark, DGX OS 24.04.5, driver 580.178.04" -->
- Engine image id: <!-- the IMAGE value, for example tensorfold-dsv41:07aa02585f64 -->

## Performance claims

<!--
Each number needs a fair A/B: the same settings on both sides, except the change that you test.
Write "no performance claim" if this pull request has none.
-->

| Setting | A (before) | B (after) |
|---|---|---|
| Engine image id | | |
| `TF_DS_PREFILL_CHUNK` | | |
| `SPILL_GIB` | | |
| `CONTEXT` / `PARALLEL` / `MTP_DRAFTS` | | |
| Host tuning (`scripts/host_tuning.sh`) | | |
| Prompt lengths, runs per side, prefix cache cold or warm | | |
| Result (median and range; prefill tok/s, decode tok/s or TTFT) | | |
| Saved run (`runs/YYYY-MM-DD/...`) | | |

## Checklist

- [ ] One focused change. No unrelated files, no mass reformat.
- [ ] `tests/run_offline.sh`, `ruff check .` and `shellcheck` pass.
- [ ] Each performance number links to a saved run under `runs/YYYY-MM-DD/`, and the day `INDEX.md` lists the run.
- [ ] `runs/` stays tracked: no change to `.gitignore` that ignores it, and no other layout (for example `runs/YYYY/MM/`).
- [ ] A new setting is in `.env.example`, with a safe default and a comment.
- [ ] The docs (`README.md`, `docs/`, `tests/README.md`) match the change.
- [ ] A new feature is off by default, or it has its own start script.
- [ ] No secrets, keys, tokens or machine-specific values (host names, user names, home paths, public IP addresses).
- [ ] New docs use ASD-STE100 Simplified Technical English and the standard serving terms.
