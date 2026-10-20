# Runs of 19 October 2026 (UTC): tool-output guard probes (pull request #6)

Files of the display-backed KV author (0rand). `tests/live_tool_hardening.py` sent the requests to a running display-KV server. The files do not record the engine image or the settings. They are records of an earlier state of the guard: in these files an unoffered tool call gives HTTP 500; the current code gives HTTP 422 (a client retries a 5xx response).

| Run (UTC) | Checks passed | Remarks | File |
|---|---|---|---|
| 00:20Z | 4 of 4 | unoffered call (JSON and stream), offered call after a rejection, default thinking | `tool-hardening-live-20261019T002031Z.json` |
| 00:21Z | 4 of 6 | the same, plus "mixed calls SSE atomicity" and "literal unavailable-tool example", which failed because the model did not produce the fixtures, so both are inconclusive (see `docs/TOOL-OUTPUT.md`) | `tool-hardening-live-20261019T002139Z.json` |
| 00:23Z | 4 of 6 | the same two checks failed again | `tool-hardening-live-20261019T002312Z.json` |

## Tester report (pull request #6)

Reported by Sora39831 on two Spark nodes. The maintainers did not measure these numbers. The file keeps the numbers with the caveats of the author.

| Run | What it measured | File |
|---|---|---|
| 19 Oct (UTC) | Launcher test and a sequential A/B of the stock lane against the display lane: prefill and decode at 32K to about 1M tokens, prefix cache, host memory. Open caveats: 512 against 2048 prefill chunk, a local patch in the stock image, one run per size | `display-kv-tester-ab.md` |
