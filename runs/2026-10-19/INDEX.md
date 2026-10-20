# Runs of 19 October 2026 (UTC): display-KV tester report (pull request #6)

Reported by Sora39831 on two Spark nodes. The maintainers did not measure these numbers. The file keeps the numbers with the caveats of the author.

| Run | What it measured | File |
|---|---|---|
| 19 Oct (UTC) | Launcher test and a sequential A/B of the stock lane against the display lane: prefill and decode at 32K to about 1M tokens, prefix cache, host memory. Open caveats: 512 against 2048 prefill chunk, a local patch in the stock image, one run per size | `display-kv-tester-ab.md` |
