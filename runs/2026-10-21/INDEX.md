# Runs of 21 October 2026 (UTC)

## Parser isolation and conservative prose streaming

[Parser guardrails record](parser-guardrails/README.md) documents the upstream
review, integrated strict-validator foundation, conservative prose projector,
installed-source CPU checks and old-image model-backed baseline.

The code is deployed as `tensorfold-dsv41:display-kv-58e319cb2866` on both nodes
and live-validated: `./verify.sh` 5/5, parser probe 12 PASS / 2 INCONCLUSIVE /
0 FAIL. The two inconclusive cases are fixture-induction limits, not parser
failures. The `TOOL_OUTPUT_STREAMING=0` rollback was also validated live with a
negative control.
