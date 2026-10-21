#!/usr/bin/env python3
"""Repeat the no-tools SSE terminator probe and record both raw HTTP bodies."""
import argparse
import collections
import datetime
import importlib.util
import json
from pathlib import Path
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    spec = importlib.util.spec_from_file_location("live_probe", repo / "tests/live_tool_streaming.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)
    payload = {"model": "DeepSeek-V4.1-Flash-1017", "messages": [{"role": "user", "content": "Write four short sentences explaining why the sky looks blue. Use plain prose, not a list."}], "stream": True, "temperature": 0.4, "top_p": 1.0, "seed": 0, "max_tokens": 512, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}
    results = []
    for repeat in range(1, 4):
        for side, port in (("direct", 8100), ("proxy", 8000)):
            record = probe.request("http://127.0.0.1:" + str(port), payload, 60)
            kinds = [event["kind"] for event in record["events"]]
            record.update(repeat=repeat, side=side, done_count=kinds.count("done"), finish_count=kinds.count("finish"))
            record["verdict"] = "PASS" if record["http_status"] == 200 and record["done_count"] == 1 and record["finish_count"] == 1 else "FAIL"
            (args.output / f"repeat-{repeat}-{side}.json").write_text(json.dumps(record, indent=2) + "\n")
            summary = {key: record[key] for key in ("repeat", "side", "verdict", "http_status", "done_count", "finish_count", "metrics")}
            results.append(summary)
            print(json.dumps(summary), flush=True)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    for side, port in (("direct", 8100), ("proxy", 8000)):
        request = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        with opener.open(request, timeout=60) as response:
            raw = response.read(4 * 1024 * 1024)
        (args.output / f"raw-{side}.sse").write_bytes(raw)
        raw_result = {"side": side, "raw_done_count": raw.count(b"data: [DONE]"), "ends_in_blank_line": raw.endswith(b"\n\n"), "bytes": len(raw)}
        print(json.dumps(raw_result), flush=True)
        results.append(raw_result)
    report = {"at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "kind": "sse_terminator_regression_reproduction", "results": results, "summary": dict(collections.Counter(result["verdict"] for result in results if "verdict" in result))}
    (args.output / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    print("SUMMARY " + json.dumps(report["summary"]), flush=True)


if __name__ == "__main__":
    main()
