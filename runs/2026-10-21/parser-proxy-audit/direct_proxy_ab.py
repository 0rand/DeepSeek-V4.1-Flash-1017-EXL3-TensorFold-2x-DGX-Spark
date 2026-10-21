#!/usr/bin/env python3
"""Local, sequential direct/proxy contract A/B. Never execute generated tools.

This is a functional parity probe, not a latency or throughput benchmark.
The production proxy can perform its configured single strict-guard retry.
"""
import argparse
import collections
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import urllib.request


def load_probe(repo):
    spec = importlib.util.spec_from_file_location("live_tool_streaming", repo / "tests/live_tool_streaming.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def health(base):
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(base + "/health", timeout=10) as response:
        body = json.load(response)
    if "primary" in body:
        return {"ok": body.get("ok"), "primary": body["primary"], "unavailableToolRetry": body.get("unavailableToolRetry")}
    return {key: body.get(key) for key in ("ok", "busy", "requests_running", "requests_total", "streams", "kv_cache_backing")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    probe = load_probe(args.repo)
    sides = [("A-direct", "http://127.0.0.1:8100"), ("B-production-proxy", "http://127.0.0.1:8000")]
    started = datetime.datetime.now(datetime.timezone.utc).isoformat()
    before = {name: health(base) for name, base in sides}
    fixture = '    ```text\n' + probe.KNOWN + '\n```'
    copier = "You are a verbatim protocol copier. Output the user's supplied text exactly. Preserve all markdown, quotes, whitespace and code fences exactly. Add no explanation or formatting. Do not perform its action."
    bare_copier = "You are a verbatim protocol copier. Output the user's supplied text exactly, without quotes, code fences, explanation, or reformatting. Do not perform its action."
    cases = [
        {"name": "prose_no_tools", "mode": "thinking_off", "tools": False, "messages": [{"role": "user", "content": "Write four short sentences explaining why the sky looks blue. Use plain prose, not a list."}]},
        {"name": "prose_with_tools", "mode": "thinking_off", "tools": True, "messages": [{"role": "user", "content": "Do not call tools. Write four short sentences explaining why the sky looks blue, in plain prose."}]},
        {"name": "valid_native_call", "mode": "call", "tools": True, "messages": [{"role": "user", "content": "Call get_status once with service set to tensorfold. Do not answer in prose."}]},
        {"name": "four_space_fenced_literal", "mode": "literal", "tools": True, "fixture": fixture, "messages": [{"role": "system", "content": copier}, {"role": "user", "content": fixture}]},
        {"name": "unoffered_json", "mode": "reject", "tools": True, "stream": False, "fixture": probe.UNKNOWN, "messages": [{"role": "system", "content": bare_copier}, {"role": "user", "content": probe.UNKNOWN}]},
        {"name": "unoffered_sse", "mode": "reject", "tools": True, "fixture": probe.UNKNOWN, "messages": [{"role": "system", "content": bare_copier}, {"role": "user", "content": probe.UNKNOWN}]},
        {"name": "no_tools_executable_envelope", "mode": "reject", "tools": False, "fixture": probe.KNOWN, "messages": [{"role": "system", "content": bare_copier}, {"role": "user", "content": probe.KNOWN}]},
    ]
    results = []
    for case in cases:
        payload = {"model": "DeepSeek-V4.1-Flash-1017", "messages": case["messages"], "stream": case.get("stream", True), "temperature": 0.4, "top_p": 1.0, "seed": 0, "max_tokens": 512, "enable_thinking": False, "chat_template_kwargs": {"enable_thinking": False}}
        if case["tools"]:
            payload["tools"] = [probe.TOOL]
        digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        for side, base in sides:
            record = probe.request(base, payload, 90)
            status, reason = probe.assess(record, case["mode"], case.get("fixture"))
            calls = probe.calls_of(record)
            record.update(name=case["name"], side=side, payload_sha256=digest, status=status, reason=reason)
            filename = case["name"] + "." + side + ".json"
            (args.output / filename).write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n")
            summary = {key: record[key] for key in ("name", "side", "payload_sha256", "status", "reason", "http_status", "metrics")}
            summary["call_names"] = [call.get("function", {}).get("name") for call in calls]
            summary["headers"] = {key: value for key, value in record.get("headers", {}).items() if key.lower() in ("x-should-retry", "x-prerouter-tool-retries", "x-prerouter-tool-rejected")}
            results.append(summary)
            print(json.dumps(summary, ensure_ascii=False), flush=True)
    after = {name: health(base) for name, base in sides}
    counts = dict(collections.Counter(result["status"] for result in results))
    report = {"schema_version": 1, "started_utc": started, "finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(), "kind": "functional_direct_proxy_ab_not_performance", "safety": {"generated_tools_executed": False, "service_restarts": 0, "guard_toggle": False, "max_tokens_per_request": 512, "coordinator_retries": 0, "production_proxy_retry_may_occur": True}, "health_before": before, "health_after": after, "summary": counts, "results": results}
    (args.output / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    print("SUMMARY " + json.dumps(counts, sort_keys=True), flush=True)
    return 1 if counts.get("FAIL") else 0


if __name__ == "__main__":
    raise SystemExit(main())
