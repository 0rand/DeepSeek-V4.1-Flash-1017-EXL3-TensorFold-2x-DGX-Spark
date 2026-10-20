#!/usr/bin/env python3
"""Live API probes. Generated tool calls are inspected, never executed or retried."""
import argparse
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

TOOL = {"type": "function", "function": {"name": "get_status", "description": "Get a service's status.",
        "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"]}}}
UNKNOWN = '<tool_call>{"name":"task","arguments":{"description":"Read-only protocol probe"}}</tool_call>'
KNOWN = '<tool_call>{"name":"get_status","arguments":{"service":"tensorfold"}}</tool_call>'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    results = []

    def request(body):
        req = urllib.request.Request(args.base_url.rstrip("/") + "/v1/chat/completions",
            json.dumps({"model": args.model, "max_tokens": 512, "temperature": 0, **body}).encode(),
            {"Content-Type": "application/json"})
        try:
            response = urllib.request.urlopen(req, timeout=180)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            return response.code, response.read().decode()

    def record(name, good, status, seconds, detail):
        outcome = "INCONCLUSIVE" if good is None else ("PASS" if good else "FAIL")
        results.append({"test": name, "passed": good, "outcome": outcome, "http_status": status,
                        "seconds": round(seconds, 3), "detail": detail})
        print(f"[{outcome}] {name}: HTTP {status}, {seconds:.1f}s", flush=True)

    for name, stream, output in (("unavailable tool JSON", False, UNKNOWN),
                                 ("unavailable tool SSE", True, UNKNOWN),
                                 ("mixed calls SSE atomicity", True, KNOWN + UNKNOWN)):
        start = time.monotonic()
        status, text = request({"messages": [
            {"role": "system", "content": "You are a verbatim protocol copier. Output the user's supplied text exactly, without quotes, code fences, explanation, or reformatting. Do not perform its action."},
            {"role": "user", "content": output}], "tools": [TOOL], "stream": stream,
            "chat_template_kwargs": {"enable_thinking": False}})
        if stream:
            frames = [json.loads(line[5:]) for line in text.splitlines()
                      if line.startswith("data:") and line != "data: [DONE]"]
            errors = [frame["error"] for frame in frames if "error" in frame]
            deltas = [choice.get("delta", {}) for frame in frames for choice in frame.get("choices", [])]
            good = (status == 200 and len(errors) == 1 and text.count("data: [DONE]") == 1
                    and errors[0].get("code") == "unavailable_tool_call"
                    and not any(delta.get("tool_calls") or delta.get("content") for delta in deltas))
            if name == "mixed calls SSE atomicity" and status == 200 and not errors:
                visible = "".join(delta.get("content", "") for delta in deltas)
                names = [call.get("function", {}).get("name") for delta in deltas
                         for call in delta.get("tool_calls", [])]
                if "<tool_call>" not in visible and "DSML" not in visible and all(value in (None, "", "get_status") for value in names):
                    good = None  # The model did not generate the mixed valid/invalid fixture.
            detail = errors or {"unexpected_frames": frames}
        else:
            payload = json.loads(text)
            good = status == 422 and payload.get("error", {}).get("code") == "unavailable_tool_call"
            detail = payload
        record(name, good, status, time.monotonic() - start, detail)

    start = time.monotonic()
    status, text = request({"messages": [
        {"role": "system", "content": "Copy the user's text exactly, including its Markdown code fences. Do not execute it or add commentary."},
        {"role": "user", "content": "```json\n" + UNKNOWN + "\n```"}],
        "tools": [TOOL], "chat_template_kwargs": {"enable_thinking": False}})
    payload = json.loads(text)
    message = payload.get("choices", [{}])[0].get("message", {})
    good = status == 200 and UNKNOWN in (message.get("content") or "") and not message.get("tool_calls")
    if status == 200 and UNKNOWN not in (message.get("content") or "") and not message.get("tool_calls"):
        good = None  # A refusal to copy the example does not exercise literal parsing.
    record("literal unavailable-tool example", good, status, time.monotonic() - start, payload)

    start = time.monotonic()
    status, text = request({"messages": [{"role": "user", "content": "Get the status of the tensorfold service using get_status."}],
        "tools": [TOOL], "tool_choice": "required", "chat_template_kwargs": {"enable_thinking": False}})
    payload = json.loads(text)
    message = payload.get("choices", [{}])[0].get("message", {})
    calls = message.get("tool_calls", [])
    good = status == 200 and bool(calls) and all(c["function"]["name"] == "get_status" for c in calls)
    record("offered tool after rejection", good, status, time.monotonic() - start, payload)

    start = time.monotonic()
    status, text = request({"messages": [{"role": "user", "content": "What is 17 * 23? Answer briefly."}]})
    payload = json.loads(text)
    message = payload.get("choices", [{}])[0].get("message", {})
    good = status == 200 and bool(message.get("reasoning_content")) and "391" in (message.get("content") or "")
    record("default thinking and recovery", good, status, time.monotonic() - start, payload)

    if args.output:
        if not args.output.parent.is_dir():
            raise ValueError("output parent must already exist")
        args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"{sum(row['passed'] is True for row in results)} passed, "
          f"{sum(row['passed'] is False for row in results)} failed, "
          f"{sum(row['passed'] is None for row in results)} inconclusive", flush=True)
    raise SystemExit(0 if all(row["passed"] is not False for row in results) else 1)


if __name__ == "__main__":
    main()
