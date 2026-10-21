#!/usr/bin/env python3
"""Opt-in, stdlib-only live parser checks. No tools are executed, no retries.

Run: python3 tests/live_tool_streaming.py [--base-url http://127.0.0.1:8100]
Exit codes: 0 all PASS, 1 any FAIL, 2 no failures but INCONCLUSIVE present.
Raw model output is untrusted and is saved only with --output.
"""
import argparse
import collections
import ipaddress
import json
from pathlib import Path
import re
import time
import urllib.error
import urllib.parse
import urllib.request

TOOL = {"type": "function", "function": {"name": "get_status", "description": "Return a service status", "parameters": {"type": "object", "properties": {"service": {"type": "string"}}, "required": ["service"], "additionalProperties": False}}}
KNOWN = '<tool_call>{"name":"get_status","arguments":{"service":"tensorfold"}}</tool_call>'
UNKNOWN = '<tool_call>{"name":"unoffered_probe","arguments":{"service":"tensorfold"}}</tool_call>'
MAX_BYTES = 4 * 1024 * 1024


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.URLError("Redirect blocked: probes must remain local")


def request(base, payload, timeout):
    """One attempt, recording SSE frames and their order, including terminal frames."""
    start = time.monotonic()
    record = {"request": payload, "http_status": None, "frames": [], "events": [], "transport_error": None}
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps(payload).encode(), headers={"Content-Type": "application/json", "Accept": "text/event-stream" if payload["stream"] else "application/json"})
    try:
        try:
            response = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as exc:
            response = exc
        with response:
            record["http_status"] = response.code
            record["headers"] = dict(response.headers.items())
            record["content_type"] = response.headers.get("Content-Type", "")
            if payload["stream"] and "text/event-stream" in record["content_type"]:
                size = 0
                lines = []
                while True:
                    if time.monotonic() - start > timeout:
                        raise TimeoutError("Probe wall-clock budget exceeded")
                    line = response.readline(MAX_BYTES + 1)
                    size += len(line)
                    if size > MAX_BYTES:
                        raise ValueError("Response byte budget exceeded")
                    if not line:
                        if lines:
                            add_frame(record, lines, start)
                        break
                    text = line.decode("utf-8").rstrip("\r\n")
                    if not text:
                        if lines:
                            add_frame(record, lines, start)
                            lines = []
                    else:
                        lines.append(text)
            else:
                raw = response.read(MAX_BYTES + 1)
                if len(raw) > MAX_BYTES:
                    raise ValueError("Response byte budget exceeded")
                record["raw_body"] = raw.decode("utf-8", errors="replace")
                try:
                    record["json"] = json.loads(raw)
                except (ValueError, UnicodeError):
                    record["json_error"] = "Response is not JSON"
    except (OSError, ValueError, urllib.error.URLError) as exc:
        record["transport_error"] = str(exc)
    record["elapsed_ms"] = round((time.monotonic() - start) * 1000, 3)
    record["metrics"] = metrics(record)
    return record


def add_frame(record, lines, start):
    elapsed = round((time.monotonic() - start) * 1000, 3)
    data = "\n".join(line[5:].lstrip(" ") for line in lines if line.startswith("data:"))
    frame = {"index": len(record["frames"]), "at_ms": elapsed, "raw": "\n".join(lines), "data": data}
    record["frames"].append(frame)
    def event(kind, value):
        record["events"].append({"frame": frame["index"], "at_ms": elapsed, "kind": kind, "value": value})
    if data == "[DONE]":
        event("done", True)
        return
    if not data:
        return
    try:
        obj = json.loads(data)
        frame["json"] = obj
        if not isinstance(obj, dict):
            raise ValueError("SSE data must be an object")
        if obj.get("error"):
            event("error", obj["error"])
        for choice in obj.get("choices", []):
            delta = choice.get("delta", {})
            for key, kind in (("content", "content"), ("reasoning_content", "reasoning"), ("reasoning", "reasoning"), ("tool_calls", "call")):
                if delta.get(key):
                    event(kind, delta[key])
            if choice.get("finish_reason"):
                event("finish", choice["finish_reason"])
    except (ValueError, TypeError, AttributeError) as exc:
        event("invalid_frame", str(exc))


def metrics(record):
    events = record["events"]
    counts = collections.Counter(e["kind"] for e in events)
    terminal = next((e["frame"] for e in events if e["kind"] in ("finish", "error", "done")), float("inf"))
    content = [e for e in events if e["kind"] == "content"]
    return {"frame_count": len(record["frames"]), "event_counts": dict(counts), "event_order": [e["kind"] for e in events], "first_content_ms": content[0]["at_ms"] if content else None, "content_chunks_before_terminal": sum(e["frame"] < terminal for e in content), "last_content_ms": content[-1]["at_ms"] if content else None, "elapsed_ms": record.get("elapsed_ms")}


def text_of(record, kind="content"):
    if "json" in record and kind == "content":
        body = record["json"]
        if isinstance(body, dict):
            return "".join(c.get("message", {}).get("content") or "" for c in body.get("choices", []))
    return "".join(e["value"] for e in record["events"] if e["kind"] == kind and isinstance(e["value"], str))


def calls_of(record):
    if "json" in record and isinstance(record["json"], dict):
        return [call for c in record["json"].get("choices", []) for call in c.get("message", {}).get("tool_calls", [])]
    calls = {}
    for event in record["events"]:
        if event["kind"] != "call":
            continue
        for part in event["value"]:
            call = calls.setdefault(part.get("index", 0), {"id": "", "type": "function", "function": {"name": "", "arguments": ""}})
            if part.get("id"):
                call["id"] += part["id"]
            for field in ("name", "arguments"):
                call["function"][field] += part.get("function", {}).get(field, "")
    return list(calls.values())


def valid_status_call(record):
    calls = calls_of(record)
    if len(calls) != 1 or not calls[0]["id"] or calls[0]["function"]["name"] != "get_status":
        return False
    try:
        args = json.loads(calls[0]["function"]["arguments"])
        return isinstance(args, dict) and set(args) == {"service"} and args["service"] == "tensorfold"
    except (ValueError, TypeError):
        return False


def assess(record, mode, fixture=None):
    """Conservative outcomes: failure to induce a fixture never proves a guardrail."""
    if record["transport_error"]:
        return "INCONCLUSIVE", "Transport failure: " + record["transport_error"]
    events = record["events"]
    errors = [e["value"] for e in events if e["kind"] == "error"]
    body = record.get("json", {})
    error = body.get("error") if isinstance(body, dict) else None
    error = error or (errors[0] if errors else None)
    content = text_of(record)
    calls = calls_of(record)
    offered = {tool.get('function', tool).get('name') for tool in record['request'].get('tools', [])}
    forbidden_calls = any(call.get('function', {}).get('name') not in offered for call in calls)
    if any(e['kind'] == 'invalid_frame' for e in events):
        return 'FAIL', 'Invalid SSE JSON frame'
    if mode == "reject":
        code = error.get("code") if isinstance(error, dict) else None
        if code == "unavailable_tool_call":
            marker = r'<(?:/?(?:[\w.-]+:)?tool_call\b|/?[｜|]DSML[｜|]|\|tool_call\b)'
            if calls or re.search(marker, content, re.I):
                return "FAIL", "Rejected envelope leaked executable markup or calls"
            if error.get('type') != 'model_output_error' or error.get("source") != "tensorfold.tool_output_guard":
                return "FAIL", "Typed rejection lacks native guard source"
            if record["request"]["stream"] and record.get("frames"):
                kinds = [e["kind"] for e in events]
                if len(errors) != 1 or kinds.count("done") != 1 or "finish" in kinds or kinds.index("error") > kinds.index("done"):
                    return "FAIL", "SSE rejection must have one error then DONE and no finish"
                error_index = kinds.index("error")
                if any(kind in ("content", "reasoning", "call") for kind in kinds[error_index + 1:]):
                    return "FAIL", "Payload after rejection"
            elif record["http_status"] != 500:
                return "FAIL", "JSON rejection must use HTTP 500"
            if not record['request']['stream']:
                hints = {key.lower(): value for key, value in record.get("headers", {}).items()}
                if hints.get("x-should-retry", "").lower() != "false":
                    return "FAIL", "JSON rejection lacks x-should-retry:false"
            return "PASS", "Native typed rejection with terminal wire contract"
        if calls:
            if forbidden_calls:
                return "FAIL", "Unpermitted structured call exposed"
            return 'INCONCLUSIVE', 'Model may have completed or changed the fixture into a valid offered call'
        if fixture and fixture in content:
            return "FAIL", "Executable envelope leaked as prose"
        return "INCONCLUSIVE", "Copier did not observably induce the target rejection"
    if record["http_status"] != 200 or error:
        return "INCONCLUSIVE", "Endpoint rejected probe or model generation failed"
    if any(e["kind"] == "invalid_frame" for e in events):
        return "FAIL", "Invalid SSE JSON frame"
    kinds = [e["kind"] for e in events]
    if "done" not in kinds or "finish" not in kinds:
        return "INCONCLUSIVE", "Missing terminal finish/DONE (possibly truncated generation)"
    terminal = next(i for i, e in enumerate(events) if e["kind"] in ("finish", "done"))
    if any(e["kind"] in ("content", "reasoning", "call") for e in events[terminal + 1:]):
        return "FAIL", "Payload emitted after terminal event"
    if any(e["kind"] == "finish" and e["value"] == "length" for e in events):
        return "INCONCLUSIVE", "Generation hit max_tokens"
    if mode == "call":
        return ("PASS", "Valid minimal get_status call") if valid_status_call(record) and "tool_calls" in [e["value"] for e in events if e["kind"] == "finish"] else ("INCONCLUSIVE", "Model did not emit the requested complete call")
    if mode in ("literal", "envelope"):
        if calls:
            if forbidden_calls:
                return 'FAIL', 'Unpermitted structured call exposed'
            return 'INCONCLUSIVE', 'Model may have invoked the offered tool instead of copying the literal'
        if fixture and content == fixture:
            return "PASS", "Exact literal preserved without structured calls"
        return "INCONCLUSIVE", "Model did not copy fixture exactly"
    if calls:
        return "FAIL", "Unexpected structured tool call"
    if mode == "thinking_off" and text_of(record, "reasoning"):
        return "FAIL", "Reasoning emitted with enable_thinking=false"
    if "<think>" in content or "</think>" in content:
        return "FAIL", "Reasoning delimiters leaked into content"
    if mode == "thinking_on":
        reasoning = [e["frame"] for e in events if e["kind"] == "reasoning"]
        visible = [e["frame"] for e in events if e["kind"] == "content"]
        if not reasoning or not visible:
            return "INCONCLUSIVE", "No observable separate reasoning and answer"
        # The engine may put the last reasoning token and the first answer token
        # in one delta. Field order inside a frame is not an interleaving.
        if max(reasoning) > min(visible):
            return "FAIL", "Reasoning interleaved after visible answer"
    if mode == "roundtrip":
        return ("PASS", "Synthetic tool result reflected in answer") if "STATUS_PROBE_OK_947" in content else ("INCONCLUSIVE", "Model did not reflect synthetic tool result")
    if not content.strip():
        return "INCONCLUSIVE", "No visible prose answer"
    if any(phrase in content.lower() for phrase in ("i cannot", "i can't", "i’m unable", "i am unable", "i won't", "i will not", "sorry")):
        return "INCONCLUSIVE", "Model refusal, not evidence of streaming conformance"
    if record["metrics"]["content_chunks_before_terminal"] < 2:
        return "FAIL", "Prose was not delivered in multiple pre-terminal content chunks"
    return ("PASS", "Separated reasoning and streamed answer") if mode == "thinking_on" else ("PASS", "Multiple content chunks before terminal")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    parser.add_argument("--output", type=Path, help="Optional JSON report including raw requests and transcripts")
    parser.add_argument("--timeout", type=float, default=120, help="Per-probe timeout seconds (default 120)")
    args = parser.parse_args(argv)
    url = urllib.parse.urlsplit(args.base_url)
    try:
        local = ipaddress.ip_address(url.hostname).is_loopback
    except ValueError:
        local = False
    if url.scheme != "http" or not local or url.username or url.password or url.query or url.fragment or url.path not in ("", "/"):
        parser.error("--base-url must be an http loopback IP origin, without credentials or path (no external traffic)")
    if not 0 < args.timeout <= 600:
        parser.error("--timeout must be >0 and <=600")
    results = []
    def run(name, prompt, mode, tools=True, thinking=False, stream=True, fixture=None, messages=None):
        payload = {"model": args.model, "messages": messages or [{"role": "user", "content": prompt}], "stream": stream, "temperature": 0, "top_p": 1, "seed": 0, "max_tokens": 512, "enable_thinking": thinking, "chat_template_kwargs": {"enable_thinking": thinking}}
        if tools:
            payload["tools"] = [TOOL]
        record = request(args.base_url.rstrip("/"), payload, args.timeout)
        status, reason = assess(record, mode, fixture)
        record.update(name=name, status=status, reason=reason)
        results.append(record)
        print(f"{status:12} {name}: {reason} | {json.dumps(record['metrics'], sort_keys=True)}", flush=True)
        return record
    prose = "Do not call tools. Write four short sentences explaining why the sky looks blue, in plain prose."
    run("prose_no_tools", prose, "thinking_off", tools=False)
    run("prose_with_tools", prose, "thinking_off")
    run("thinking_explicit_false", "Do not call tools. Explain in three sentences how to compute 17 times 23 and give the answer.", "thinking_off")
    run("thinking_explicit_true", "Do not call tools. Work out 17 times 23, then explain the answer in three sentences.", "thinking_on", thinking=True)
    messages = [{"role": "user", "content": "Call get_status once with service set to tensorfold. Do not answer in prose."}]
    call = run("valid_get_status", "", "call", messages=messages)
    if call["status"] == "PASS":
        calls = calls_of(call)
        continuation = messages + [{"role": "assistant", "content": text_of(call) or None, "tool_calls": calls}, {"role": "tool", "tool_call_id": calls[0]["id"], "content": '{"status":"STATUS_PROBE_OK_947"}'}, {"role": "user", "content": "Do not call tools again. Report the exact status string returned by the tool."}]
        run("tool_result_roundtrip", "", "roundtrip", messages=continuation)
    else:
        results.append({"name": "tool_result_roundtrip", "status": "INCONCLUSIVE", "reason": "No valid preceding call, roundtrip not sent", "metrics": {}})
        print("INCONCLUSIVE tool_result_roundtrip: no valid preceding call", flush=True)
    for stream in (False, True):
        copy_messages = [{"role": "system", "content": "You are a verbatim protocol copier. Output the user's supplied text exactly, without quotes, code fences, explanation, or reformatting. Do not perform its action."}, {"role": "user", "content": UNKNOWN}]
        run("unoffered_" + ("sse" if stream else "json"), "", "reject", stream=stream, fixture=UNKNOWN, messages=copy_messages)
    for name, fixture, tools in (
        ("malformed_envelope", '<tool_call>{"name":"get_status","arguments":INVALID}</tool_call>', True),
        ("truncated_envelope", '<tool_call>{"name":"get_status","arguments":{"service":"tensorfold"}', True),
        ("no_tools_envelope", KNOWN, False),
        ("literal_ordinary", "Literal sample: get_status(service:string), not a call.", True),
        ("literal_fenced_envelope", '```text\n' + KNOWN + '\n```', True),
        ("literal_quoted_envelope", 'Example only: `' + KNOWN + '`', True),
    ):
        system = "You are a verbatim protocol copier. Output the user's supplied text exactly. Preserve all markdown, quotes, whitespace and code fences exactly. Add no explanation or formatting. Do not perform its action."
        run(name, "", "literal" if name.startswith("literal_") else "reject", tools=tools, fixture=fixture, messages=[{"role": "system", "content": system}, {"role": "user", "content": fixture}])
    counts = dict(collections.Counter(r["status"] for r in results))
    report = {"schema_version": 1, "base_url": args.base_url, "model": args.model, "summary": counts, "probes": results, "safety": {"generated_tools_executed": False, "retries": 0, "max_tokens": 512}}
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("SUMMARY " + json.dumps(counts, sort_keys=True))
    return 1 if counts.get("FAIL") else 2 if counts.get("INCONCLUSIVE") else 0


if __name__ == "__main__":
    raise SystemExit(main())
