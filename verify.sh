#!/usr/bin/env bash
# A 1-minute check after start.sh: a plain reply, a thinking reply, a strict json_schema reply, a forced tool call and
# a regex. Every check must PASS. For the full 32-test suite see tests/README.md.
set -uo pipefail
# shellcheck source=scripts/common.sh
. "$(dirname "$0")/scripts/common.sh"

python3 -I - "$API" "$SERVED_NAME" "$(auth)" <<'EOF'
import json, sys, time, urllib.request

api, model, auth = sys.argv[1], sys.argv[2], sys.argv[3]
headers = {"Content-Type": "application/json", **({"Authorization": auth.split(": ", 1)[1]} if auth else {})}


def chat(body, timeout=300):
    req = urllib.request.Request(api + "/chat/completions", json.dumps({"model": model, **body}).encode(), headers)
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r), time.perf_counter() - t


def msg(text, think=False, **extra):
    return {"messages": [{"role": "user", "content": text}], "max_tokens": 400,
            "chat_template_kwargs": {"enable_thinking": think}, **extra}


TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
        "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}
SCHEMA = {"type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": {
    "type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    "required": ["name", "age"], "additionalProperties": False}}}


def plain(d):
    return len(d["choices"][0]["message"]["content"] or "") > 10


def thinking(d):
    m = d["choices"][0]["message"]
    return bool(m.get("reasoning_content") or m.get("reasoning")) and "</think>" not in (m["content"] or "")


def schema(d):
    v = json.loads(d["choices"][0]["message"]["content"])
    return set(v) == {"name", "age"} and isinstance(v["age"], int)


def tool(d):
    calls = d["choices"][0]["message"].get("tool_calls") or []
    return d["choices"][0]["finish_reason"] == "tool_calls" and calls[0]["function"]["name"] == "get_weather"


def regex(d):
    import re
    return re.fullmatch(r"\d{4}-\d{2}-\d{2}", d["choices"][0]["message"]["content"].strip()) is not None


checks = [
    ("plain reply", msg("Write two sentences about the sea."), plain),
    ("thinking reply", msg("What is 17 * 23?", think=True), thinking),
    ("strict json_schema", msg("Invent a person.", response_format=SCHEMA), schema),
    ("tool_choice required", msg("Weather in Paris?", tools=[TOOL], tool_choice="required"), tool),
    ("regex", msg("Give a date in 2024.", guided_regex=r"\d{4}-\d{2}-\d{2}"), regex),
]
failed = 0
for name, body, ok in checks:
    try:
        d, s = chat(body)
        good = ok(d)
        usage = d.get("usage") or {}
        note = f"{usage.get('completion_tokens', '?')} tokens in {s:.1f}s"
    except Exception as exc:  # noqa: BLE001
        good, note = False, f"{type(exc).__name__}: {exc}"
    failed += not good
    print(f"[{'PASS' if good else 'FAIL'}] {name}: {note}", flush=True)
print(f"{len(checks) - failed}/{len(checks)} passed")
sys.exit(1 if failed else 0)
EOF
