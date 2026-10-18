"""API-shaped V4.1 conversations rendered by DeepSeek's reference encoder (encoding.py from the 1017 repo) and by
TensorFold's V4.1 prompt path (v41_conformance.render); each prompt must match byte for byte.

Usage: python v41_differential.py MODEL_DIR ENCODING_DIR CONFORMANCE_DIR
"""

from __future__ import annotations

import copy
import difflib
import sys
from pathlib import Path

model_dir, encoding_dir, conformance_dir = (Path(a) for a in sys.argv[1:4])
sys.path[:0] = [str(encoding_dir), str(conformance_dir)]
import encoding as ref  # noqa: E402
from v41_conformance import render, v41_template  # noqa: E402

WEATHER = {"type": "function", "function": {"name": "get_weather", "description": "Weather for a city",
           "parameters": {"type": "object", "properties": {"city": {"type": "string"}, "days": {"type": "integer"}},
                          "required": ["city"]}}}
TIME = {"type": "function", "function": {"name": "get_time", "description": "Local time",
        "parameters": {"type": "object", "properties": {"tz": {"type": "string"}}}}}
CALL = {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris", "days": 2}'}}
CALL2 = {"id": "call_2", "type": "function", "function": {"name": "get_time", "arguments": '{"tz": "CET"}'}}

CASES = {
    "chat, one turn": dict(messages=[{"role": "user", "content": "Hi"}]),
    "chat, system + two turns": dict(messages=[{"role": "system", "content": "Be brief."},
                                               {"role": "user", "content": "A"}, {"role": "assistant", "content": "B"},
                                               {"role": "user", "content": "C"}]),
    "thinking low": dict(thinking_mode="thinking", reasoning_effort="low",
                         messages=[{"role": "user", "content": "2+2?"}]),
    "thinking max + system": dict(thinking_mode="thinking", reasoning_effort="max",
                                  messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "Q"}]),
    "thinking, earlier reasoning dropped": dict(thinking_mode="thinking", messages=[
        {"role": "user", "content": "A"}, {"role": "assistant", "reasoning_content": "r1", "content": "B"},
        {"role": "user", "content": "C"}]),
    "tools, chat": dict(tools=[WEATHER, TIME], messages=[{"role": "system", "content": "S"},
                                                         {"role": "user", "content": "Paris weather?"}]),
    "tools, call + result, thinking": dict(thinking_mode="thinking", tools=[WEATHER], messages=[
        {"role": "system", "content": "S"}, {"role": "user", "content": "Paris?"},
        {"role": "assistant", "reasoning_content": "need tool", "content": "", "tool_calls": [CALL]},
        {"role": "tool", "tool_call_id": "call_1", "content": '{"temp": 18}'}]),
    "tools, two parallel calls + results": dict(tools=[WEATHER, TIME], messages=[
        {"role": "user", "content": "Weather and time?"},
        {"role": "assistant", "content": "", "tool_calls": [CALL, CALL2]},
        {"role": "tool", "tool_call_id": "call_1", "content": "18C"},
        {"role": "tool", "tool_call_id": "call_2", "content": "13:00"}]),
    "mid-conversation system": dict(messages=[{"role": "system", "content": "S1"}, {"role": "user", "content": "A"},
                                              {"role": "assistant", "content": "B"},
                                              {"role": "system", "content": "S2"}, {"role": "user", "content": "C"}]),
    "json_object, no system": dict(response_format={"type": "json_object"},
                                   messages=[{"role": "user", "content": "JSON please"}]),
    "json_object + system + tools, thinking": dict(thinking_mode="thinking", response_format={"type": "json_object"},
                                                   tools=[WEATHER], messages=[{"role": "system", "content": "S"},
                                                                              {"role": "user", "content": "JSON"}]),
    "two leading systems + tools": dict(tools=[WEATHER], messages=[
        {"role": "system", "content": "S1"}, {"role": "system", "content": "S2"}, {"role": "user", "content": "Q"}]),
}


def reference(case):
    messages = copy.deepcopy(case["messages"])
    if (case.get("tools") or case.get("response_format")) and messages[0]["role"] != "system":
        messages.insert(0, {"role": "system", "content": ""})       # deepseek-recipe inserts an empty system turn
    if case.get("tools"):
        messages[0]["tools"] = case["tools"]
    if case.get("response_format"):
        messages[0]["response_format"] = case["response_format"]
    prompt, _ = ref.encode_messages(messages, thinking_mode=case.get("thinking_mode") or "chat",
                                    reasoning_effort=case.get("reasoning_effort"), return_multi_modal_data=True)
    return prompt


template = v41_template(model_dir)
failed = 0
for name, case in CASES.items():
    try:
        want = reference(case)
    except Exception as exc:  # noqa: BLE001
        print(f"[SKIP] {name}: the reference refuses it ({type(exc).__name__}: {exc})")
        continue
    try:
        got, _ = render(template, copy.deepcopy(case))
    except Exception as exc:  # noqa: BLE001
        failed += 1
        print(f"[FAIL] {name}: TensorFold refuses it ({type(exc).__name__}: {exc})")
        continue
    if got == want:
        print(f"[PASS] {name}")
        continue
    failed += 1
    print(f"[FAIL] {name}")
    for line in difflib.unified_diff(want.splitlines(True), got.splitlines(True), "deepseek", "tensorfold", n=0):
        sys.stdout.write("    " + repr(line)[:240] + "\n")
print(f"{len(CASES) - failed}/{len(CASES)} match (a SKIP counts as a match)")
sys.exit(1 if failed else 0)
