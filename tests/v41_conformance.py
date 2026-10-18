"""Render DeepSeek's official V4.1 encoding fixtures (encoding/tests in deepseek-ai/DeepSeek-V4.1-Flash-1017) through
TensorFold's V4.1 prompt path and compare each prompt with DeepSeek's golden output, byte for byte.

The prompt path is the one DsApp uses: ChatTemplate over the served chat_template.jinja, the reasoning_effort names
mapped to DeepSeek's numbers (EFFORT_VALUES), and image parts joined as DsVision.split joins them (the placeholder
and the text parts, joined by a blank line). No DeepSeek code runs: the fixtures are data.

Usage: python v41_conformance.py MODEL_DIR FIXTURE_DIR
"""

from __future__ import annotations

import difflib
import json
import sys
from pathlib import Path

from tensorfold.cuda.chat_template import ChatTemplate
from tensorfold.families.deepseek_v41.cuda.app import EFFORT_VALUES
from tensorfold.server.messages import _normalize_tool_call_arguments
from tensorfold.families.deepseek_v41.cuda.vision import PLACEHOLDER

API_ROLES = ("system", "developer", "user", "assistant", "tool")    # what the server accepts


def join_images(messages):
    out = []
    for m in messages:
        content = m.get("content")
        if not isinstance(content, list):
            out.append(m)
            continue
        texts = [PLACEHOLDER if p.get("type") in ("image_url", "image") else p["text"] for p in content]
        out.append({**m, "content": "\n\n".join(texts)})
    return out


def render(template, case):
    thinking = (case.get("thinking_mode") or "chat") == "thinking"
    extra = {}
    if thinking:
        name = case.get("reasoning_effort") or "high"
        extra["reasoning_effort"] = EFFORT_VALUES.get(name, name)
    if case.get("response_format"):                 # what DsApp._response_format hands the template
        extra["response_format"] = case["response_format"]
    messages = join_images(case["messages"])
    if any(m.get("role") not in API_ROLES for m in messages):     # DeepSeek app roles: the template alone
        tools = case.get("tools") or next((m["tools"] for m in messages if m.get("tools")), None)
        messages = _normalize_tool_call_arguments(messages)        # as ChatTemplate.render does
        return template.template.render(**template.specials, messages=messages, tools=tools,
                                        add_generation_prompt=True, enable_thinking=thinking, **extra), "raw template"
    return template.render(messages, tools=case.get("tools"), enable_thinking=thinking, extra=extra), "API path"


def v41_template(model_dir):
    """ChatTemplate as DsApp sets it up (the merge flag only: render() maps the efforts)."""

    template = ChatTemplate(model_dir)
    template.merge_leading = False
    return template


def main():
    model_dir, fixtures = Path(sys.argv[1]), Path(sys.argv[2])
    template = v41_template(model_dir)
    cases = sorted(fixtures.glob("test_input_*.json"))
    failed = 0
    for path in cases:
        case_id = path.stem.split("_")[-1]
        case = json.loads(path.read_text())
        if isinstance(case, list):              # a bare message list, as DeepSeek's load_cases reads it
            case = {"messages": case}
        want = (fixtures / f"test_output_{case_id}.txt").read_text()
        try:
            got, path_used = render(template, case)
        except Exception as exc:  # noqa: BLE001 - report every case
            failed += 1
            print(f"[FAIL] case {case_id}: {type(exc).__name__}: {exc}")
            continue
        if got == want:
            print(f"[PASS] case {case_id} via {path_used} ({len(want)} chars, thinking_mode={case.get('thinking_mode')}, "
                  f"tools={len(case.get('tools') or [])}, effort={case.get('reasoning_effort')})")
            continue
        failed += 1
        print(f"[FAIL] case {case_id} via {path_used}: prompt differs")
        for line in difflib.unified_diff(want.splitlines(True), got.splitlines(True), "deepseek", "tensorfold", n=1):
            sys.stdout.write("    " + repr(line) + "\n")
    print(f"{len(cases) - failed}/{len(cases)} cases match")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
