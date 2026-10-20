#!/usr/bin/env python3
"""Capture a small agent-style tool response without executing any generated call."""
import argparse
import json
import urllib.request


TASK = {"type": "function", "function": {
    "name": "task", "description": "Delegate read-only source research to a subagent.",
    "parameters": {"type": "object", "properties": {
        "description": {"type": "string"}, "prompt": {"type": "string"},
        "subagent_type": {"type": "string", "enum": ["explore", "general"]},
        "task_id": {"type": "string"}, "command": {"type": "string"}},
        "required": ["description", "prompt", "subagent_type"], "additionalProperties": False}}}


def post(base, route, body):
    request = urllib.request.Request(base.rstrip("/") + route, json.dumps(body).encode(),
                                     {"Content-Type": "application/json"})
    return urllib.request.urlopen(request, timeout=180)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8100")
    parser.add_argument("--model", default="DeepSeek-V4.1-Flash-1017")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--tool-choice", choices=("auto", "required"), default="auto")
    args = parser.parse_args()
    body = {"model": args.model, "messages": [
        {"role": "system", "content": "You are a software engineering assistant. Use the provided tools to research source code. Do not invent tool results."},
        {"role": "user", "content": "Investigate the implementation needed for SSD spillover of a display-backed KV cache, including metrics. Delegate source research to the explore subagent using task. Research only, no edits or runtime actions. Start by making that tool call."}],
        "tools": [TASK], "tool_choice": args.tool_choice, "max_tokens": 768,
        "temperature": 0, "stream": args.stream, "return_token_ids": True,
        "chat_template_kwargs": {"enable_thinking": False}}
    with post(args.base_url, "/v1/chat/completions", body) as response:
        if args.stream:
            for line in response:
                text = line.decode().strip()
                if text.startswith("data:"):
                    print(text, flush=True)
        else:
            result = json.load(response)
            print(json.dumps(result, ensure_ascii=False, indent=2))
            ids = result.get("tensorfold", {}).get("token_ids")
            if ids:
                with post(args.base_url, "/detokenize", {"tokens": ids}) as decoded:
                    print("RAW GENERATION:", json.load(decoded)["prompt"])


if __name__ == "__main__":
    main()
