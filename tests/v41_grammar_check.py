"""Structured output under --parallel: grammar and plain requests in flight together on the DeepSeek-V4.1 TensorFold
server. Each grammar reply must match its format; the plain replies must still come back; nothing may hang.

Usage: python3 -I v41_grammar_check.py [rounds]   (stdlib only; requests leave from source address 127.0.0.1)
Environment: V41_BASE_URL (default http://172.17.0.1:8888/v1) and V41_MODEL (default DeepSeek-v4.1-Flash-1017-EXL3).
"""

from __future__ import annotations

import concurrent.futures as cf
import http.client
import json
import os
import re
import sys
import time
from urllib.parse import urlsplit

_URL = urlsplit(os.environ.get("V41_BASE_URL", "http://172.17.0.1:8888/v1"))
HOST, PORT = _URL.hostname, _URL.port or 80
PATH = _URL.path.rstrip("/") + "/chat/completions"
MODEL = os.environ.get("V41_MODEL", "DeepSeek-v4.1-Flash-1017-EXL3")
PERSON = {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},
                                           "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}},
          "required": ["name", "age", "tags"], "additionalProperties": False}


def post(body: dict, timeout: float = 300.0) -> dict:
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout, source_address=("127.0.0.1", 0))
    try:
        conn.request("POST", PATH, json.dumps({"model": MODEL, **body}),
                     {"Content-Type": "application/json"})
        resp = conn.getresponse()
        data = json.loads(resp.read() or b"{}")
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {data}")
        return data
    finally:
        conn.close()


def chat(text: str, think: bool, **extra) -> dict:
    return {"messages": [{"role": "user", "content": text}], "max_tokens": 600,
            "chat_template_kwargs": {"enable_thinking": think}, **extra}


def is_person(content: str) -> bool:
    try:
        v = json.loads(content)
    except json.JSONDecodeError:
        return False
    return (isinstance(v, dict) and set(v) == {"name", "age", "tags"} and isinstance(v["name"], str)
            and isinstance(v["age"], int) and isinstance(v["tags"], list) and len(v["tags"]) <= 3
            and all(isinstance(t, str) for t in v["tags"]))


CASES = [
    ("json_schema off", chat("Invent a fictional person.", False, response_format={
        "type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": PERSON}}), is_person),
    ("json_schema think", chat("Invent a fictional person.", True, response_format={
        "type": "json_schema", "json_schema": {"name": "person", "strict": True, "schema": PERSON}}), is_person),
    ("json_object", chat("Give me a JSON object describing a cat.", False, response_format={"type": "json_object"}),
     lambda c: isinstance(json.loads(c), dict)),
    ("regex date", chat("What is a date in 2024? Answer with the date only.", False,
                        guided_regex=r"2024-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9])"),
     lambda c: re.fullmatch(r"2024-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9])", c) is not None),
    ("choice", chat("Is the sky blue on a clear day? Answer yes or no.", False, guided_choice=["yes", "no"]),
     lambda c: c in ("yes", "no")),
    ("plain prose", chat("Write two sentences about the sea.", False), lambda c: len(c) > 20),
    ("plain code", chat("Write a Python function that reverses a string.", False), lambda c: "def " in c),
]


def run(case) -> tuple[str, bool, str, float]:
    name, body, check = case
    t = time.perf_counter()
    try:
        content = (post(body)["choices"][0]["message"].get("content") or "").strip()
        ok = bool(check(content))
    except Exception as exc:  # noqa: BLE001
        content, ok = f"{type(exc).__name__}: {exc}", False
    return name, ok, content, time.perf_counter() - t


def main() -> int:
    rounds = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    failed = 0
    for r in range(rounds):
        with cf.ThreadPoolExecutor(max_workers=4) as pool:     # --parallel 4: the lanes hold mixed requests
            for name, ok, content, s in pool.map(run, CASES):
                failed += not ok
                print(f"[{'PASS' if ok else 'FAIL'}] round {r + 1} {name} {s:.1f}s: {content[:110]!r}", flush=True)
    total = rounds * len(CASES)
    print(f"{total - failed}/{total} passed", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
