"""Concurrency and prefix-cache benchmark for the DeepSeek-V4.1 TensorFold server (--parallel 4).

1. Concurrency: 1, 2, 3 and 4 streams at once, prose and code, thinking off, 512 new tokens each: per-stream decode
   tok/s, aggregate tok/s, time to first token (streamed, so TTFT and decode are measured apart).
2. Prefix cache: a ~24K-token document, then a second question on the same document, then a fresh document of the
   same size: TTFT and usage's cached_tokens for each.

Usage: python3 -I v41_bench.py   (stdlib only; writes bench-results.json to the current directory)
Environment: V41_BASE_URL (default http://172.17.0.1:8888/v1) and V41_MODEL (default DeepSeek-v4.1-Flash-1017-EXL3).
The requests leave from source address 127.0.0.1.
"""

from __future__ import annotations

import concurrent.futures as cf
import http.client
import json
import os
import random
import statistics
import time
from urllib.parse import urlsplit

_URL = urlsplit(os.environ.get("V41_BASE_URL", "http://172.17.0.1:8888/v1"))
HOST, PORT = _URL.hostname, _URL.port or 80
PATH = _URL.path.rstrip("/") + "/chat/completions"
MODEL = os.environ.get("V41_MODEL", "DeepSeek-v4.1-Flash-1017-EXL3")
PROSE = ["Write a detailed essay about the history of lighthouses.", "Describe how a city changes over a century.",
         "Explain the life cycle of a star to a curious teenager.", "Write a long story about a sailor and a storm."]
CODE = ["Write a Python module that implements an LRU cache with tests.",
        "Write a JavaScript debounce and throttle utility with docs and examples.",
        "Write a Rust function that parses a CSV line with quoted fields, with tests.",
        "Write a C function that reverses a linked list, with a small test harness."]


def stream(messages, max_tokens=512, timeout=900.0) -> dict:
    body = {"model": MODEL, "messages": messages, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "chat_template_kwargs": {"enable_thinking": False},
            "ignore_eos": True}
    conn = http.client.HTTPConnection(HOST, PORT, timeout=timeout, source_address=("127.0.0.1", 0))
    t0 = time.perf_counter()
    ttft, usage = None, {}
    try:
        conn.request("POST", PATH, json.dumps(body), {"Content-Type": "application/json"})
        resp = conn.getresponse()
        if resp.status != 200:
            raise RuntimeError(f"HTTP {resp.status}: {resp.read()[:300]!r}")
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                d = ch.get("delta") or {}
                if ttft is None and (d.get("content") or d.get("reasoning_content")):
                    ttft = time.perf_counter() - t0
    finally:
        conn.close()
    total = time.perf_counter() - t0
    n = int(usage.get("completion_tokens") or 0)
    decode = total - (ttft or 0.0)
    return {"ttft": ttft or total, "total": total, "tokens": n, "tok_s": (n - 1) / decode if decode > 0 else 0.0,
            "prompt": int(usage.get("prompt_tokens") or 0),
            "cached": int(((usage.get("prompt_tokens_details") or {}).get("cached_tokens")) or 0)}


def concurrency(kind: str, prompts: list[str], n: int) -> dict:
    msgs = [[{"role": "user", "content": prompts[i % len(prompts)]}] for i in range(n)]
    t0 = time.perf_counter()
    with cf.ThreadPoolExecutor(max_workers=n) as pool:
        runs = list(pool.map(stream, msgs))
    wall = time.perf_counter() - t0
    out = {"kind": kind, "streams": n, "per_stream_tok_s": round(statistics.mean(r["tok_s"] for r in runs), 1),
           "min_stream_tok_s": round(min(r["tok_s"] for r in runs), 1),
           "aggregate_tok_s": round(sum(r["tokens"] for r in runs) / wall, 1),
           "ttft_s": round(statistics.mean(r["ttft"] for r in runs), 2), "tokens": sum(r["tokens"] for r in runs)}
    print(f"[bench] {kind:5s} x{n}: per-stream {out['per_stream_tok_s']} tok/s (min {out['min_stream_tok_s']}), "
          f"aggregate {out['aggregate_tok_s']} tok/s, TTFT {out['ttft_s']} s", flush=True)
    return out


def document(seed: int, words: int = 18000) -> str:
    rng = random.Random(seed)
    vocab = ("river stone harbor lantern copper meadow signal winter orchard engine cobalt valley thread "
             "archive pilot ember canyon ledger quartz beacon tundra mosaic falcon prism delta").split()
    lines, cur = [], []
    for i in range(words):
        cur.append(rng.choice(vocab))
        if len(cur) == 14:
            lines.append(f"{i // 14}. " + " ".join(cur) + ".")
            cur = []
    return "\n".join(lines)


def prefix_cache() -> list[dict]:
    doc_a, doc_b = document(1), document(2)
    steps = [("cold: document A, question 1", doc_a, "How many numbered lines are there? Answer briefly."),
             ("warm: document A, question 2", doc_a, "Which word appears in line 7? Answer briefly."),
             ("cold: document B (same size)", doc_b, "How many numbered lines are there? Answer briefly.")]
    out = []
    for name, doc, q in steps:
        r = stream([{"role": "system", "content": "You read documents carefully."},
                    {"role": "user", "content": doc + "\n\n" + q}], max_tokens=32)
        row = {"step": name, "prompt_tokens": r["prompt"], "cached_tokens": r["cached"], "ttft_s": round(r["ttft"], 2)}
        print(f"[prefix] {name}: prompt {r['prompt']} tok, cached {r['cached']}, TTFT {row['ttft_s']} s", flush=True)
        out.append(row)
    return out


def main() -> None:
    stream([{"role": "user", "content": "Say hi."}], max_tokens=8)                  # warm the path
    results = {"concurrency": [], "prefix": None}
    for kind, prompts in (("prose", PROSE), ("code", CODE)):
        for n in (1, 2, 3, 4):
            results["concurrency"].append(concurrency(kind, prompts, n))
    results["prefix"] = prefix_cache()
    with open("bench-results.json", "w") as f:
        json.dump(results, f, indent=1)
    print("[bench] done -> bench-results.json", flush=True)


if __name__ == "__main__":
    main()
