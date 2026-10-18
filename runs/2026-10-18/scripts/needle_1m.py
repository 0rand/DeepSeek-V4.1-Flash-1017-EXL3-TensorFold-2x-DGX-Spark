"""Needle test near the 1M-token context limit (1,048,576).
The filler is not one repeated sentence: each line has a random record number, item, place and year,
so the prompt cannot come from the prefix cache. One needle sits at a set depth.
Usage: python3 needle_1m.py TARGET_TOKENS DEPTH OUT.json"""
import http.client
import json
import random
import sys
import time

target = int(sys.argv[1]) if len(sys.argv) > 1 else 1_035_000
depth = float(sys.argv[2]) if len(sys.argv) > 2 else 0.63
out = sys.argv[3] if len(sys.argv) > 3 else "needle-1m.json"
model = "DeepSeek-v4.1-Flash-1017-TF"
code = "ORCHID-%04d" % random.Random(time.time()).randrange(1000, 9999)

rng = random.Random(int(time.time()))
items = ["weather logs", "ship manifests", "tax rolls", "grain receipts", "court minutes", "harbor charts",
         "census sheets", "mill ledgers", "toll records", "letters of credit"]
places = ["Ashford", "Bellmere", "Corran", "Dunwick", "Elsby", "Farrow", "Galesend", "Holloway", "Ivel", "Jarrow"]


def line(i):
    return (f"Record {rng.randrange(100000, 999999)}: the clerk filed {rng.choice(items)} from "
            f"{rng.choice(places)} for the year {rng.randrange(1700, 1900)}. ")


def conn():
    return http.client.HTTPConnection("172.17.0.1", 8888, timeout=7200, source_address=("127.0.0.1", 0))


def count_tokens(text):
    c = conn()
    c.request("POST", "/tokenize", json.dumps({"model": model, "prompt": text}), {"Content-Type": "application/json"})
    r = c.getresponse()
    d = json.loads(r.read())
    return d.get("count") or len(d.get("tokens", []))


sample = "".join(line(i) for i in range(2000))
per_line = count_tokens(sample) / 2000
n = int(target / per_line)
lines = [line(i) for i in range(n)]
lines.insert(int(n * depth), f"The secret launch code is {code}. ")
prompt = "".join(lines) + "\nWhat is the secret launch code? Reply with the code only."

body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": 32, "temperature": 0,
        "stream": True, "stream_options": {"include_usage": True}, "reasoning_effort": "none"}
c = conn()
t0 = time.perf_counter()
c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
r = c.getresponse()
ttft, text, usage = None, "", {}
if r.status != 200:
    err = r.read().decode(errors="replace")[:500]
    res = {"status": r.status, "error": err, "target": target, "lines": n}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps(res))
    sys.exit(1)
for raw in r:
    s = raw.decode(errors="replace").strip()
    if not s.startswith("data:") or s == "data: [DONE]":
        continue
    d = json.loads(s[5:])
    if d.get("usage"):
        usage = d["usage"]
    for ch in d.get("choices", []):
        delta = ch.get("delta", {})
        piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
        if piece and ttft is None:
            ttft = time.perf_counter() - t0
        text += delta.get("content") or ""
total = time.perf_counter() - t0
pt = usage.get("prompt_tokens")
res = {"code": code, "found": code in text, "answer": text.strip()[:80], "depth": depth, "prompt_tokens": pt,
       "context_limit": 1048576, "ttft_s": round(ttft or 0, 1), "total_s": round(total, 1),
       "prefill_tps": round(pt / ttft) if pt and ttft else None, "usage": usage,
       "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
json.dump(res, open(out, "w"), indent=1)
print(json.dumps(res))
