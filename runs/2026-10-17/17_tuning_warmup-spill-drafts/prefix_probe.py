"""Prefix probe: one long document, one question. Prints TTFT and cached tokens.
  prefix_probe.py SEED QUESTION_NO [TOKENS]
Same SEED = same document, so a later call (after a restart) shows whether the spill kept the prompt."""
import http.client
import json
import random
import sys
import time

seed, q = int(sys.argv[1]), int(sys.argv[2])
words = int(sys.argv[3]) if len(sys.argv) > 3 else 18000
rng = random.Random(seed)
vocab = ("river stone lamp keeper harbor signal north tide glass engine copper winter road field market "
         "letter bridge cloud orchard forge ledger compass lantern mill quarry valley").split()
doc = f"Document {seed}.\n" + " ".join(rng.choice(vocab) for _ in range(words))
questions = ["How often does the word 'lamp' appear in the first sentence?",
             "Name three words from the document that relate to water."]
body = {"model": "DeepSeek-v4.1-Flash-1017-TF", "stream": True, "max_tokens": 16,
        "stream_options": {"include_usage": True}, "chat_template_kwargs": {"thinking": False},
        "messages": [{"role": "user", "content": doc + "\n\n" + questions[q % 2]}]}
c = http.client.HTTPConnection("172.17.0.1", 8888, timeout=900, source_address=("127.0.0.1", 0))
t0 = time.perf_counter()
c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
r = c.getresponse()
ttft, usage = None, {}
for raw in r:
    line = raw.decode().strip()
    if not line.startswith("data: ") or line == "data: [DONE]":
        continue
    d = json.loads(line[6:])
    if ttft is None and d.get("choices") and any((d["choices"][0].get("delta") or {}).get(k) for k in ("content", "reasoning_content", "reasoning")):
        ttft = time.perf_counter() - t0
    if d.get("usage"):
        usage = d["usage"]
cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens")
print(json.dumps({"seed": seed, "q": q, "status": r.status, "prompt_tokens": usage.get("prompt_tokens"),
                  "cached_tokens": cached, "ttft_s": round(ttft, 2) if ttft else None}), flush=True)
