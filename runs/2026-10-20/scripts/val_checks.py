"""Live checks for fork PRs #13 (overlapping image requests), #14 (--vision-max-images) and #15
(TF_DROP_UNOFFERED_TOOL_CALLS), plus the server's thinking default (recipe PR #20).

Run on the head: python3 -I val_checks.py TAG [checks...]   checks: images limit tools thinking (default: all but thinking)
Writes val-TAG.json next to this file and prints one line for each check.
"""
import base64
import concurrent.futures as cf
import http.client
import json
import struct
import sys
import time
import zlib
from pathlib import Path

HOST, PORT = "172.17.0.1", 8888


def post(body, timeout=900):
    c = http.client.HTTPConnection(HOST, PORT, timeout=timeout, source_address=("127.0.0.1", 0))
    c.request("POST", "/v1/chat/completions", json.dumps({"model": "x", **body}), {"Content-Type": "application/json"})
    r = c.getresponse()
    raw = r.read()
    try:
        return r.status, json.loads(raw or b"{}")
    except ValueError:
        return r.status, {"raw": raw[:500].decode("utf-8", "replace")}


def png(w, h, seed):
    """A w x h RGB PNG with a simple pattern (no PIL on the host)."""
    rows = bytearray()
    for y in range(h):
        rows.append(0)
        for x in range(w):
            rows += bytes(((x * 7 + seed * 31) % 256, (y * 5 + seed * 17) % 256, ((x + y) * 3 + seed) % 256))
    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    data = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 6)) + chunk(b"IEND", b""))
    return "data:image/png;base64," + base64.b64encode(data).decode()


def image_msg(n, text, size=(256, 192), seed=0):
    parts = [{"type": "image_url", "image_url": {"url": png(size[0], size[1], seed + i)}} for i in range(n)]
    return [{"role": "user", "content": parts + [{"type": "text", "text": text}]}]


def check_images():
    """#13: identical image requests at the same time (the crash needs equal prompts; the later fill ends first)."""
    msgs = image_msg(1, "Describe the colours in this image in one sentence.", size=(448, 448))
    body = {"messages": msgs, "max_tokens": 48, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}}
    results = []
    for wave in range(3):
        with cf.ThreadPoolExecutor(6) as ex:
            futs = []
            for i in range(6):
                futs.append(ex.submit(post, body))
                time.sleep(0.15 * (wave % 2))
            results += [f.result() for f in futs]
    statuses = [s for s, _ in results]
    errors = [d.get("error", d.get("raw")) for s, d in results if s != 200]
    alive = post({"messages": [{"role": "user", "content": "Say ok."}], "max_tokens": 8,
                  "chat_template_kwargs": {"enable_thinking": False}})[0]
    ok = all(s == 200 for s in statuses) and alive == 200
    return ok, {"statuses": statuses, "errors": [str(e)[:300] for e in errors[:3]], "after": alive}


def check_limit():
    """#14: with --vision-max-images 32, 9 and 32 images pass, 33 get the limit message."""
    out = {}
    for n in (9, 32, 33):
        s, d = post({"messages": image_msg(n, "How many images are there? Answer with a number.", size=(64, 64)),
                     "max_tokens": 16, "temperature": 0, "chat_template_kwargs": {"enable_thinking": False}})
        msg = json.dumps(d.get("error", ""))[:200] if s != 200 else (d["choices"][0]["message"].get("content") or "")[:80]
        out[n] = [s, msg]
    ok = out[9][0] == 200 and out[32][0] == 200 and out[33][0] == 400 and "at most 32" in out[33][1]
    return ok, out


TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "Read a file.",
          "parameters": {"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]}}}]


def tool_history():
    call = {"id": "call_1", "type": "function",
            "function": {"name": "apply_patch", "arguments": json.dumps({"patch": "*** Update File: app.py\n-x = 1\n+x = 2"})}}
    return [
        {"role": "system", "content": "You are a coding agent. Earlier in this session you had a tool named apply_patch."},
        {"role": "user", "content": "Change x to 2 in app.py."},
        {"role": "assistant", "content": "", "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call_1", "content": "Done. app.py updated."},
        {"role": "user", "content": "Now change x to 3 in app.py the same way, with apply_patch. Do not explain, just do it."},
    ]


def check_tools():
    """#15: a call to an unoffered tool (apply_patch) with tools offered and with no tools. Reports what the server
    returned: with the flag on, no tool markup may reach the content."""
    out = {}
    for label, extra in (("tools", {"tools": TOOLS}), ("no_tools", {})):
        s, d = post({"messages": tool_history(), "max_tokens": 300, "temperature": 0,
                     "chat_template_kwargs": {"enable_thinking": False}, **extra})
        m = d["choices"][0]["message"] if s == 200 else {}
        content = m.get("content") or ""
        out[label] = {"status": s, "content": content[:400], "tool_calls": [c["function"]["name"] for c in m.get("tool_calls") or []],
                      "markup_in_content": any(t in content for t in ("<tool_call>", "DSML", "apply_patch\"")),
                      "finish": d["choices"][0].get("finish_reason") if s == 200 else None}
    ok = all(v["status"] == 200 and not v["markup_in_content"] for v in out.values())
    return ok, out


def check_thinking():
    """--thinking on the server: a request with no thinking field gets reasoning_content."""
    s, d = post({"messages": [{"role": "user", "content": "What is 17 * 23?"}], "max_tokens": 400})
    m = d["choices"][0]["message"] if s == 200 else {}
    ok = s == 200 and bool(m.get("reasoning_content"))
    return ok, {"status": s, "reasoning_chars": len(m.get("reasoning_content") or ""), "content": (m.get("content") or "")[:80]}


CHECKS = {"images": check_images, "limit": check_limit, "tools": check_tools, "thinking": check_thinking}

if __name__ == "__main__":
    tag = sys.argv[1]
    names = sys.argv[2:] or ["images", "limit", "tools"]
    report = {}
    for name in names:
        t0 = time.time()
        try:
            ok, detail = CHECKS[name]()
        except Exception as exc:  # noqa: BLE001
            ok, detail = False, {"exception": repr(exc)[:300]}
        report[name] = {"ok": ok, "seconds": round(time.time() - t0, 1), "detail": detail}
        print(f"[{'PASS' if ok else 'FAIL'}] {name} ({report[name]['seconds']} s): {json.dumps(detail)[:600]}", flush=True)
    Path(__file__).with_name(f"val-{tag}.json").write_text(json.dumps(report, indent=1))
