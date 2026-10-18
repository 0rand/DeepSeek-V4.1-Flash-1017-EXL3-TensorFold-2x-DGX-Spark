#!/usr/bin/env python3
"""Drive Hermes's own AIAgent against the live DGX router at every thinking level.

  set PYTHONIOENCODING=utf-8
  %LOCALAPPDATA%\\hermes\\hermes-agent\\venv\\Scripts\\python.exe hermes_thinking_modes.py
      [--levels none,minimal,...] [--modes stream,quiet] [--tasks math,tool] [--report out.json]

Isolation: HERMES_HOME is a throwaway directory holding only the model and
provider sections of the live config.yaml (no MCP servers, plugins or memory)
and is deleted at exit, so live Hermes sessions, memory and logs are never
touched. Hermes resolves the runtime provider itself (resolve_runtime_provider),
so the request shape is exactly what production sends. Every /chat/completions
body is summarized through an httpx hook (messages, tools and headers, and so
the key, are never recorded).

Per level x mode x task it checks: the reasoning field Hermes put on the wire,
whether reasoning reached Hermes (streamed deltas, the post-response callback
or the stored assistant message), the answer, and that no <think> tags leak.
"""

import argparse
import json
import os
import shutil
import sys
import tempfile
import time

# HERMES_DIR overrides the Hermes install directory (default: %LOCALAPPDATA%\hermes on Windows).
HERMES = os.environ.get("HERMES_DIR") or os.path.join(os.environ["LOCALAPPDATA"], "hermes")
# HERMES_PROVIDER_KEY_ENV names the variable that holds the provider key (the value is never printed).
PROVIDER_KEY_ENV = os.environ.get("HERMES_PROVIDER_KEY_ENV", "HERMES_CUSTOM_DGX_SPARK_API_KEY")
REPO = os.path.join(HERMES, "hermes-agent")
LEVELS = ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"]
THINKING_ON = {"low", "medium", "high", "xhigh", "max", "ultra"}
WIRE = {"none": "none", "ultra": "max"}  # what the custom profile should send; others verbatim
PROMPTS = {
    "math": "Pens cost $4 for a pack of 3. How much do 27 pens cost? Answer with the dollar amount only.",
    "tool": "Use the todo_list tool to add one task, 'buy milk'. Then tell me how many tasks are on the list.",
}
CAPTURE = []


def install_capture():
    import httpx

    def record(request):
        try:
            if request.url.path.endswith("/chat/completions"):
                body = json.loads(request.content or b"{}")
                dump = os.environ.get("HERMES_PROBE_DUMP_BODY")
                if dump and not os.path.exists(dump):
                    with open(dump, "w", encoding="utf-8") as f:
                        json.dump(body, f)
                row = {k: v for k, v in body.items() if k not in ("messages", "tools")}
                row["n_messages"] = len(body.get("messages") or [])
                row["n_tools"] = len(body.get("tools") or [])
                CAPTURE.append(row)
        except Exception as e:  # noqa: BLE001
            CAPTURE.append({"capture_error": repr(e)[:120]})

    send, asend = httpx.Client.send, httpx.AsyncClient.send

    def _send(self, request, **kw):
        record(request)
        return send(self, request, **kw)

    async def _asend(self, request, **kw):
        record(request)
        return await asend(self, request, **kw)

    httpx.Client.send, httpx.AsyncClient.send = _send, _asend


def load_provider_key(name=PROVIDER_KEY_ENV):
    """Copy the one key variable the custom provider reads from the live .env (never printed)."""
    if os.environ.get(name):
        return
    with open(os.path.join(HERMES, ".env"), encoding="utf-8") as f:
        for line in f:
            key, sep, value = line.strip().partition("=")
            if sep and key.strip() == name:
                os.environ[name] = value.strip().strip("'\"")
                return


def make_home():
    import yaml

    with open(os.path.join(HERMES, "config.yaml"), encoding="utf-8") as f:
        live = yaml.safe_load(f)
    cfg = {k: live[k] for k in ("model", "providers", "custom_providers") if k in live}
    aux = (live.get("auxiliary") or {}).get("title_generation")
    if aux:                                    # the title task routes as production does
        cfg["auxiliary"] = {"title_generation": aux}
    cfg["agent"] = {"api_max_retries": 0}
    home = tempfile.mkdtemp(prefix="hermes-thinking-probe-")
    with open(os.path.join(home, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f)
    return home


def run_one(rt, level, mode, task):
    from hermes_constants import parse_reasoning_effort
    from run_agent import AIAgent

    seen = {"reasoning": "", "reasoning_calls": 0, "content": ""}

    def on_reasoning(text):
        seen["reasoning"] += text or ""
        seen["reasoning_calls"] += 1

    def on_delta(text):
        seen["content"] += text or ""

    kw = dict(base_url=rt["base_url"], api_key=rt["api_key"], provider=rt["provider"],
              api_mode=rt.get("api_mode") or "chat_completions", model="dgx-model",
              enabled_toolsets=["todo"] if task == "tool" else [], quiet_mode=True, skip_memory=True,
              skip_context_files=True, skip_background_review=True, max_iterations=6,
              reasoning_config=parse_reasoning_effort(level), reasoning_callback=on_reasoning)
    if mode == "stream":
        kw["stream_delta_callback"] = on_delta
    first = len(CAPTURE)
    t0 = time.time()
    agent = AIAgent(**kw)
    try:
        result = agent.run_conversation(PROMPTS[task])
    finally:
        close = getattr(agent, "close", None)
        if callable(close):
            try:
                close()
            except Exception:  # noqa: BLE001
                pass
    secs = time.time() - t0
    wire = CAPTURE[first:]
    final = result.get("final_response") or ""
    msgs = result.get("messages") or []
    stored = "".join(str(m.get("reasoning") or m.get("reasoning_content") or "")
                     for m in msgs if m.get("role") == "assistant")
    tool_msgs = [m for m in msgs if m.get("role") == "tool"]
    got_reasoning = bool(seen["reasoning"].strip() or stored.strip())
    want_on = level in THINKING_ON
    efforts = sorted({str(w.get("reasoning_effort")) for w in wire})
    want_wire = WIRE.get(level, level)
    checks = {
        "wire": efforts == [want_wire],
        "reasoning": got_reasoning == want_on,
        "answer": ("36" in final) if task == "math" else (bool(tool_msgs) and "milk" in final.lower()
                                                          and ("1" in final or "one" in final.lower())),
        "no_tags": "<think>" not in final and "</think>" not in final and "<think>" not in seen["content"],
    }
    if mode == "stream":
        checks["streamed"] = bool(seen["content"].strip())
    dump_dir = os.environ.get("HERMES_PROBE_DUMP_DIR")
    if dump_dir:
        os.makedirs(dump_dir, exist_ok=True)
        with open(os.path.join(dump_dir, f"{task}-{mode}-{level}-{int(t0)}.json"), "w", encoding="utf-8") as f:
            json.dump({"final": final, "messages": msgs, "wire": wire}, f, indent=1, ensure_ascii=False, default=str)
    return {"level": level, "mode": mode, "task": task, "ok": all(checks.values()), "checks": checks,
            "s": round(secs, 1), "wire_efforts": efforts, "wire_stream": sorted({str(w.get("stream")) for w in wire}),
            "wire_other_reasoning_keys": sorted({k for w in wire for k in w
                                                 if any(s in k for s in ("reason", "think", "chat_template"))
                                                 and k != "reasoning_effort"}),
            "api_calls": len(wire), "tool_msgs": len(tool_msgs), "reasoning_chars": len(seen["reasoning"]),
            "stored_reasoning_chars": len(stored), "final": final.strip()[:70],
            "reasoning_preview": (seen["reasoning"] or stored)[:90]}


TITLE_OPENERS = [
    "My nginx reverse proxy returns 502 after I moved the app to port 9000. How do I debug it?",
    "Plan a three-day trip to Kyoto in November for two people who like gardens and food.",
    "Explain why my Rust borrow checker complains when I push to a Vec while iterating over it.",
]


def run_title(rt, opener):
    """Hermes's own session-title call: json_schema response_format, thinking disabled."""
    from agent.title_generator import generate_title

    main_runtime = {"model": "dgx-model", "provider": rt.get("provider") or "", "base_url": rt.get("base_url") or "",
                    "api_key": rt.get("api_key") or "", "api_mode": rt.get("api_mode") or "chat_completions"}
    errors = []
    first = len(CAPTURE)
    t0 = time.time()
    title = generate_title(opener, timeout=120, failure_callback=lambda *a, **k: errors.append(repr(a)[:200]),
                           main_runtime=main_runtime)
    wire = CAPTURE[first:]
    formats = sorted({str((w.get("response_format") or {}).get("type")) for w in wire})
    checks = {"title": bool(title) and 1 <= len(title.split()) <= 10 and "{" not in title and "<" not in title,
              "wire_schema": formats == ["json_schema"], "no_errors": not errors}
    return {"level": "-", "mode": "aux", "task": "title", "ok": all(checks.values()), "checks": checks,
            "s": round(time.time() - t0, 1), "wire_efforts": sorted({str(w.get("reasoning_effort")) for w in wire}),
            "api_calls": len(wire), "final": str(title)[:70], "errors": errors}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--levels", default=",".join(LEVELS))
    ap.add_argument("--modes", default="stream,quiet")
    ap.add_argument("--tasks", default="math,tool,title")
    ap.add_argument("--report", default="")
    a = ap.parse_args()
    load_provider_key()
    home = make_home()
    os.environ["HERMES_HOME"] = home
    sys.path.insert(0, REPO)
    os.chdir(home)
    rows = []
    try:
        install_capture()
        from hermes_cli.runtime_provider import resolve_runtime_provider

        rt = resolve_runtime_provider()
        print(f"runtime provider={rt.get('provider')} api_mode={rt.get('api_mode')} base_url={rt.get('base_url')} "
              f"key={'set' if rt.get('api_key') else 'MISSING'}", flush=True)
        for task in a.tasks.split(","):
            if task == "title":
                for opener in TITLE_OPENERS:
                    try:
                        row = run_title(rt, opener)
                    except Exception as e:  # noqa: BLE001
                        row = {"level": "-", "mode": "aux", "task": "title", "ok": False, "error": repr(e)[:300]}
                    rows.append(row)
                    bad = [k for k, v in (row.get("checks") or {}).items() if not v]
                    print(f"{'PASS' if row['ok'] else 'FAIL'} title wire={row.get('wire_efforts')} "
                          f"calls={row.get('api_calls')} {row.get('s')}s {('FAILED ' + ','.join(bad)) if bad else ''}"
                          f"{row.get('error', '')} title={row.get('final')!r} {row.get('errors') or ''}", flush=True)
                continue
            for mode in a.modes.split(","):
                for level in a.levels.split(","):
                    try:
                        row = run_one(rt, level, mode, task)
                    except Exception as e:  # noqa: BLE001
                        row = {"level": level, "mode": mode, "task": task, "ok": False, "error": repr(e)[:300]}
                    rows.append(row)
                    bad = [k for k, v in (row.get("checks") or {}).items() if not v]
                    print(f"{'PASS' if row['ok'] else 'FAIL'} {task:4s} {mode:6s} {level:8s} "
                          f"wire={row.get('wire_efforts')} calls={row.get('api_calls')} tools={row.get('tool_msgs')} "
                          f"reasoning={row.get('reasoning_chars')}/{row.get('stored_reasoning_chars')} "
                          f"{row.get('s')}s {('FAILED ' + ','.join(bad)) if bad else ''}{row.get('error', '')} "
                          f"final={row.get('final')!r} thinking={row.get('reasoning_preview')!r}", flush=True)
    finally:
        os.chdir(os.path.dirname(os.path.abspath(__file__)))
        shutil.rmtree(home, ignore_errors=True)
    if a.report:
        with open(a.report, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=1, ensure_ascii=False)
    passed = sum(r["ok"] for r in rows)
    print(f"\n{passed}/{len(rows)} passed")
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    sys.exit(main())
