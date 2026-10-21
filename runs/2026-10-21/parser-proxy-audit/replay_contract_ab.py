#!/usr/bin/env python3
"""Replay exact strict-stack literals through isolated copies of the deployed DLL.

Only ToolOutputContract changes between the two sides. Both request raw curing.
The synthetic upstream and proxy bind random loopback ports. No model is started.
"""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import urllib.request


def module_at(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


class Upstream(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        mixed = request["messages"][-1]["content"] == "mixed"
        content = self.server.mixed_literal if mixed else self.server.literal
        calls = [{"id": "call_fixture_native", "type": "function", "function": {"name": "get_status", "arguments": '{"service":"tensorfold"}'}}] if mixed else []
        common = {"id": "chatcmpl-fixture", "created": 1, "model": "fixture"}
        if request["stream"]:
            frames = [dict(common, object="chat.completion.chunk", choices=[{"index": 0, "delta": {"role": "assistant", "content": content}, "finish_reason": None}])]
            if calls:
                frames.append(dict(common, object="chat.completion.chunk", choices=[{"index": 0, "delta": {"tool_calls": [dict(calls[0], index=0)]}, "finish_reason": None}]))
            frames.append(dict(common, object="chat.completion.chunk", choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls" if mixed else "stop"}]))
            body = ("".join("data: " + json.dumps(frame) + "\n\n" for frame in frames) + "data: [DONE]\n\n").encode()
            content_type = "text/event-stream"
        else:
            message = {"role": "assistant", "content": content}
            if calls:
                message["tool_calls"] = calls
            body = json.dumps(dict(common, object="chat.completion", choices=[{"index": 0, "message": message, "finish_reason": "tool_calls" if mixed else "stop"}])).encode()
            content_type = "application/json"
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dll", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    repo = Path(__file__).resolve().parents[3]
    probe = module_at("live_probe", repo / "tests/live_tool_streaming.py")
    guard_module = module_at("strict_guard", repo / "experiments/display-kv/overlay/tool_output.py")
    literal = '    ```\n<tool_call><name>get_status</name><parameters>{"service":"tensorfold"}</parameters></tool_call>\n```\n\n'
    mixed_literal = '    ```\n<tool_call>{"name":"example","arguments":{}}</tool_call>\n```\n\n'
    guard = guard_module.ToolOutputGuard([probe.TOOL])
    for text in (literal, mixed_literal):
        classified, calls = guard.parse(text)
        if classified != text or calls:
            raise RuntimeError("Fixture not classified as pure literal by strict guard")
    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    upstream.literal = literal
    upstream.mixed_literal = mixed_literal
    worker = threading.Thread(target=upstream.serve_forever, daemon=True)
    worker.start()
    upstream_url = "http://127.0.0.1:" + str(upstream.server_port)
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    results = []
    try:
        for contract in ("LegacyRaw", "StrictNative"):
            with socket.socket() as reservation:
                reservation.bind(("127.0.0.1", 0))
                port = reservation.getsockname()[1]
            base = "http://127.0.0.1:" + str(port)
            settings = {"RoutingOptions": {"ListenUrl": base, **{name: {"BaseUrl": upstream_url, "ToolOutputContract": contract, "CureRawToolCalls": True} for name in ("PrimaryBackend", "FastBackend", "HeavyBackend")}}, "MultimodalOptions": {"Enabled": False}, "LoopGuardOptions": {"Enabled": False, "WedgeEnabled": False}, "UnavailableToolRetryOptions": {"Enabled": False}}
            root = args.output / contract
            root.mkdir()
            (root / "appsettings.json").write_text(json.dumps(settings, indent=2) + "\n")
            env = {key: value for key, value in os.environ.items() if not key.startswith(("RoutingOptions__", "MultimodalOptions__", "LoopGuardOptions__", "UnavailableToolRetryOptions__", "ASPNETCORE_"))}
            env["ASPNETCORE_ENVIRONMENT"] = "Audit"
            with (root / "proxy.log").open("w") as log:
                process = subprocess.Popen([shutil.which("dotnet"), str(args.dll.resolve())], cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
                try:
                    for _ in range(100):
                        if process.poll() is not None:
                            raise RuntimeError("Isolated proxy stopped during startup")
                        try:
                            with opener.open(base + "/health", timeout=1) as response:
                                health = json.load(response)
                            break
                        except OSError:
                            time.sleep(0.1)
                    else:
                        raise TimeoutError("Isolated proxy startup")
                    (root / "health.json").write_text(json.dumps({"ok": health["ok"], "primary": health["primary"]}, indent=2) + "\n")
                    for case in ("literal", "mixed"):
                        for stream in (False, True):
                            payload = {"model": "fixture", "messages": [{"role": "user", "content": case}], "tools": [probe.TOOL], "stream": stream}
                            record = probe.request(base, payload, 10)
                            text = probe.text_of(record)
                            calls = probe.calls_of(record)
                            errors = [event["value"] for event in record["events"] if event["kind"] == "error"]
                            if record.get("json", {}).get("error"):
                                errors.append(record["json"]["error"])
                            preserved = record["http_status"] == 200 and not errors and text == (mixed_literal if case == "mixed" else literal) and len(calls) == (1 if case == "mixed" else 0)
                            record.update(contract=contract, case=case, preserves_contract=preserved)
                            (root / (case + ("-sse" if stream else "-json") + ".json")).write_text(json.dumps(record, indent=2) + "\n")
                            result = {"contract": contract, "case": case, "stream": stream, "http_status": record["http_status"], "preserves_contract": preserved, "call_count": len(calls), "errors": errors}
                            results.append(result)
                            print(json.dumps(result), flush=True)
                finally:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    finally:
        upstream.shutdown()
        upstream.server_close()
        worker.join(timeout=5)
    (args.output / "results.json").write_text(json.dumps({"synthetic": True, "generated_tools_executed": False, "temporary_listeners_closed": True, "results": results}, indent=2) + "\n")
    if not all(result["preserves_contract"] for result in results if result["contract"] == "StrictNative"):
        raise SystemExit("StrictNative fixture regression")


if __name__ == "__main__":
    main()
