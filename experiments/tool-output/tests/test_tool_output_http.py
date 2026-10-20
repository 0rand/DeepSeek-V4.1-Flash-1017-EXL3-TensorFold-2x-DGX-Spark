"""Loopback tests of the transformed pinned HTTP fixture with a fake app."""
import http.client
import json
from pathlib import Path
import threading
import unittest
from http.server import ThreadingHTTPServer
from types import ModuleType

from support import ROOT, fixtures, load

TOOLS = [{"type": "function", "function": {"name": "task"}}]
KNOWN = '<tool_call>{"name":"task","arguments":{}}</tool_call>'
UNKNOWN = KNOWN.replace('"task"', '"unoffered"')
apply = load("tool_http_apply", ROOT / "apply.py")


class ToolOutputHttpTests(unittest.TestCase):
    def setUp(self):
        ctx = fixtures()
        self.guard = ctx.__enter__()
        self.addCleanup(ctx.__exit__, None, None, None)
        source = (Path(__file__).parent / "fixtures/http.py.txt").read_text()
        self.assertEqual(__import__("hashlib").sha256(source.encode()).hexdigest(), apply.FILES[apply.HTTP])
        http = ModuleType("fixture_http")
        exec(compile(apply.transform(apply.HTTP, source), "fixture_http", "exec"), http.__dict__)
        guard = self.guard
        self.runs = []
        runs = self.runs

        class FakeApp:
            def prepare(self, body, chat):
                return None

            def reply_model(self, body):
                return "fixture"

            def run(self, body, chat, emit, **kwargs):
                runs.append(body)
                tools = body.get("tools") or []
                validator = guard.ToolOutputGuard(tools, request=body)
                text = body["answer"]
                sent = 0
                for size in range(1, len(text) + 1):
                    shown = validator.stream(text[:size])
                    if len(shown) > sent:
                        emit({"content": shown[sent:]})
                        sent = len(shown)
                content, calls = validator.parse(text)
                return {"content": content, "reasoning": "", "calls": calls,
                        "final": {"content": content[sent:]} if content[sent:] else {},
                        "finish": "tool_calls" if calls else "stop", "stats": {},
                        "prompt_tokens": 1, "completion_tokens": len(text)}

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), http.make_handler(FakeApp()))
        worker = threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        worker.start()
        def close():
            self.httpd.shutdown()
            self.httpd.server_close()
            worker.join()
        self.addCleanup(close)

    def post(self, body):
        connection = http.client.HTTPConnection("127.0.0.1", self.httpd.server_port, timeout=3)
        try:
            connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
            response = connection.getresponse()
            return response.status, response.read().decode()
        finally:
            connection.close()

    def test_rejected_sse_has_one_error_done_no_calls_finish_or_retry(self):
        for tools in (None, [], TOOLS):
            body = {"answer": "prose " + KNOWN + UNKNOWN, "stream": True}
            if tools is not None:
                body["tools"] = tools
            status, text = self.post(body)
            frames = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
            self.assertEqual(status, 200)
            self.assertEqual(sum("error" in frame for frame in frames), 1)
            error = frames[-1]["error"]
            self.assertEqual(error["code"], "unavailable_tool_call")
            self.assertNotIn("unoffered", json.dumps(error))
            self.assertNotIn("tool_calls", text)
            self.assertFalse(any(choice["finish_reason"] for frame in frames for choice in frame.get("choices", [])))
            self.assertEqual(text.count("data: [DONE]"), 1)
            self.assertIn('"content": "p"', text)
        self.assertEqual(len(self.runs), 3)

    def test_nonstream_safe_500_then_allowed_call_recovers(self):
        status, text = self.post({"answer": UNKNOWN, "tools": TOOLS})
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(text)["error"]["type"], "model_output_error")
        status, text = self.post({"answer": KNOWN, "tools": TOOLS})
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text)["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "task")
        self.assertEqual(len(self.runs), 2)

    def test_prose_and_fenced_examples_have_no_calls(self):
        for answer in ("plain streamed answer", "```\n" + UNKNOWN + "\n```\n"):
            status, text = self.post({"answer": answer, "tools": TOOLS, "stream": True})
            self.assertEqual(status, 200)
            frames = [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]
            content = ''.join(choice["delta"].get("content", "") for frame in frames for choice in frame.get("choices", []))
            self.assertEqual(content, answer)
            self.assertFalse(any("error" in frame for frame in frames))
            self.assertNotIn('"tool_calls"', text)
