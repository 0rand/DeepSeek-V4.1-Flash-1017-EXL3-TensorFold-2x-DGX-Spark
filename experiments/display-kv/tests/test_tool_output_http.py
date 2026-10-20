"""HTTP contracts for the opt-in strict DeepSeek tool-output guard (TF_DS_STRICT_TOOL_OUTPUT=1).

These tests execute the transformer output, rather than an installed checkout.
They intentionally use no model, CUDA, tokenizers, or chat-template construction.
"""

import http.client
import importlib
import importlib.util
import json
import os
import sys
import threading
import unittest
from contextlib import contextmanager
from http.server import ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import patch


ROOT = os.path.dirname(os.path.dirname(__file__))
SOURCE = os.environ.get("TF_DISPLAY_SOURCE", "/tmp/opencode/tfv41-display-source")
SRC = os.path.join(SOURCE, "src")
sys.path.insert(0, SRC)

apply_spec = importlib.util.spec_from_file_location("display_kv_apply_for_http", os.path.join(ROOT, "apply.py"))
apply = importlib.util.module_from_spec(apply_spec)
apply_spec.loader.exec_module(apply)

TOOLS = [{"type": "function", "function": {"name": "task", "parameters": {"type": "object"}}}]
KNOWN = '<tool_call>{"name":"task","arguments":{"value":1}}</tool_call>'
UNKNOWN = '<tool_call>{"name":"unoffered","arguments":{"secret":1}}</tool_call>'
MIXED = '<tool_call><function=task></function></tool_call>' + UNKNOWN
LEAD = "Let me check the task list first. "
# Two real calls: the first has a file with an unclosed fence and a stray backtick in its argument.
FENCED = ('<tool_call>{"name":"task","arguments":{"text":"```js\\nconst a = `x\\n"}}</tool_call>'
          '<tool_call>{"name":"task","arguments":{"value":2}}</tool_call>')


def exec_transformed(name, relative):
    """Execute one transformed source file under its real package name."""
    path = os.path.join(SRC, relative)
    with open(path) as source:
        text = source.read()
    text = apply.transform(relative, text)
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module  # dataclasses and intra-module imports require registration during exec.
    exec(compile(text, path, "exec"), module.__dict__)
    return module


def load_modules():
    importlib.import_module("tensorfold.cuda")
    tool_spec = importlib.util.spec_from_file_location(
        "tensorfold.cuda.tool_output", os.path.join(ROOT, "overlay", "tool_output.py"))
    tool = importlib.util.module_from_spec(tool_spec)
    sys.modules[tool_spec.name] = tool
    tool_spec.loader.exec_module(tool)
    http = exec_transformed("tensorfold.cuda.http", "tensorfold/cuda/http.py")
    server = exec_transformed("tensorfold.cuda.server", "tensorfold/cuda/server.py")
    app = exec_transformed("tensorfold.families.deepseek_v41.cuda.app", "tensorfold/families/deepseek_v41/cuda/app.py")
    return tool, http, server, app


class CharTokenizer:
    def encode(self, text, **kwargs):
        return SimpleNamespace(ids=[ord(char) for char in text])

    def decode(self, ids, **kwargs):
        return "".join(chr(token) for token in ids if token)

    def token_to_id(self, text):
        return None


class FakeEngine:
    eos = (0,)
    concurrent = False

    def __init__(self):
        self.request = threading.local()
        self.calls = []

    def generate(self, prompt, max_tokens, sampling, on_tokens, draft=True):
        route = chr(prompt[-1])
        text = {"A": KNOWN, "B": UNKNOWN, "C": "ordinary answer", "D": "<think>" + UNKNOWN + "</think>answer",
                "E": '<｜DSML｜ calls><｜DSML｜ invoke name="task"></｜DSML｜ invoke></｜DSML｜ calls>',
                "F": MIXED, "G": '```\n' + UNKNOWN + '\n```', "H": '"' + KNOWN + '"',
                "I": LEAD + KNOWN, "J": LEAD + UNKNOWN, "K": FENCED}[route]
        self.calls.append((route, list(prompt), max_tokens, getattr(self.request, "stop_eos", None)))
        for char in text:  # Deliberately split every envelope across callback rounds.
            if on_tokens([ord(char)]):
                break
        return {"generated": len(text), "rounds": len(text)}


def app_for(cls, server):
    app = cls.__new__(cls)  # Avoid App.__init__: it needs tokenizers and a Jinja template.
    app.engine = FakeEngine()
    app.served, app.aliases = "fake-deepseek", ()
    app.tok = CharTokenizer()
    app.default_thinking = False
    app.sampling, app.max_tokens = {"temperature": 0.0}, 1024
    app.context_window = app.native_context_window = 0
    app.vision = None

    def prepare(body, chat):
        from tensorfold.server.tools import active_tool_specs
        tools = active_tool_specs(body.get("tools"), body.get("tool_choice"))
        route = body.get("route", "C")
        return server.PreparedRequest([ord(char) for char in "route:" + route], 1024, tools,
                                      bool(body.get("thinking")), None)

    app.prepare = prepare
    return app


@contextmanager
def http_server(handler, app):
    class QuietServer(ThreadingHTTPServer):
        def handle_error(self, request, client_address):
            pass

    httpd = QuietServer(("127.0.0.1", 0), handler.make_handler(app))
    worker = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    worker.start()
    try:
        yield httpd.server_port
    finally:
        httpd.shutdown()
        httpd.server_close()
        worker.join()


def post(port, body):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
        reply = connection.getresponse()
        return reply.status, dict(reply.getheaders()), reply.read().decode()
    finally:
        connection.close()


def frames(text):
    return [json.loads(line[6:]) for line in text.splitlines() if line.startswith("data: {")]


@unittest.skipUnless(os.path.isdir(SOURCE), "needs TF_DISPLAY_SOURCE: a checkout of the pinned TensorFold revision (docs/DISPLAY-KV.md, Offline tests)")
class ToolOutputHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool, cls.http, cls.server, family = load_modules()
        cls.ds_app = family.DsApp
        cls.env = patch.dict(os.environ, {"TF_DS_STRICT_TOOL_OUTPUT": "1"})   # the guard is opt-in
        cls.env.start()

    @classmethod
    def tearDownClass(cls):
        cls.env.stop()

    def request(self, route, *, tools=None, stream=False, **extra):
        body = {"messages": [{"role": "user", "content": "x"}], "route": route, "stream": stream, **extra}
        if tools is not None:
            body["tools"] = tools
        return body

    def test_family_capability_and_generic_legacy_behavior(self):
        self.assertTrue(app_for(self.ds_app, self.server).strict_tool_output)
        self.assertFalse(hasattr(self.server.App, "strict_tool_output"))
        generic = app_for(self.server.App, self.server)
        with http_server(self.http, generic) as port:
            status, _, text = post(port, self.request("B", tools=TOOLS))
        self.assertEqual(status, 200)
        self.assertIn("unoffered", json.loads(text)["choices"][0]["message"]["content"])

    def test_guard_is_off_unless_the_switch_is_set(self):
        for value in (None, "0", "", "true"):
            with self.subTest(value=value):
                env = {} if value is None else {"TF_DS_STRICT_TOOL_OUTPUT": value}
                with patch.dict(os.environ, env):
                    if value is None:
                        os.environ.pop("TF_DS_STRICT_TOOL_OUTPUT", None)
                    app = app_for(self.ds_app, self.server)
                    self.assertFalse(app.strict_tool_output)
                    with http_server(self.http, app) as port:
                        status, _, text = post(port, self.request("B", tools=TOOLS))
                self.assertEqual(status, 200)
                self.assertIn("unoffered", json.loads(text)["choices"][0]["message"].get("content") or "")

    def test_nonstream_unknown_is_safe_422_and_recovers(self):
        app = app_for(self.ds_app, self.server)
        with http_server(self.http, app) as port:
            status, _, text = post(port, self.request("B", tools=TOOLS))
            self.assertEqual(status, 422)
            error = json.loads(text)["error"]
            self.assertEqual(error["type"], "model_output_error")
            self.assertEqual(error["code"], "unavailable_tool_call")
            self.assertEqual(error["source"], "tensorfold.tool_output_guard")
            self.assertIn(error["rejection_id"], error["message"])
            self.assertNotIn("unoffered", error["message"])
            self.assertNotIn("secret", error["message"])
            status, _, text = post(port, self.request("A", tools=TOOLS))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text)["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "task")
        self.assertEqual([call[0] for call in app.engine.calls], ["B", "A"])
        self.assertEqual(app.engine.calls[-1][2:], (1024, True))

    def test_stream_unknown_opens_then_sends_only_error_and_done(self):
        app = app_for(self.ds_app, self.server)
        with http_server(self.http, app) as port:
            status, headers, text = post(port, self.request("B", tools=TOOLS, stream=True))
        self.assertEqual(status, 200)
        self.assertEqual(headers["Content-Type"], "text/event-stream")
        payloads = frames(text)
        self.assertEqual(len([item for item in payloads if "error" in item]), 1)
        self.assertEqual(payloads[-1]["error"]["code"], "unavailable_tool_call")
        self.assertFalse(any("content" in choice.get("delta", {}) or "tool_calls" in choice.get("delta", {})
                             for item in payloads if "choices" in item for choice in item["choices"]))
        self.assertEqual(text.count("data: [DONE]"), 1)
        self.assertEqual(len(app.engine.calls), 1)

    def test_reasoning_streams_before_generation_finishes_and_rejection_is_identifiable(self):
        app = app_for(self.ds_app, self.server)
        release = threading.Event()

        def generate(prompt, max_tokens, sampling, on_tokens, draft=True):
            for char in "<think>Working on the request.":
                on_tokens([ord(char)])
            if not release.wait(5):
                raise TimeoutError("client did not receive streaming reasoning")
            for char in "</think>" + UNKNOWN:
                on_tokens([ord(char)])
            return {"rounds": 1}

        app.engine.generate = generate
        with http_server(self.http, app) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                body = self.request("B", tools=TOOLS, stream=True, thinking=True)
                connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
                reply = connection.getresponse()
                self.assertEqual(reply.status, 200)
                while True:
                    line = reply.readline().decode()
                    self.assertTrue(line, "stream ended before reasoning was delivered")
                    if line.startswith("data: {"):
                        item = json.loads(line[6:])
                        if any(choice.get("delta", {}).get("reasoning_content") for choice in item.get("choices", [])):
                            break
                self.assertFalse(release.is_set())
                release.set()
                tail = reply.read().decode()
                error = next(item["error"] for item in frames(tail) if "error" in item)
                self.assertEqual(error["source"], "tensorfold.tool_output_guard")
                self.assertEqual(error["reason"], "unavailable_tool")
                self.assertIn(error["rejection_id"], error["message"])
                self.assertNotIn("secret", json.dumps(error))
            finally:
                release.set()
                connection.close()

    def test_valid_calls_dsml_and_plain_content_are_structured_or_plain(self):
        app = app_for(self.ds_app, self.server)
        with http_server(self.http, app) as port:
            for route in ("A", "E"):
                status, _, text = post(port, self.request(route, tools=TOOLS))
                message = json.loads(text)["choices"][0]["message"]
                self.assertEqual(status, 200)
                self.assertEqual(message["tool_calls"][0]["function"]["name"], "task")
                self.assertNotIn("tool_call", message["content"] or "")
            status, _, text = post(port, self.request("C", tools=TOOLS))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text)["choices"][0]["message"]["content"], "ordinary answer")

    def test_valid_stream_emits_complete_call_after_validation(self):
        app = app_for(self.ds_app, self.server)
        with http_server(self.http, app) as port:
            status, _, text = post(port, self.request("A", tools=TOOLS, stream=True))
        self.assertEqual(status, 200)
        payloads = frames(text)
        self.assertFalse(any("error" in item for item in payloads))
        calls = [call for item in payloads for choice in item.get("choices", [])
                 for call in choice.get("delta", {}).get("tool_calls", [])]
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "task")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"value": 1})
        self.assertTrue(any(choice.get("finish_reason") == "tool_calls"
                            for item in payloads for choice in item.get("choices", [])))
        self.assertEqual(text.count("data: [DONE]"), 1)
        self.assertEqual(len(app.engine.calls), 1)

    def streamed_content(self, payloads):
        return "".join(choice.get("delta", {}).get("content") or ""
                       for item in payloads if "choices" in item for choice in item["choices"])

    def test_stream_sends_text_before_a_call_and_still_sends_the_call(self):
        app = app_for(self.ds_app, self.server)
        with http_server(self.http, app) as port:
            status, _, text = post(port, self.request("I", tools=TOOLS, stream=True))
        self.assertEqual(status, 200)
        payloads = frames(text)
        self.assertEqual(self.streamed_content(payloads).strip(), LEAD.strip())
        kinds = []
        for item in payloads:
            for choice in item.get("choices", []):
                delta = choice.get("delta", {})
                if delta.get("content"):
                    kinds.append("content")
                if delta.get("tool_calls"):
                    kinds.append("call")
        self.assertEqual(kinds[0], "content")
        self.assertIn("call", kinds)
        self.assertLess(kinds.index("content"), kinds.index("call"))
        self.assertNotIn("tool_call", self.streamed_content(payloads))

    def test_stream_text_before_an_unoffered_call_arrives_then_the_error(self):
        app = app_for(self.ds_app, self.server)
        with http_server(self.http, app) as port:
            status, _, text = post(port, self.request("J", tools=TOOLS, stream=True))
        self.assertEqual(status, 200)
        payloads = frames(text)
        self.assertEqual(self.streamed_content(payloads).strip(), LEAD.strip())
        self.assertEqual(payloads[-1]["error"]["code"], "unavailable_tool_call")
        self.assertFalse(any("tool_calls" in choice.get("delta", {})
                             for item in payloads if "choices" in item for choice in item["choices"]))

    def test_first_content_does_not_wait_for_the_whole_answer(self):
        app = app_for(self.ds_app, self.server)
        release = threading.Event()

        def generate(prompt, max_tokens, sampling, on_tokens, draft=True):
            for char in LEAD:
                on_tokens([ord(char)])
            if not release.wait(5):
                raise TimeoutError("client did not receive the text before the call")
            for char in KNOWN:
                on_tokens([ord(char)])
            return {"rounds": 1}

        app.engine.generate = generate
        with http_server(self.http, app) as port:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
            try:
                body = self.request("I", tools=TOOLS, stream=True)
                connection.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
                reply = connection.getresponse()
                seen = ""
                while LEAD.strip() not in seen:
                    line = reply.readline().decode()
                    self.assertTrue(line, "stream ended before the text arrived")
                    if line.startswith("data: {"):
                        seen += self.streamed_content([json.loads(line[6:])])
                self.assertFalse(release.is_set())
                release.set()
                self.assertIn("task", reply.read().decode())
            finally:
                release.set()
                connection.close()

    def test_call_arguments_with_an_open_fence_do_not_turn_the_next_call_into_text(self):
        app = app_for(self.ds_app, self.server)
        with http_server(self.http, app) as port:
            status, _, text = post(port, self.request("K", tools=TOOLS))
        self.assertEqual(status, 200)
        message = json.loads(text)["choices"][0]["message"]
        self.assertEqual([call["function"]["name"] for call in message["tool_calls"]], ["task", "task"])
        self.assertNotIn("tool_call", message["content"] or "")

    def test_atomic_mixed_and_literal_calls(self):
        app = app_for(self.ds_app, self.server)
        with http_server(self.http, app) as port:
            status, _, stream = post(port, self.request("F", tools=TOOLS, stream=True))
            self.assertEqual(status, 200)
            self.assertFalse(any("content" in choice.get("delta", {}) or "tool_calls" in choice.get("delta", {})
                                 for item in frames(stream) if "choices" in item for choice in item["choices"]))
            for route in ("G", "H"):
                status, _, text = post(port, self.request(route, tools=TOOLS))
                message = json.loads(text)["choices"][0]["message"]
                self.assertEqual(status, 200)
                self.assertIsNone(message.get("tool_calls"))
                self.assertIn("tool_call", message["content"])

    def test_reasoning_is_not_checked_as_answer_and_choice_none_is_checked(self):
        app = app_for(self.ds_app, self.server)
        with http_server(self.http, app) as port:
            status, _, text = post(port, self.request("D", tools=TOOLS, thinking=True))
            self.assertEqual(status, 200)
            message = json.loads(text)["choices"][0]["message"]
            self.assertEqual(message["content"], "answer")
            self.assertIn("unoffered", message["reasoning_content"])
            status, _, text = post(port, self.request("B", tools=TOOLS, tool_choice="none"))
            self.assertEqual(status, 422)
            status, _, text = post(port, self.request("B"))
        self.assertEqual(status, 200)
        self.assertIn("unoffered", json.loads(text)["choices"][0]["message"]["content"])

    def test_strict_path_never_constructs_incremental_tool_streamer(self):
        app = app_for(self.ds_app, self.server)
        with patch.object(self.server, "ToolCallStreamer", side_effect=AssertionError("must buffer strict output")):
            with http_server(self.http, app) as port:
                status, _, text = post(port, self.request("A", tools=TOOLS))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text)["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "task")

    def test_concurrent_requests_keep_request_tool_sets_isolated(self):
        app = app_for(self.ds_app, self.server)
        app.engine.concurrent = True
        original = app.engine.generate
        barrier = threading.Barrier(2)

        def generate(prompt, max_tokens, sampling, on_tokens, draft=True):
            barrier.wait(timeout=5)
            return original(prompt, max_tokens, sampling, on_tokens, draft=draft)

        app.engine.generate = generate
        other_tools = [{"type": "function", "function": {"name": "todowrite", "parameters": {"type": "object"}}}]
        answers = []
        with http_server(self.http, app) as port:
            threads = [threading.Thread(target=lambda body=body: answers.append(post(port, body))) for body in (
                self.request("A", tools=TOOLS), self.request("A", tools=other_tools))]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
        self.assertEqual(sorted(status for status, _, _ in answers), [200, 422])
        self.assertEqual(sorted(call[0] for call in app.engine.calls), ["A", "A"])


if __name__ == "__main__":
    unittest.main()
