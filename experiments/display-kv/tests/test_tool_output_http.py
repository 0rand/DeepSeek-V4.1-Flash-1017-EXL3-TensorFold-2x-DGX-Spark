"""CPU HTTP contracts against transformed or actually installed pinned source.

Use run_tool_output_cpu.py for a full install into a temporary Git checkout.
No model, CUDA, tokenizers, or chat-template construction is used.
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
SOURCE = os.environ.get("TF_DISPLAY_SOURCE", "")
SRC = os.path.join(SOURCE, "src")
sys.path.insert(0, SRC)

apply_spec = importlib.util.spec_from_file_location("display_kv_apply_for_http", os.path.join(ROOT, "apply.py"))
apply = importlib.util.module_from_spec(apply_spec)
apply_spec.loader.exec_module(apply)

TOOLS = [{"type": "function", "function": {"name": "task", "parameters": {"type": "object"}}}]
KNOWN = '<tool_call>{"id":"call_exact","name":"task","arguments":{"value":1}}</tool_call>'
UNKNOWN = '<tool_call>{"name":"unoffered","arguments":{"secret":1}}</tool_call>'
MIXED = '<tool_call><function=task></function></tool_call>' + UNKNOWN


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
    if not SOURCE:
        raise RuntimeError("set TF_DISPLAY_SOURCE or use run_tool_output_cpu.py --source")
    if os.environ.get("TF_TOOL_OUTPUT_INSTALLED") == "1":
        names = ("tensorfold.cuda.tool_output", "tensorfold.cuda.http", "tensorfold.cuda.server",
                 "tensorfold.families.deepseek_v41.cuda.app")
        modules = tuple(importlib.import_module(name) for name in names)
        for module in modules:
            if not os.path.realpath(module.__file__).startswith(os.path.realpath(SRC) + os.sep):
                raise RuntimeError(f"import escaped installed source: {module.__file__}")
        return modules
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
        prompt_text = "".join(map(chr, prompt))
        route = chr(prompt[-1]) if prompt_text.startswith("route:") else "fixture"
        text = {"A": KNOWN, "B": UNKNOWN, "C": "ordinary answer", "D": "<think>" + UNKNOWN + "</think>answer",
                "E": '<｜DSML｜ calls><｜DSML｜ invoke name="task"></｜DSML｜ invoke></｜DSML｜ calls>',
                "F": MIXED, "G": '```\n' + UNKNOWN + '\n```', "H": '"' + KNOWN + '"'}.get(route, prompt_text)
        self.calls.append((route, list(prompt), max_tokens, getattr(self.request, "stop_eos", None)))
        for char in text:  # Deliberately split every envelope across callback rounds.
            if on_tokens([ord(char)]):
                break
        on_tokens([0])  # EOS: do not mistake fixtures for continuation requests.
        return {"generated": len(text) + 1, "rounds": len(text)}


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
        return server.PreparedRequest([ord(char) for char in body.get("fixture_text", "route:" + route)], 1024, tools,
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


def answer_content(wire, stream):
    if stream:
        return ''.join(c.get('delta', {}).get('content', '') for p in frames(wire) for c in p.get('choices', []))
    return json.loads(wire)['choices'][0]['message']['content']


class ToolOutputHttpTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tool, cls.http, cls.server, family = load_modules()
        cls.ds_app = family.DsApp

    def request(self, route, *, tools=None, stream=False, **extra):
        body = {"messages": [{"role": "user", "content": "x"}], "route": route, "stream": stream, **extra}
        if tools is not None:
            body["tools"] = tools
        return body

    def test_no_native_modules_loaded(self):
        for name in ("torch", "tensorfold_cuda", "tensorfold.families.deepseek_v41.cuda.engine"):
            self.assertNotIn(name, sys.modules)

    def test_family_capability_and_generic_legacy_behavior(self):
        self.assertTrue(self.ds_app.strict_tool_output)
        self.assertFalse(hasattr(self.server.App, "strict_tool_output"))
        generic = app_for(self.server.App, self.server)
        with http_server(self.http, generic) as port:
            status, _, text = post(port, self.request("B", tools=TOOLS))
        self.assertEqual(status, 200)
        self.assertIn("unoffered", json.loads(text)["choices"][0]["message"]["content"])

    def test_nonstream_unknown_is_safe_500_and_recovers(self):
        app = app_for(self.ds_app, self.server)
        with http_server(self.http, app) as port:
            status, _, text = post(port, self.request("B", tools=TOOLS))
            self.assertEqual(status, 500)
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
                self.assertEqual(error["reason"], "invalid_executable_tool_output")
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

    def test_plain_prose_reaches_client_before_generation_finishes(self):
        prose = '  visible prose\nwith trailing spaces  '
        for tools in (None, TOOLS):
            app = app_for(self.ds_app, self.server)
            release = threading.Event()

            def generate(prompt, max_tokens, sampling, on_tokens, **kwargs):
                for char in prose:
                    on_tokens([ord(char)])
                if not release.wait(3):
                    raise RuntimeError('client did not receive prose before generation finished')
                on_tokens([0])
                return {'rounds': len(prose)}

            app.engine.generate = generate
            with http_server(self.http, app) as port:
                connection = http.client.HTTPConnection('127.0.0.1', port, timeout=5)
                try:
                    body = self.request('C', tools=tools, stream=True)
                    connection.request('POST', '/v1/chat/completions', json.dumps(body), {'Content-Type': 'application/json'})
                    reply = connection.getresponse()
                    prefix = ''
                    while True:
                        line = reply.readline().decode()
                        self.assertTrue(line, 'stream ended before prose')
                        prefix += line
                        if any(c.get('delta', {}).get('content') for p in frames(prefix) for c in p.get('choices', [])):
                            break
                    self.assertFalse(release.is_set())
                    release.set()
                    wire = prefix + reply.read().decode()
                finally:
                    release.set()
                    connection.close()
            self.assertFalse(any('error' in p for p in frames(wire)))
            content = ''.join(c.get('delta', {}).get('content', '') for p in frames(wire) for c in p.get('choices', []))
            self.assertEqual(content, prose)
            self.assertEqual(wire.count('data: [DONE]'), 1)

    def test_roll_back_to_whole_answer_buffering(self):
        app = app_for(self.ds_app, self.server)
        for enabled in ('0', 'invalid'):
            with patch.dict(os.environ, {'TF_TOOL_OUTPUT_STREAMING': enabled}):
                with http_server(self.http, app) as port:
                    status, _, wire = post(port, self.request('C', tools=TOOLS, stream=True,
                        fixture_text='inert prefix ' + KNOWN + UNKNOWN))
            self.assertEqual(status, 200)
            self.assertEqual(len([p for p in frames(wire) if 'error' in p]), 1)
            self.assertFalse(any(c.get('delta', {}).get(key) for p in frames(wire)
                for c in p.get('choices', []) for key in ('content', 'tool_calls')))

    def test_final_decoder_rewrite_is_terminal_not_silent_success(self):
        app = app_for(self.ds_app, self.server)
        original = app.tok.decode

        def rewritten(ids, **kwargs):
            text = original(ids, **kwargs)
            return 'rewritten' if text == 'ordinary answer' else text

        app.tok.decode = rewritten
        with http_server(self.http, app) as port:
            status, _, wire = post(port, self.request('C', stream=True))
        self.assertEqual(status, 200)
        errors = [p['error'] for p in frames(wire) if 'error' in p]
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0]['reason'], 'stream_prefix_changed')
        self.assertFalse(any(c.get('finish_reason') for p in frames(wire) for c in p.get('choices', [])))

    def test_cancellation_during_buffer_never_emits_executable_delta(self):
        app = app_for(self.ds_app, self.server)
        generated = []
        text = 'inert prefix ' + KNOWN + UNKNOWN

        def generate(prompt, max_tokens, sampling, on_tokens, **kwargs):
            for char in text:
                generated.append(char)
                if on_tokens([ord(char)]):
                    break
            return {'rounds': len(generated)}

        app.engine.generate = generate
        deltas = []
        with self.assertRaises(self.server.RequestCancelled):
            app.run(self.request('C', tools=TOOLS, stream=True), True,
                    lambda delta: deltas.append(delta) or True,
                    cancelled=lambda: len(generated) > len('inert prefix <tool_call>'))
        self.assertLess(len(generated), len(text))
        self.assertEqual(''.join(delta.get('content', '') for delta in deltas), 'inert prefix ')
        self.assertFalse(any(delta.get('tool_calls') for delta in deltas))

    def test_fence_info_and_blockquote_literals_never_emit_executable_calls(self):
        xml = '<tool_call><function=task><parameter=value>1</parameter></function></tool_call>'
        dsml = '<｜DSML｜tool_calls><｜DSML｜invoke name="task"><｜DSML｜parameter name="value" string="false">1</｜DSML｜parameter></｜DSML｜invoke></｜DSML｜tool_calls>'
        literals = []
        for opener in ('~~~ ~~~', '~~~ ~~~~', '~~~~ ~~~~', '~~~ ~~~ info'):
            closing = opener.split()[0]
            for body in (KNOWN, xml, dsml, UNKNOWN, '<tool_call>{broken'):
                for closed in (False, True):
                    literals.append(opener + '\n' + body + ('\n' + closing if closed else ''))
        literals.extend('> Example: ' + body for body in (KNOWN, UNKNOWN, '<tool_call>{broken'))
        for stream in (False, True):
            app = app_for(self.ds_app, self.server)
            with http_server(self.http, app) as port:
                for literal in literals:
                    with self.subTest(stream=stream, literal=literal):
                        status, _, wire = post(port, self.request('C', tools=TOOLS, stream=stream, fixture_text=literal))
                        self.assertEqual(status, 200)
                        if stream:
                            payloads = frames(wire)
                            self.assertFalse(any('error' in item for item in payloads))
                            choices = [choice for item in payloads for choice in item.get('choices', [])]
                            self.assertFalse(any(choice.get('delta', {}).get('tool_calls') for choice in choices))
                            self.assertFalse(any(choice.get('finish_reason') == 'tool_calls' for choice in choices))
                            content = ''.join(choice.get('delta', {}).get('content') or '' for choice in choices)
                            self.assertEqual(wire.count('data: [DONE]'), 1)
                        else:
                            message = json.loads(wire)['choices'][0]['message']
                            self.assertFalse(message.get('tool_calls'))
                            content = message['content']
                        self.assertEqual(content, literal)
            self.assertEqual(len(app.engine.calls), len(literals))

    def test_quote_owned_inline_state_cannot_mask_unquoted_rejection(self):
        strict_tools = [{"type": "function", "function": {"name": "task", "parameters": {
            "type": "object", "properties": {"value": {"type": "integer"}},
            "required": ["value"], "additionalProperties": False}}}]
        bads = ('<tool_call>{broken', UNKNOWN, '<tool_call>{"name":"task","arguments":{"value":"wrong"}}</tool_call>')
        cases = []
        for bad in bads:
            for prefix in ('> ', '> Example: ', '  > > Quoted: '):
                for run in ('`', '``'):
                    cases.append(prefix + run + '<tool_call>{broken\n' + bad + '\n' + run)
            cases.extend(('`\n' + bad + '\n> closing `', '`\n' + bad + '\n> quoted prose\n`', '`\n' + bad + '\n\n`'))
        for stream in (False, True):
            app = app_for(self.ds_app, self.server)
            with http_server(self.http, app) as port:
                for text in cases:
                    for valid_prefix in ('', KNOWN + '\n'):
                        with self.subTest(stream=stream, text=text, valid_prefix=bool(valid_prefix)):
                            status, _, wire = post(port, self.request('C', tools=strict_tools, stream=stream, fixture_text=valid_prefix + text))
                            self.assertEqual(status, 200 if stream else 500)
                            if stream:
                                payloads = frames(wire)
                                self.assertEqual(len([item for item in payloads if 'error' in item]), 1)
                                deltas = [choice.get('delta', {}) for item in payloads for choice in item.get('choices', [])]
                                self.assertFalse(any(delta.get('tool_calls') for delta in deltas))
                                prefix = (valid_prefix + text)[:self.tool.ToolOutputGuard._marker.search(valid_prefix + text).start()]
                                self.assertEqual(''.join(delta.get('content', '') for delta in deltas), prefix)
                                self.assertEqual(wire.count('data: [DONE]'), 1)
                            else:
                                self.assertEqual(json.loads(wire)['error']['source'], 'tensorfold.tool_output_guard')
                status, _, wire = post(port, self.request('C', stream=stream))
                self.assertEqual(status, 200)
                self.assertEqual(answer_content(wire, stream), 'ordinary answer')
            self.assertEqual(len(app.engine.calls), len(cases) * 2 + 1)

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
            self.assertEqual(status, 500)
            status, _, text = post(port, self.request("B"))
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(text)["error"]["code"], "unavailable_tool_call")

    def test_strict_path_never_constructs_incremental_tool_streamer(self):
        app = app_for(self.ds_app, self.server)
        with patch.object(self.server, "ToolCallStreamer", side_effect=AssertionError("must buffer strict output")):
            with http_server(self.http, app) as port:
                status, _, text = post(port, self.request("A", tools=TOOLS))
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(text)["choices"][0]["message"]["tool_calls"][0]["function"]["name"], "task")

    def test_atomic_schema_protocol_failures_and_recovery(self):
        bad_schema = {"type": "object", "properties": {"value": {"type": "string"}}}
        recursive = {"properties": {"value": {"not": {"$ref": "#/$defs/loop"}}},
                     "$defs": {"loop": {"$ref": "#/$defs/loop"}}}
        regex = {"properties": {"value": {"pattern": "^(a+)+$"}}}
        cases = [(TOOLS, KNOWN + UNKNOWN), (TOOLS, KNOWN + '<tool_call>{broken'),
                 (None, KNOWN), ([], KNOWN)]
        cases += [([{"type": "function", "function": {"name": "task", "parameters": schema}}], KNOWN)
                  for schema in (bad_schema, recursive, regex)]
        for stream in (False, True):
            app = app_for(self.ds_app, self.server)
            with http_server(self.http, app) as port:
                for tools, text in cases:
                    status, headers, wire = post(port, self.request("C", tools=tools, stream=stream,
                        fixture_text="PRIVATE_SENTINEL" + text))
                    self.assertEqual(status, 200 if stream else 500)
                    self.assertIn('"model_output_error"', wire)
                    self.assertNotIn("call_exact", wire)
                    if stream:
                        self.assertEqual(wire.count("data: [DONE]"), 1)
                        deltas = [choice.get('delta', {}) for frame in frames(wire) for choice in frame.get('choices', [])]
                        self.assertFalse(any(delta.get('tool_calls') for delta in deltas))
                        self.assertEqual(''.join(delta.get('content', '') for delta in deltas), 'PRIVATE_SENTINEL')
                        self.assertNotIn('PRIVATE_SENTINEL', json.dumps([p['error'] for p in frames(wire) if 'error' in p]))
                    else:
                        self.assertNotIn('PRIVATE_SENTINEL', wire)
                        self.assertEqual(headers.get("x-should-retry"), "false")
                    status, _, wire = post(port, self.request("C", stream=stream))
                    self.assertEqual(status, 200)
                    self.assertEqual(answer_content(wire, stream), 'ordinary answer')
            self.assertEqual(len(app.engine.calls), 2 * len(cases))

    def test_argument_literal_contamination_rejects_without_executable_release(self):
        from test_tool_output import TEXT_TOOLS, contamination_cases
        for stream in (False, True):
            app = app_for(self.ds_app, self.server)
            with http_server(self.http, app) as port:
                for label, text in contamination_cases():
                    with self.subTest(stream=stream,label=label):
                        status, headers, wire = post(port, self.request('C', tools=TEXT_TOOLS,
                            stream=stream, fixture_text='INERT_PREFIX'+text))
                        self.assertEqual(status, 200 if stream else 500)
                        payloads = frames(wire) if stream else [json.loads(wire)]
                        errors = [p['error'] for p in payloads if 'error' in p]
                        self.assertEqual(len(errors),1)
                        self.assertEqual(errors[0]['type'],'model_output_error')
                        self.assertEqual(errors[0]['reason'],'invalid_executable_tool_output')
                        self.assertEqual(errors[0]['source'],'tensorfold.tool_output_guard')
                        deltas = [c.get('delta', {}) for p in payloads for c in p.get('choices', [])]
                        self.assertFalse(any(delta.get('tool_calls') for delta in deltas))
                        self.assertEqual(''.join(delta.get('content', '') for delta in deltas), 'INERT_PREFIX' if stream else '')
                        self.assertFalse(any(c.get('finish_reason')
                            for p in payloads for c in p.get('choices',[])))
                        for sentinel in ('r1_exact','unoffered','secret'):
                            self.assertNotIn(sentinel,wire)
                        self.assertNotIn('INERT_PREFIX', json.dumps(errors))
                        if stream:
                            self.assertEqual(wire.count('data: [DONE]'),1)
                        else:
                            self.assertNotIn('choices',payloads[0])
                            self.assertEqual(headers.get('x-should-retry'),'false')

    def test_argument_literals_preserved_in_installed_output(self):
        from test_tool_output import TEXT_TOOLS, text_call, literal_controls
        value = '`<tool_call>{broken}</tool_call><｜DSML｜ calls>~~~'
        cases = [(literal,literal,[]) for literal in literal_controls()]
        cases += [(text_call(value),'',[value])]
        for form in ('json','nested','encoded','xml','glm','dsml','dsml4'):
            for value in ('`','``','```','line\n```\nend','line\n~~~\nend'):
                cases.append((text_call(value,form)+'\n'+text_call(value,form,'r1_second'),
                              '\n',[value,value]))
        for literal in literal_controls():
            cases.append((text_call('`')+'\n'+literal,'\n'+literal,['`']))
        for stream in (False,True):
            app = app_for(self.ds_app,self.server)
            with http_server(self.http,app) as port:
                for text, expected_content, expected_values in cases:
                    with self.subTest(stream=stream,text=text):
                        status, _, wire = post(port,self.request('C',tools=TEXT_TOOLS,
                            stream=stream,fixture_text=text))
                        self.assertEqual(status,200)
                        payloads = frames(wire) if stream else [json.loads(wire)]
                        self.assertFalse(any('error' in p for p in payloads))
                        messages = [c.get('delta',{}) if stream else c['message']
                            for p in payloads for c in p.get('choices',[])]
                        content = ''.join(m.get('content') or '' for m in messages)
                        calls = [call for m in messages for call in m.get('tool_calls') or []]
                        self.assertEqual(content,expected_content)
                        self.assertEqual([json.loads(c['function']['arguments'])['text']
                                          for c in calls],expected_values)
                        if calls and text.startswith('<tool_call>{'):
                            self.assertEqual(calls[0]['id'],'r1_exact')
                            if len(calls)==2:
                                self.assertEqual(calls[1]['id'],'r1_second')
                        if stream:
                            self.assertEqual(wire.count('data: [DONE]'),1)
                            self.assertTrue(any(c.get('finish_reason') ==
                                ('tool_calls' if calls else 'stop')
                                for p in payloads for c in p.get('choices',[])))

    def test_nullable_parallel_and_explicit_false(self):
        twice = KNOWN + KNOWN.replace("call_exact", "call_second")
        for stream in (False, True):
            app = app_for(self.ds_app, self.server)
            with http_server(self.http, app) as port:
                for parallel in (None, True, False):
                    status, _, wire = post(port, self.request("C", stream=stream, parallel_tool_calls=parallel))
                    self.assertEqual(status, 200)
                    self.assertEqual(answer_content(wire, stream), 'ordinary answer')
                    status, _, wire = post(port, self.request("A", tools=TOOLS, stream=stream,
                        parallel_tool_calls=parallel, fixture_text=twice))
                    self.assertEqual(status, 500 if parallel is False and not stream else 200)
                    self.assertEqual('"model_output_error"' in wire, parallel is False)
                    if parallel is not False:
                        self.assertIn("call_exact", wire)
                        self.assertIn("call_second", wire)
            self.assertEqual(len(app.engine.calls), 6)

    def test_retry_header_only_for_native_terminal_error(self):
        native = self.tool.ModelOutputError().error
        for payload, expected in (({"error": native}, "false"),
                ({"error": {**native, "source": "other_backend"}}, None),
                ({"error": {**native, "code": "other_error"}}, None),
                ({"error": {**native, "type": "other_error"}}, None),
                ({"error": "ordinary backend failure"}, None)):
            handler = self.http.make_handler(app_for(self.ds_app, self.server))
            class ErrorHandler(handler):
                def do_POST(self):
                    self.close_connection = True
                    self._json(500, payload)
            wrapper = SimpleNamespace(make_handler=lambda app: ErrorHandler)
            with http_server(wrapper, None) as port:
                status, headers, _ = post(port, {})
            self.assertEqual(status, 500)
            self.assertEqual(headers.get("x-should-retry"), expected)

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
        self.assertEqual(sorted(status for status, _, _ in answers), [200, 500])
        self.assertEqual(sorted(call[0] for call in app.engine.calls), ["A", "A"])


if __name__ == "__main__":
    unittest.main()
