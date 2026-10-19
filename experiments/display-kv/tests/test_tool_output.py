import importlib.util
import os
import sys
import unittest


SOURCE = os.environ.get("TF_DISPLAY_SOURCE", "/tmp/opencode/tfv41-display-source")
MODULE = os.path.join(os.path.dirname(__file__), "..", "overlay", "tool_output.py")


def load():
    sys.path.insert(0, os.path.join(SOURCE, "src"))
    spec = importlib.util.spec_from_file_location("tool_output_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ToolOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load()
        cls.tools = [{"type": "function", "function": {"name": "task", "parameters": {"type": "object"}}}]

    def guard(self):
        return self.module.ToolOutputGuard(self.tools)

    def assert_unavailable(self, text):
        with self.assertRaises(self.module.ModelOutputError) as raised:
            self.guard().parse(text)
        self.assertEqual(raised.exception.code, "unavailable_tool_call")
        self.assertEqual(raised.exception.error, {
            "type": "model_output_error", "code": "unavailable_tool_call",
            "message": "The model attempted to call a tool not offered by this request.",
        })
        self.assertNotIn("todowrite", str(raised.exception.error))
        self.assertNotIn("secret", str(raised.exception.error))

    def test_benign_json_task_payload_is_preserved_by_cuda_parser(self):
        content, calls = self.guard().parse('before <tool_call>{"name":"task","arguments":{"saved":"benign"}}</tool_call> after')
        self.assertEqual(content, "before  after")
        self.assertEqual(calls[0]["function"]["name"], "task")
        self.assertEqual(calls[0]["function"]["arguments"], '{"saved":"benign"}')

    def test_rejects_unknown_json_xml_glm_and_dsml(self):
        for text in (
            '<tool_call>{"function":{"name":"todowrite","arguments":{"secret":1}}}</tool_call>',
            '<tool_call><function=todowrite><parameter=x>secret</parameter></function></tool_call>',
            '<tool_call>todowrite<arg_key>x</arg_key><arg_value>secret</arg_value></tool_call>',
            '<｜DSML｜ calls><｜DSML｜ invoke name="todowrite"></｜DSML｜ invoke></｜DSML｜ calls>',
        ):
            with self.subTest(text=text): self.assert_unavailable(text)

    def test_unknown_after_valid_call_fails_even_at_limit(self):
        text = '<tool_call>{"name":"task","arguments":{}}</tool_call><tool_call>{"name":"todowrite","arguments":{}}</tool_call>'
        with self.assertRaises(self.module.ModelOutputError):
            self.guard().parse(text, max_calls=1)

    def test_reported_task_envelope_is_rejected_when_task_is_not_offered(self):
        tools = [{"type": "function", "function": {"name": "todowrite", "parameters": {"type": "object"}}}]
        text = '<tool_call>{"name":"task","arguments":{"description":"Trace spill internals","prompt":"Research only, no edits.","subagent_type":"explore","task_id":""}}</tool_call>'
        with self.assertRaises(self.module.ModelOutputError):
            self.module.ToolOutputGuard(tools).parse(text)

    def test_unknown_json_with_closing_tag_in_argument_is_still_rejected(self):
        self.assert_unavailable('<tool_call>{"name":"todowrite","arguments":{"secret":"literal </tool_call> text"}}</tool_call>')

    def test_argument_code_and_quotes_are_not_rewritten(self):
        import json
        args = {"prompt": 'Explain `x` and ``a ` b``. Quote: "example".\n```python\nx = 1\n```'}
        text = '<tool_call>' + json.dumps({"name": "task", "arguments": args}) + '</tool_call>'
        content, calls = self.guard().parse(text)
        self.assertEqual(content, "")
        self.assertEqual(json.loads(calls[0]["function"]["arguments"]), args)

    def test_offered_calls_keep_prefix_and_obey_global_limit(self):
        text = 'prefix <tool_call>{"name":"TASK","arguments":{}}</tool_call> middle <tool_call>{"name":"task","arguments":{}}</tool_call> suffix'
        content, calls = self.guard().parse(text, max_calls=1)
        self.assertEqual(content, "prefix  middle  suffix")
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0]["function"]["name"], "task")

    def test_quoted_fenced_inline_and_blockquote_calls_remain_literal(self):
        known = '<tool_call>{"name":"task","arguments":{}}</tool_call>'
        unknown = '<tool_call>{"name":"todowrite","arguments":{}}</tool_call>'
        for text in (f'"{known}"', f"'{unknown}'", f'`{known}`', f'> {unknown}', f'```\n{known}\n{unknown}\n'):
            with self.subTest(text=text):
                content, calls = self.guard().parse(text)
                self.assertEqual(content, text)
                self.assertIsNone(calls)

    def test_long_inline_delimiters_tilde_fences_and_spaced_quotes(self):
        text = '<tool_call>{"name":"todowrite","arguments":{}}</tool_call>'
        for literal in (f'``{text}``', f'~~~json\n{text}\n~~~\n', f'"  {text}  "', f'>> {text}'):
            with self.subTest(literal=literal):
                content, calls = self.guard().parse(literal)
                self.assertEqual(content, literal)
                self.assertIsNone(calls)

    def test_code_example_followed_by_real_call_is_not_executed_twice(self):
        literal = '<tool_call>{"name":"task","arguments":{"example":true}}</tool_call>'
        real = '<tool_call>{"name":"task","arguments":{"example":false}}</tool_call>'
        content, calls = self.guard().parse('```json\n' + literal + '\n```\n' + real)
        self.assertIn(literal, content)
        self.assertEqual(len(calls), 1)
        self.assertIn('false', calls[0]["function"]["arguments"])

    def test_malformed_and_truncated_wrappers_are_not_misclassified(self):
        for text in (
            '<tool_call>{"name":}</tool_call>',
            '<tool_call>{"name":"todowrite","arguments":{}',
            '<tool_call>todowrite<arg_key>x</arg_key><arg_value>v</arg_value> extra</tool_call>',
            'mention <tool_call> only',
        ):
            with self.subTest(text=text):
                content, calls = self.guard().parse(text)
                self.assertIsNone(calls)
                self.assertIn(text.replace("</tool_call>", "") if text.endswith("{}") else text[:10], content)

    def test_dsml_parameter_literal_tool_call_is_not_a_top_level_call(self):
        text = ('<｜DSML｜ calls><｜DSML｜ invoke name="task">'
                '<｜DSML｜ parameter name="note" string="true">'
                '<tool_call>{"name":"todowrite","arguments":{}}</tool_call>'
                '</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>')
        content, calls = self.guard().parse(text)
        self.assertIsNone(calls)
        self.assertIn("todowrite", content)


if __name__ == "__main__":
    unittest.main()
