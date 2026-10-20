import json
import unittest
from unittest.mock import patch


from support import fixtures, guard_module


def load():
    return guard_module()


class ToolOutputTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.module = load()
        cls.tools = [{"type": "function", "function": {"name": "task", "parameters": {"type": "object"}}}]

    def setUp(self):
        self.fixture_context = fixtures()
        self.fixture_context.__enter__()
        self.addCleanup(self.fixture_context.__exit__, None, None, None)

    def guard(self):
        return self.module.ToolOutputGuard(self.tools)

    def test_prose_streams_for_absent_empty_and_offered_tools(self):
        for body in ({}, {"tools": []}, {"tools": self.tools}):
            guard = self.module.ToolOutputGuard(body.get("tools"), request=body)
            text = "Ordinary streamed prose."
            for size in range(1, len(text) + 1):
                self.assertEqual(guard.stream(text[:size]), text[:size])
            self.assertEqual(guard.parse(text), (text, None))

    def test_split_openers_hold_only_candidate_not_prose(self):
        for opener in ("<tool_call>", "<TOOL_CALL>", "<｜DSML｜ calls>"):
            for split in range(1, len(opener)):
                guard = self.guard()
                self.assertEqual(guard.stream("prefix " + opener[:split]), "prefix ")
                self.assertEqual(guard.stream("prefix " + opener + "arguments"), "prefix ")
                self.assertEqual(guard.stream("prefix " + opener + "arguments and more"), "prefix ")
        guard = self.guard()
        self.assertEqual(guard.stream("a <too"), "a ")
        self.assertEqual(guard.stream("a <tooth"), "a <tooth")

    def test_fenced_example_keeps_streaming_then_real_call_is_held(self):
        literal = '```\n<tool_call>{"name":"unoffered"}</tool_call>\n```\n'
        guard = self.guard()
        for size in range(1, len(literal) + 1):
            shown = guard.stream(literal[:size])
            self.assertTrue(literal.startswith(shown))
        self.assertEqual(guard.stream(literal), literal)
        self.assertEqual(guard.stream(literal + '<tool_call>{'), literal)

    def test_real_argument_backticks_do_not_hide_later_calls(self):
        for argument in ("`", "```\n", "~~~\n", '"', "</tool_call> `", "``a ` b"):
            first = '<tool_call>' + json.dumps({"name": "task", "arguments": {"note": argument}}) + '</tool_call>'
            second = '<tool_call>{"name":"task","arguments":{"second":true}}</tool_call>'
            content, calls = self.guard().parse(first + '\n' + second)
            self.assertEqual(content, '\n')
            self.assertEqual(len(calls), 2)
            self.assertEqual(json.loads(calls[0]["function"]["arguments"]), {"note": argument})
            with self.assertRaises(self.module.ModelOutputError):
                self.guard().parse(first + '\n' + second.replace('"task"', '"unoffered"'))

    def test_unknown_call_rejected_with_no_tools(self):
        for body in ({}, {"tools": []}):
            guard = self.module.ToolOutputGuard(body.get("tools"), request=body)
            with self.assertRaises(self.module.ModelOutputError):
                guard.parse('<tool_call>{"name":"task","arguments":{}}</tool_call>')

    def assert_unavailable(self, text):
        with self.assertRaises(self.module.ModelOutputError) as raised:
            self.guard().parse(text)
        self.assertEqual(raised.exception.code, "unavailable_tool_call")
        self.assertEqual(raised.exception.error["type"], "model_output_error")
        self.assertEqual(raised.exception.error["code"], "unavailable_tool_call")
        self.assertEqual(raised.exception.error["source"], "tensorfold.tool_output_guard")
        self.assertIn(raised.exception.rejection_id, raised.exception.error["message"])
        self.assertNotIn("todowrite", str(raised.exception.error))
        self.assertNotIn("secret", str(raised.exception.error))

    def test_diagnostics_identify_wrapper_and_tools_without_arguments(self):
        cases = (
            ('<tool_call>{"name":"unoffered","arguments":{"token":"PRIVATE_PAYLOAD"}}</tool_call>', "json"),
            ('<tool_call><function=unoffered><parameter=x>PRIVATE_PAYLOAD</parameter></function></tool_call>', "qwen_xml"),
            ('<tool_call>unoffered<arg_key>x</arg_key><arg_value>PRIVATE_PAYLOAD</arg_value></tool_call>', "glm"),
            ('<｜DSML｜ calls><｜DSML｜ invoke name="unoffered"></｜DSML｜ invoke></｜DSML｜ calls>', "dsml"),
        )
        for text, kind in cases:
            with self.subTest(kind=kind), self.assertRaises(self.module.ModelOutputError) as raised:
                self.guard().parse(text)
            exc = raised.exception
            self.assertEqual(exc.diagnostic["wrapper"], kind)
            self.assertEqual(exc.diagnostic["tool_name"], "unoffered")
            self.assertEqual(exc.diagnostic["effective_tools"], ["task"])
            self.assertEqual(exc.diagnostic["span_start"], 0)
            self.assertEqual(exc.diagnostic["span_end"], len(text))
            self.assertEqual(len(exc.diagnostic["answer_sha256"]), 64)
            self.assertNotIn("PRIVATE_PAYLOAD", exc.log_line())
            self.assertEqual(json.loads(exc.log_line().split("unavailable_tool_call ", 1)[1]), exc.diagnostic)

    def test_filtered_tools_are_distinct_from_unoffered_tools_in_diagnostics(self):
        guard = self.module.ToolOutputGuard([], request={"tools": self.tools, "tool_choice": "none"})
        with self.assertRaises(self.module.ModelOutputError) as raised:
            guard.parse('<tool_call>{"name":"task","arguments":{}}</tool_call>')
        info = raised.exception.diagnostic
        self.assertEqual(info["reason"], "disallowed_by_tool_choice")
        self.assertEqual(info["declared_tools"], ["task"])
        self.assertEqual(info["effective_tools"], [])
        self.assertEqual(info["tool_choice"], "none")

    def test_diagnostics_escape_log_control_characters_and_bound_names(self):
        name = "unavailable\n" + "x" * 300
        text = '<tool_call>' + json.dumps({"name": name, "arguments": {"secret": "PRIVATE_PAYLOAD"}}) + '</tool_call>'
        with self.assertRaises(self.module.ModelOutputError) as raised:
            self.guard().parse(text)
        self.assertEqual(len(raised.exception.diagnostic["tool_name"]), 96)
        self.assertNotIn("\n", raised.exception.log_line())
        self.assertNotIn("PRIVATE_PAYLOAD", raised.exception.log_line())

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
            with self.subTest(text=text):
                self.assert_unavailable(text)

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

    def test_a_quote_in_the_prose_before_a_call_does_not_make_the_call_literal(self):
        call = '{"name":"unoffered","arguments":{"secret":"PRIVATE"}}'
        envelope = f"<tool_call>{call}</tool_call>"
        for text in (f'The user said "go"\n{envelope}', f"It's done '\n{envelope}",
                     f'He said "go" {envelope}',
                     f'"{envelope}',                                     # no closing quote after the call
                     f'"{envelope}\''):                                  # another quote mark after the call
            with self.subTest(text=text):
                with self.assertRaises(self.module.ModelOutputError):
                    self.guard().parse(text)

    def test_streaming_never_releases_a_call_that_follows_a_quote_mark(self):
        call = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        for text in (f'He said "go"\n{call}', f'"{call}"'):
            with self.subTest(text=text):
                guard = self.guard()
                for size in range(1, len(text) + 1):
                    shown = guard.stream(text[:size])
                    self.assertNotIn("PRIVATE", shown)
                    self.assertNotIn("<tool_call>", shown)

    def test_flat_name_is_used_when_the_nested_function_object_is_empty(self):
        for payload in ('{"name":"unoffered","function":{},"arguments":{"secret":"PRIVATE"}}',
                        '{"name":"unoffered","function":{"name":""},"arguments":{}}',
                        '{"name":"unoffered","function":"x","arguments":{}}'):
            with self.subTest(payload=payload):
                with self.assertRaises(self.module.ModelOutputError):
                    self.guard().parse(f"<tool_call>{payload}</tool_call>")
        content, calls = self.guard().parse('<tool_call>{"name":"task","function":{},"arguments":{}}</tool_call>')
        self.assertEqual(calls[0]["function"]["name"], "task")

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

    def test_two_separate_quoted_spans_do_not_make_the_call_literal(self):
        envelope = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        for text in (f'The user said "go" {envelope} "and left."', f"'a' {envelope} 'b'",
                     f'He said "go" and "stop" {envelope} "left"'):
            with self.subTest(text=text):
                with self.assertRaises(self.module.ModelOutputError):
                    self.guard().parse(text)
        known = '<tool_call>{"name":"task","arguments":{}}</tool_call>'
        for text in (f'He said "go" and "{known}"', f"It's an example: '{known}'", f'"{known}" and "stop"'):
            with self.subTest(text=text):
                self.assertEqual(self.guard().parse(text), (text, None))

    def test_literal_example_body_does_not_change_the_state_for_a_later_call(self):
        real = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        for mark in ("`", "```", "> "):
            literal = '<tool_call>{"name":"task","arguments":{"note":"' + mark + '"}}</tool_call>'
            for text in (f"'{literal}'\n{real}", f'"{literal}" {real}'):
                with self.subTest(mark=mark, text=text):
                    with self.assertRaises(self.module.ModelOutputError):
                        self.guard().parse(text)

    def test_streaming_scans_only_new_text(self):
        positions = []
        real = self.module.ToolOutputGuard._opener_re

        class Spy:
            @staticmethod
            def finditer(text, position=0):
                positions.append(position)
                return real.finditer(text, position)

        guard = self.guard()
        text = "word " * 400
        with patch.object(self.module.ToolOutputGuard, "_opener_re", Spy):
            for size in range(50, len(text) + 1, 50):
                self.assertEqual(guard.stream(text[:size]), text[:size])
        self.assertEqual(positions, sorted(positions))
        self.assertGreater(positions[-1], len(text) - 100)               # the last call skipped the old text

    def test_incremental_streaming_gives_the_same_result_as_a_fresh_scan(self):
        literal = '<tool_call>{"name":"task","arguments":{"note":"`"}}</tool_call>'
        real = '<tool_call>{"name":"unoffered","arguments":{"secret":"PRIVATE"}}</tool_call>'
        for text in (f"'{literal}' and then {real}", f"```\n{literal}\n```\nmore {real}", f'say "{literal}" now'):
            with self.subTest(text=text):
                guard = self.guard()
                for size in range(1, len(text) + 1):
                    self.assertEqual(guard.stream(text[:size]), self.guard().stream(text[:size]))

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
