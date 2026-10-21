"""Replay the conservative projector without CUDA, tokenizers or a model.

The whole-answer parser is the only execution authority. Prefixes are never
parsed, since neither provisional literal ownership nor decoder failure is
monotone under continuation.
"""
import json
import random
import unittest
from unittest.mock import patch

from test_tool_output import GOOD, TOOLS, ModelOutputError, ToolOutputGuard


class ProseProjectionTests(unittest.TestCase):
    def replay(self, text, cuts):
        guard = ToolOutputGuard(TOOLS)
        previous = ''
        marker = guard._marker.search(text)
        for end in cuts:
            shown = guard.stream(text[:end])
            self.assertTrue(shown.startswith(previous), (text, end, previous, shown))
            self.assertEqual(shown, text[:len(shown)])
            if marker is not None:
                self.assertLessEqual(len(shown), marker.start())
            previous = shown
        try:
            content, calls = guard.parse(text)
        except ModelOutputError:
            self.assertIsNotNone(marker)
            return
        guard.check_stream_prefix(text, content)
        self.assertTrue(content.startswith(previous))
        return content, calls

    def test_all_character_and_two_chunk_splits(self):
        dsml = '<｜DSML｜ calls><｜DSML｜ invoke name="terminal"><｜DSML｜ parameter name="x" string="false">7</｜DSML｜ parameter></｜DSML｜ invoke></｜DSML｜ calls>'
        v4 = dsml.replace('DSML｜ calls', 'DSML｜tool_calls').replace('DSML｜ invoke', 'DSML｜invoke').replace('DSML｜ parameter', 'DSML｜parameter')
        fixtures = [
            '', 'plain answer with trailing spaces  \n', '日本語 and café 🦜',
            'a < b and c > d', 'some <html> prose </html> and < unfinished',
            'prefix ' + GOOD + ' trailing prose', GOOD + GOOD,
            'prefix ' + dsml + GOOD + ' tail', 'prefix ' + v4 + ' tail',
            '<tool_call><function=terminal><parameter=x>7</parameter></function></tool_call>',
            '<tool_call>terminal<arg_key>x</arg_key><arg_value>7</arg_value></tool_call>',
            'before `' + GOOD + '`', 'before `' + GOOD + '`' + '` trailing',
            'before `' + GOOD + '\na\nb\nend`', '"' + GOOD + '\n\n"',
            '> quoted: <tool_call>{broken\n' + GOOD,
            '~~~ ~~~\n<tool_call>{broken\n~~~\n' + GOOD,
            'prefix <tool_call>{broken', 'prefix <｜DSML｜tool_calls>',
            'prefix <namespace:tool_call>{}</namespace:tool_call>',
            'prefix <|tool_call>call:terminal{}<tool_call|>',
            'prefix <tool_callX:tool_call>{}</tool_call>',
            '<tool_call>{"name":"terminal","arguments":{"x":7,"secret":"</tool_call> literal"}}</tool_call>',
        ]
        rng = random.Random(41)
        for text in fixtures:
            with self.subTest(text=text, mode='characters'):
                self.replay(text, range(len(text) + 1))
            for cut in range(len(text) + 1):
                with self.subTest(text=text, cut=cut):
                    self.replay(text, [cut, len(text)])
            self.replay(text, sorted(set([0, len(text)] + [rng.randrange(len(text) + 1) for _ in range(20)])))

    def test_proper_marker_prefixes_cannot_escape(self):
        markers = [
            '<tool_call>', '</TOOL_CALL>', '<foo:tool_call>',
            '<' + 'namespace.' * 80 + ':tool_call>', '<名.字:ToOl_CaLl>',
            '<İ:tool_call>', '<ſ:tool_call>', '<K:tool_call>',
            '<｜DSML｜ calls>', '</｜DSML｜tool_calls>', '<|DSML| calls>',
            '<|tool_call>',
        ]
        for marker in markers:
            guard = ToolOutputGuard(TOOLS)
            self.assertIsNotNone(guard._marker.search(marker))
            for end in range(len(marker) + 1):
                with self.subTest(marker=marker, end=end):
                    self.assertEqual(guard.stream('prose ' + marker[:end]), 'prose ')

    def test_false_alarms_release_and_provisional_markers_only_freeze(self):
        guard = ToolOutputGuard(TOOLS)
        self.assertEqual(guard.stream('text <html'), 'text ')
        self.assertEqual(guard.stream('text <html>'), 'text <html>')
        self.assertEqual(guard.stream('text <html> a < b'), 'text <html> a < b')
        self.assertEqual(guard.stream('text <html> a < b <tool_call'), 'text <html> a < b ')
        # Word-boundary recognition can vanish, but freezing early is harmless.
        self.assertEqual(guard.stream('text <html> a < b <tool_callX> tail'), 'text <html> a < b ')
        content, _ = guard.parse('text <html> a < b <tool_callX> tail')
        guard.check_stream_prefix('text <html> a < b <tool_callX> tail', content)

    def test_no_parsing_or_call_allocation_during_projection(self):
        guard = ToolOutputGuard(TOOLS)
        with patch.object(guard, 'parse', side_effect=AssertionError('partial parse')), \
             patch.object(guard, '_call', side_effect=AssertionError('early executable')):
            self.assertEqual(guard.stream('prefix ' + GOOD + ' tail'), 'prefix ')
            self.assertEqual(guard.stream('prefix ' + GOOD + ' tail ' + GOOD), 'prefix ')

    def test_ownership_revisions_and_json_prefix_failures_are_real(self):
        guard = ToolOutputGuard(TOOLS)
        for prefix, suffix in [
            ('before `' + GOOD, '\na\nb\nend`'),
            ('"' + GOOD, '\n\n"'),
        ]:
            self.assertTrue(guard.parse(prefix)[1])
            self.assertIsNone(guard.parse(prefix + suffix)[1])
        literal = 'before `' + GOOD + '`'
        self.assertIsNone(guard.parse(literal)[1])
        self.assertTrue(guard.parse(literal + '` trailing')[1])
        tools = [{'type': 'function', 'function': {'name': 'task', 'parameters': {'type': 'object'}}}]
        guard = ToolOutputGuard(tools)
        prefix = '<tool_call>{"name":"task","arguments":{"x":"</tool_call>'
        with self.assertRaises(ModelOutputError):
            guard.parse(prefix)
        complete = prefix + ' literal"}}</tool_call>'
        self.assertEqual(guard.stream(prefix), '')
        self.assertEqual(guard.stream(complete), '')
        self.assertEqual(json.loads(guard.parse(complete)[1][0]['function']['arguments']), {'x': '</tool_call> literal'})

    def test_decoder_resets_rewrites_and_final_mismatch_fail_closed(self):
        for replacement in ('', 'prefi', 'different'):
            guard = ToolOutputGuard(TOOLS)
            guard.stream('prefix ')
            with self.assertRaises(ModelOutputError):
                guard.stream(replacement)
        guard = ToolOutputGuard(TOOLS)
        guard.stream('prefix ' + GOOD)
        self.assertEqual(guard.stream('prefix ' + GOOD), 'prefix ')
        for text, content in [('rewritten', 'prefix '), ('prefix ' + GOOD, 'missing prefix')]:
            with self.assertRaises(ModelOutputError):
                guard.check_stream_prefix(text, content)

    def test_projectors_are_request_local(self):
        first, second = ToolOutputGuard(TOOLS), ToolOutputGuard([])
        self.assertEqual(first.stream('prefix ' + GOOD), 'prefix ')
        self.assertEqual(second.stream('ordinary answer'), 'ordinary answer')
        self.assertEqual(first.stream('prefix ' + GOOD + ' tail'), 'prefix ')
        self.assertEqual(second.stream('ordinary answer grows'), 'ordinary answer grows')

    def test_long_pending_prefix_freezes_without_rejecting_or_truncating(self):
        guard = ToolOutputGuard(TOOLS)
        text = 'ordinary <' + 'x' * (guard._pending_scan_limit + 1)
        self.assertEqual(guard.stream(text), 'ordinary ')
        self.assertEqual(guard._stream_cut, len('ordinary '))
        text += '> not a protocol marker'
        self.assertEqual(guard.stream(text), 'ordinary ')
        self.assertEqual(guard.parse(text), (text, None))
        guard.check_stream_prefix(text, text)


if __name__ == '__main__':
    unittest.main()
