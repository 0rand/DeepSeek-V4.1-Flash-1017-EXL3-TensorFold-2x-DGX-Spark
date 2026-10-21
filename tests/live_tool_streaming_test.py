"""CPU-only assessment tests. No network, model, Docker or CUDA calls."""
import contextlib
import importlib.util
import io
from pathlib import Path
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location('live_stream_probe', Path(__file__).with_name('live_tool_streaming.py'))
probe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(probe)


class ProbeAssessmentTests(unittest.TestCase):
    def record(self, events=(), *, stream=True, tools=True, status=200, error=None, hint=False):
        record = {'request': {'stream': stream, **({'tools': [probe.TOOL]} if tools else {})},
                  'http_status': status, 'transport_error': None,
                  'frames': [{}] if stream else [],
                  'headers': {'x-should-retry': 'false'} if hint else {},
                  'events': [{'frame': i, 'at_ms': i * 30, 'kind': kind, 'value': value}
                             for i, (kind, value) in enumerate(events)]}
        if error is not None:
            record['json'] = {'error': error}
        record['metrics'] = probe.metrics(record)
        return record

    @staticmethod
    def error():
        return {'code': 'unavailable_tool_call', 'type': 'model_output_error',
                'source': 'tensorfold.tool_output_guard'}

    def test_sse_rejection_does_not_require_impossible_late_http_header(self):
        record = self.record([('content', 'inert prefix '), ('error', self.error()), ('done', True)])
        self.assertEqual(probe.assess(record, 'reject')[0], 'PASS')

    def test_json_rejection_requires_500_and_retry_hint(self):
        for status, hint, expected in [(500, True, 'PASS'), (500, False, 'FAIL'), (200, True, 'FAIL')]:
            record = self.record(stream=False, status=status, error=self.error(), hint=hint)
            self.assertEqual(probe.assess(record, 'reject')[0], expected)

    def test_error_wire_and_markup_leaks_fail(self):
        for events in [
            [('error', self.error()), ('done', True), ('content', 'late')],
            [('error', self.error()), ('finish', 'stop'), ('done', True)],
            [('error', self.error()), ('done', True), ('done', True)],
            [('error', self.error()), ('error', self.error()), ('done', True)],
            [('content', '<TOOL_CALL>'), ('error', self.error()), ('done', True)],
            [('content', '<namespace:tool_call>'), ('error', self.error()), ('done', True)],
            [('invalid_frame', 'bad'), ('error', self.error()), ('done', True)],
        ]:
            with self.subTest(events=events):
                self.assertEqual(probe.assess(self.record(events), 'reject')[0], 'FAIL')

    def test_valid_offered_call_does_not_prove_negative_fixture_was_emitted(self):
        call = {'index': 0, 'id': 'call_probe', 'function': {'name': 'get_status', 'arguments': '{"service":"tensorfold"}'}}
        events = [('call', [call]), ('finish', 'tool_calls'), ('done', True)]
        for mode in ('reject', 'literal'):
            self.assertEqual(probe.assess(self.record(events), mode, probe.KNOWN)[0], 'INCONCLUSIVE')
            self.assertEqual(probe.assess(self.record(events, tools=False), mode, probe.KNOWN)[0], 'FAIL')
        self.assertEqual(probe.assess(self.record(events), 'call')[0], 'PASS')

    def test_exact_invalid_raw_fixture_is_failure_and_exact_literal_passes(self):
        events = [('content', probe.UNKNOWN), ('finish', 'stop'), ('done', True)]
        self.assertEqual(probe.assess(self.record(events), 'reject', probe.UNKNOWN)[0], 'FAIL')
        self.assertEqual(probe.assess(self.record(events), 'literal', probe.UNKNOWN)[0], 'PASS')

    def test_chunk_count_and_separate_reasoning_are_checked(self):
        events = [('content', 'one'), ('finish', 'stop'), ('done', True)]
        self.assertEqual(probe.assess(self.record(events), 'thinking_off')[0], 'FAIL')
        events.insert(0, ('content', 'two '))
        self.assertEqual(probe.assess(self.record(events), 'thinking_off')[0], 'PASS')
        events.insert(0, ('reasoning', 'private thought'))
        self.assertEqual(probe.assess(self.record(events), 'thinking_off')[0], 'FAIL')
        self.assertEqual(probe.assess(self.record(events), 'thinking_on')[0], 'PASS')

    def test_reasoning_interleaving_is_judged_per_frame(self):
        # The real engine emits the last reasoning token with the first answer
        # token in one delta; field order inside that frame is not interleaving.
        shared = self.record([('reasoning', 'thought'), ('reasoning', '.'), ('content', '17'),
                              ('content', ' more'), ('finish', 'stop'), ('done', True)])
        for event, frame in zip(shared['events'], [0, 0, 0, 1, 2, 3]):
            event['frame'] = frame
        self.assertEqual(probe.assess(shared, 'thinking_on')[0], 'PASS')
        # Reasoning in a frame after the first content frame still fails.
        late = self.record([('content', '17'), ('content', ' more'), ('reasoning', 'late'),
                            ('finish', 'stop'), ('done', True)])
        self.assertEqual(probe.assess(late, 'thinking_on')[0], 'FAIL')

    def test_buffered_mode_inverts_the_chunk_expectation(self):
        single = self.record([('reasoning', 'thought'), ('reasoning', ' more'),
                              ('content', 'the whole answer'), ('finish', 'stop'), ('done', True)])
        self.assertEqual(probe.assess(single, 'thinking_on')[0], 'FAIL')
        self.assertEqual(probe.assess(single, 'thinking_on', buffered=True)[0], 'PASS')
        multi = self.record([('content', 'the '), ('content', 'whole answer'),
                             ('finish', 'stop'), ('done', True)])
        self.assertEqual(probe.assess(multi, 'thinking_off', buffered=True)[0], 'FAIL')
        self.assertEqual(probe.assess(multi, 'thinking_off')[0], 'PASS')

    def test_refusal_transport_and_incomplete_generation_are_not_passes(self):
        for events in [[], [('content', 'I cannot copy that.'), ('finish', 'stop'), ('done', True)],
                       [('content', 'unfinished'), ('finish', 'length'), ('done', True)]]:
            self.assertEqual(probe.assess(self.record(events), 'thinking_off')[0], 'INCONCLUSIVE')
        record = self.record()
        record['transport_error'] = 'unavailable'
        self.assertEqual(probe.assess(record, 'reject')[0], 'INCONCLUSIVE')

    def test_cli_refuses_external_origins_before_network(self):
        for origin in ('https://127.0.0.1:8100', 'http://example.com', 'http://user@127.0.0.1',
                       'http://127.0.0.1/path', 'http://127.0.0.1?secret=x'):
            with self.subTest(origin=origin), patch.object(probe, 'request', side_effect=AssertionError('network')):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as caught:
                    probe.main(['--base-url', origin])
                self.assertEqual(caught.exception.code, 2)


if __name__ == '__main__':
    unittest.main()
