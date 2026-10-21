"""Installer protocol tests use synthetic app/server inputs and a pinned HTTP fixture."""
import ast
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from support import ROOT, load

apply = load("tool_install_test", ROOT / "apply.py")
APP = 'class DsApp(App):\n    reads_ignore_eos = True\n'
SERVER = '''from tensorfold.cuda.turns import Turns, Yield
class App:
    def run(self):
        policy = ToolCallPolicy(body)
        calls_stream = ToolCallStreamer(tools) if tools and not policy.single else None
        def visible():
            answer_raw[0] = answer
        content, calls = parse_tool_calls(raw_answer, tools, max_calls=policy.max_calls) if tools else (answer, None)
        content = policy.content(content) if tools else content
'''


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.inputs = {apply.APP: APP, apply.SERVER: SERVER,
                       apply.HTTP: (ROOT / 'tests/fixtures/http.py.txt').read_text()}
        for relative, text in self.inputs.items():
            target = self.root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        hashes = {key: hashlib.sha256(value.encode()).hexdigest() for key, value in self.inputs.items()}
        self.hashes_patch = patch.object(apply, 'FILES', hashes)
        self.hashes_patch.start()
        self.addCleanup(self.hashes_patch.stop)
        self.revision_patch = patch.object(apply.subprocess, 'check_output', return_value=apply.REVISION + '\n')
        self.revision = self.revision_patch.start()
        self.addCleanup(self.revision_patch.stop)

    def snapshot(self):
        return {path.relative_to(self.root): path.read_bytes() for path in self.root.rglob('*') if path.is_file()}

    def assert_rejected_without_writes(self, exception=ValueError):
        before = self.snapshot()
        with self.assertRaises(exception):
            apply.install(self.root)
        self.assertEqual(before, self.snapshot())

    def test_revision_hash_and_overlay_refuse_before_writes(self):
        self.revision.return_value = 'wrong'
        self.assert_rejected_without_writes()
        self.revision.return_value = apply.REVISION
        target = self.root / apply.HTTP
        original = target.read_text()
        target.write_text(original + '\n# drift\n')
        self.assert_rejected_without_writes()
        target.write_text(original)
        overlay = self.root / next(iter(apply.overlays()))
        overlay.write_text('already_here = True\n')
        self.assert_rejected_without_writes()

    def test_missing_anchor_and_invalid_ast_refuse_before_writes(self):
        with patch.object(apply, 'transform', side_effect=ValueError('anchor')):
            self.assert_rejected_without_writes()
        with patch.object(apply, 'overlays', return_value={Path('src/tensorfold/cuda/tool_output.py'): 'not python!'}):
            self.assert_rejected_without_writes(SyntaxError)
        real = apply.transform
        with patch.object(apply, 'transform', side_effect=lambda key, text: 'not python!' if key == apply.HTTP else real(key, text)):
            self.assert_rejected_without_writes(SyntaxError)

    def test_success_and_reinstall_refusal(self):
        apply.install(self.root)
        for relative, text in self.inputs.items():
            result = (self.root / relative).read_text()
            self.assertNotEqual(result, text)
            ast.parse(result)
            self.assertNotIn('_flush_spill', result)
            self.assertNotIn('on_exit', result)
        for relative, text in apply.overlays().items():
            self.assertEqual((self.root / relative).read_text(), text)
        self.assert_rejected_without_writes()

    def test_pure_protocol_and_unknown_path_noop(self):
        before = self.snapshot()
        for relative, text in self.inputs.items():
            ast.parse(apply.transform(relative, text))
        self.assertEqual(apply.transform(Path('elsewhere/server.py'), 'untouched'), 'untouched')
        self.assertEqual(before, self.snapshot())
        self.assertTrue(all(isinstance(key, Path) for key in apply.FILES))
        for relative, text in apply.overlays().items():
            self.assertFalse(relative.is_absolute())
            ast.parse(text)

    def test_anchor_count_is_exact(self):
        for text in ('missing', APP + APP):
            with self.assertRaises(ValueError):
                apply.transform(apply.APP, text)


@unittest.skipUnless(os.environ.get('TF_TOOL_SOURCE'), 'set TF_TOOL_SOURCE for pristine source validation')
class RealSourceTests(unittest.TestCase):
    def test_pinned_inputs_and_transformed_ast(self):
        root = Path(os.environ['TF_TOOL_SOURCE'])
        for relative, digest in apply.FILES.items():
            raw = (root / relative).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            ast.parse(apply.transform(relative, raw.decode()))
