"""Installer protocol checks and optional pristine-source integration."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("memory_apply_test", ROOT / "apply.py")
apply = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply)


class InstallerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkout = Path(self.temp.name)
        self.inputs = {Path("a.py"): "x = 1\n", Path("b.py"): "y = 2\n"}
        for path, text in self.inputs.items():
            (self.checkout / path).write_text(text)
        self.hashes = {path: hashlib.sha256(text.encode()).hexdigest() for path, text in self.inputs.items()}
        self.additions = {Path("guard.py"): "enabled = False\n"}
        for target, value in (("FILES", self.hashes), ("overlays", mock.Mock(return_value=self.additions)),
                              ("_git_head", mock.Mock(return_value=apply.REVISION)),
                              ("transform", mock.Mock(side_effect=lambda path, text: text + "# transformed\n"))):
            patch = mock.patch.object(apply, target, value)
            patch.start()
            self.addCleanup(patch.stop)

    def snapshot(self):
        return {path.relative_to(self.checkout): path.read_bytes()
                for path in self.checkout.rglob("*") if path.is_file()}

    def reject_unchanged(self, error=ValueError):
        before = self.snapshot()
        with self.assertRaises(error):
            apply.install(self.checkout)
        self.assertEqual(before, self.snapshot())

    def test_revision_rejected_before_writes(self):
        apply._git_head.return_value = "bad"
        self.reject_unchanged()

    def test_each_hash_rejected_before_writes(self):
        for path in self.inputs:
            with self.subTest(path=path):
                (self.checkout / path).write_text("# drift\n")
                self.reject_unchanged()
                (self.checkout / path).write_text(self.inputs[path])

    def test_overlay_presence_rejected_before_writes(self):
        (self.checkout / "guard.py").write_text("# existing\n")
        self.reject_unchanged()

    def test_dangling_overlay_symlink_rejected(self):
        (self.checkout / "guard.py").symlink_to(self.checkout / "absent")
        self.reject_unchanged()
        self.assertTrue((self.checkout / "guard.py").is_symlink())

    def test_all_generated_ast_checked_before_writes(self):
        apply.transform.side_effect = lambda path, text: "invalid (" if path.name == "b.py" else text
        self.reject_unchanged(SyntaxError)
        apply.transform.side_effect = lambda path, text: text
        self.additions[Path("guard.py")] = "invalid ("
        self.reject_unchanged(SyntaxError)

    def test_late_transform_failure_has_no_partial_writes(self):
        apply.transform.side_effect = ["x = 3\n", ValueError("anchor")]
        self.reject_unchanged()

    def test_success_and_reinstall_refusal(self):
        apply.install(self.checkout)
        for path, text in self.inputs.items():
            self.assertEqual((self.checkout / path).read_text(), text + "# transformed\n")
        self.assertEqual((self.checkout / "guard.py").read_text(), self.additions[Path("guard.py")])
        self.reject_unchanged()


class ProtocolTests(unittest.TestCase):
    def test_contract_scope_and_overlay_ast(self):
        self.assertEqual(apply.REVISION, "07aa02585f64cfb04c60a96de6554c0d4708ed46")
        self.assertEqual(set(apply.FILES), {apply.FAMILY / name for name in ("engine.py", "multi.py", "kernels.py")})
        self.assertEqual(set(apply.overlays()), {apply.FAMILY / name for name in ("memory_guard.py", "bounded_topk.py")})
        for text in apply.overlays().values():
            ast.parse(text)
        with self.assertRaises(ValueError):
            apply.transform(Path("unrelated.py"), "")

    def test_missing_duplicate_and_replaced_anchors_fail(self):
        for text in ("", "anchor anchor"):
            with self.assertRaises(ValueError):
                apply._replace_once(text, "anchor", "replacement", "fixture")
        changed = apply._replace_once("anchor", "anchor", "replacement", "fixture")
        with self.assertRaises(ValueError):
            apply._replace_once(changed, "anchor", "replacement", "fixture")


SOURCE = os.environ.get("TF_MEMORY_SOURCE")


@unittest.skipUnless(SOURCE and Path(SOURCE).is_dir(), "set TF_MEMORY_SOURCE to pristine pinned TensorFold checkout")
class SourceIntegrationTests(unittest.TestCase):
    def test_real_transform_install_and_original_ceiling_preserved(self):
        source = Path(SOURCE)
        self.assertEqual(apply._git_head(source), apply.REVISION)
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            for path, expected in apply.FILES.items():
                raw = (source / path).read_bytes()
                self.assertEqual(hashlib.sha256(raw).hexdigest(), expected)
                (checkout / path).parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(source / path, checkout / path)
            with mock.patch.object(apply, "_git_head", return_value=apply.REVISION):
                apply.install(checkout)
                with self.assertRaises(ValueError):
                    apply.install(checkout)
            for path in apply.FILES:
                original = (source / path).read_text()
                text = (checkout / path).read_text()
                tree = ast.parse(text)
                self.assertNotIn("display_kv", text)
                self.assertNotIn("cache_metrics_snapshot", text)
                with self.assertRaises(ValueError):
                    apply.transform(path, text)
                if path.name == "engine.py":
                    start = original.index('        floor = float(os.environ.get("TF_DS_MEM_FLOOR_GIB") or 2.0)')
                    end = original.index("    def warm(", start)
                    self.assertIn(original[start:end], text)
                    self.assertIn('if cap == "1":', text)
                if path.name in ("engine.py", "multi.py"):
                    method = "prefill_steps" if path.name == "engine.py" else "_round"
                    node = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == method)
                    refreshes = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                                 and isinstance(n.func, ast.Attribute) and n.func.attr == "refresh"]
                    self.assertEqual(len(refreshes), 2)
