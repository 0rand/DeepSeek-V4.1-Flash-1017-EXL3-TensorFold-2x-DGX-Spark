"""Offline installer guards and opt-in pristine-source exporter contracts."""
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest import mock

from test_cache_metrics import engine, load

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("cache_metrics_apply", ROOT / "apply.py")
apply = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply)
SOURCE = os.environ.get("TF_METRICS_SOURCE")


class InstallerTests(unittest.TestCase):
    def test_protocol_and_unknown_paths(self):
        self.assertEqual(len(apply.FILES), 3)
        self.assertEqual(list(apply.overlays()), [apply.FAMILY / "cache_metrics.py"])
        for path, text in apply.overlays().items():
            self.assertFalse(path.is_absolute())
            ast.parse(text)
        with self.assertRaises(ValueError):
            apply.transform("other/engine.py", "")

    def test_anchor_rejection(self):
        for path in apply.FILES:
            with self.assertRaises(ValueError):
                apply.transform(path, "")

    def test_validation_failure_never_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            original = {}
            for path in apply.FILES:
                target = checkout / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text("pristine\n")
                original[path] = target.read_bytes()
            with mock.patch.object(apply.subprocess, "check_output", return_value=apply.REVISION):
                with self.assertRaisesRegex(ValueError, "fingerprint"):
                    apply.install(checkout)
                for path, raw in original.items():
                    self.assertEqual((checkout / path).read_bytes(), raw)
                overlay = checkout / next(iter(apply.overlays()))
                overlay.write_text("existing")
                with self.assertRaisesRegex(ValueError, "overlay"):
                    apply.install(checkout)
                self.assertEqual(overlay.read_text(), "existing")
            with mock.patch.object(apply.subprocess, "check_output", return_value="wrong"):
                with self.assertRaisesRegex(ValueError, "HEAD"):
                    apply.install(checkout)

    def test_all_syntax_checked_before_writes(self):
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            path = apply.FAMILY / "engine.py"
            target = checkout / path
            target.parent.mkdir(parents=True)
            target.write_text("original\n")
            digest = hashlib.sha256(target.read_bytes()).hexdigest()
            with mock.patch.object(apply, "FILES", {path: digest}), \
                 mock.patch.object(apply.subprocess, "check_output", return_value=apply.REVISION), \
                 mock.patch.object(apply, "transform", return_value="valid = True\n"), \
                 mock.patch.object(apply, "overlays", return_value={apply.FAMILY / "cache_metrics.py": "invalid !"}):
                with self.assertRaises(SyntaxError):
                    apply.install(checkout)
            self.assertEqual(target.read_text(), "original\n")

    def test_absent_optional_protocols_and_dedup(self):
        value = engine()
        del value.multi.pool.display_arena
        del value.multi.spill
        tensor = value.multi.pool.comp[0][0]
        value.multi.pool.index_k = {0: tensor}
        result = load().snapshot(value)
        self.assertEqual(result["kv_cache_payload_bytes"], 13)
        self.assertEqual(result["pool_used_tokens"] + result["pool_free_tokens"], result["pool_tokens"])
        self.assertNotIn("spill", result)
        value.multi.spill = types.SimpleNamespace(info=None)
        value.multi.pool.display_arena = object()
        self.assertEqual(load().snapshot(value)["kv_cache_backing"], "cuda")
        self.assertNotIn("spill", load().snapshot(value))


@unittest.skipUnless(SOURCE, "set TF_METRICS_SOURCE to an explicit pristine source checkout")
class MetricsContractTests(unittest.TestCase):
    def test_install_in_disposable_copy(self):
        source = Path(SOURCE)
        with tempfile.TemporaryDirectory() as tmp:
            checkout = Path(tmp)
            for path in apply.FILES:
                target = checkout / path
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes((source / path).read_bytes())
            with mock.patch.object(apply.subprocess, "check_output", return_value=apply.REVISION):
                apply.install(checkout)
                for path in apply.FILES:
                    self.assertEqual((checkout / path).read_text(),
                                     apply.transform(path, (source / path).read_text()))
                for path, text in apply.overlays().items():
                    self.assertEqual((checkout / path).read_text(), text)
                with self.assertRaisesRegex(ValueError, "overlay"):
                    apply.install(checkout)

    def test_pristine_transform_health_and_prometheus(self):
        source = Path(SOURCE)
        texts = {}
        for path, digest in apply.FILES.items():
            raw = (source / path).read_bytes()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), digest)
            texts[path.name] = apply.transform(path, raw.decode())
            ast.parse(texts[path.name])
            with self.assertRaises(ValueError):
                apply.transform(path, texts[path.name])
        memory = types.ModuleType("tensorfold.server.memory_budget")
        memory.process_footprint = lambda: None
        package = types.ModuleType("tensorfold.server")
        metrics = types.ModuleType("tensorfold.server.metrics")
        package.metrics = metrics
        health = types.ModuleType("tensorfold.cuda.health")
        modules = {"tensorfold": types.ModuleType("tensorfold"), "tensorfold.server": package,
                   "tensorfold.server.memory_budget": memory, "tensorfold.server.metrics": metrics}
        with mock.patch.dict(sys.modules, modules):
            exec(compile(texts["metrics.py"], "metrics.py", "exec"), metrics.__dict__)
            exec(compile(texts["health.py"], "health.py", "exec"), health.__dict__)
        for display in (False, True):
            value = engine(display=display, gaps=((0, 100),), filling=(types.SimpleNamespace(filled=500_000),))
            value.cache_metrics_snapshot = lambda: load().snapshot(value)
            app = types.SimpleNamespace(engine=value)
            tracker = health.Health()
            tracker.totals["requests_total"] = 3
            body = tracker.snapshot(app)
            self.assertEqual(body["requests_total"], 3)
            self.assertEqual(body["stream_kv_usage_ratio"], 0.5)
            rendered = metrics.render(app)
            self.assertIn('tensorfold:kv_cache_usage_ratio{pool="0"} 0.5', rendered)
            for key, _ in apply.GAUGES:
                self.assertIn(f"# TYPE tensorfold:{key} gauge\n", rendered)
                self.assertIn(f"tensorfold:{key} {body[key]}\n", rendered)
            backing = "display" if display else "cuda"
            self.assertIn(f'tensorfold:kv_cache_backing_info{{backing="{backing}"}} 1', rendered)
        plain = types.SimpleNamespace(engine=types.SimpleNamespace())
        self.assertNotIn("tensorfold:pool_tokens", metrics.render(plain))
        self.assertNotIn("pool_tokens", health.Health().snapshot(plain))


if __name__ == "__main__":
    unittest.main()
