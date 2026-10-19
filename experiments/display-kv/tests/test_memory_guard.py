"""No CUDA needed: replay changing host headroom against a fake allocator."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("memory_guard", ROOT / "overlay/memory_guard.py")
guard_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard_module)
GIB = 1 << 30


class MemoryGuardTests(unittest.TestCase):
    def setUp(self):
        self.reserved, self.allocated = 101 * GIB, 100 * GIB
        self.fractions = []
        self.cuda = SimpleNamespace(
            mem_get_info=lambda: (5 * GIB, 122 * GIB),
            memory_reserved=lambda: self.reserved,
            memory_allocated=lambda: self.allocated,
            set_per_process_memory_fraction=self.fractions.append,
            empty_cache=mock.Mock(side_effect=self.release))

    def release(self):
        self.reserved = self.allocated

    def refresh(self, guard, *available):
        with mock.patch.object(guard_module.Path, "read_text", side_effect=[
            f"MemAvailable: {int(value * GIB) // 1024} kB\n" for value in available]):
            guard.refresh()

    def make(self, floor=2):
        return guard_module.MemoryGuard(SimpleNamespace(cuda=self.cuda), rank=0, floor_gib=floor)

    def test_budget_recovers_after_startup_and_shrinks_with_external_pressure(self):
        guard = self.make()
        self.refresh(guard, 3)
        self.assertEqual(guard.last["ceiling_bytes"], 102 * GIB)
        self.refresh(guard, 6)
        self.assertEqual(guard.last["ceiling_bytes"], 105 * GIB)
        self.refresh(guard, 4)
        self.assertEqual(guard.last["ceiling_bytes"], 103 * GIB)
        self.cuda.empty_cache.assert_not_called()

    def test_pressure_releases_cached_peak_and_remeasures(self):
        guard = self.make()
        self.refresh(guard, 1.75, 2.75)
        self.cuda.empty_cache.assert_called_once()
        self.assertEqual(guard.last["reserved_bytes"], 100 * GIB)
        self.assertEqual(guard.last["released_bytes"], GIB)
        self.assertEqual(guard.last["ceiling_bytes"], int(100.75 * GIB))

    def test_no_growth_below_floor_and_no_pointless_cache_flush(self):
        self.reserved = self.allocated
        guard = self.make()
        self.refresh(guard, 1)
        self.assertEqual(guard.last["ceiling_bytes"], self.reserved)
        self.cuda.empty_cache.assert_not_called()

    def test_positive_floor_cannot_undercut_earlyoom(self):
        self.assertEqual(self.make(0.5).floor, int(1.5 * GIB))
        self.assertEqual(self.make(3).floor, 3 * GIB)
        guard = self.make(0)
        guard.refresh()
        self.assertEqual(self.fractions, [])
        for value in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.make(value)

    def test_missing_meminfo_fails_initialization_but_preserves_existing_cap(self):
        guard = self.make()
        with mock.patch.object(guard_module.Path, "read_text", side_effect=OSError):
            with self.assertRaisesRegex(RuntimeError, "MemAvailable"):
                guard.refresh()
        self.refresh(guard, 3)
        previous = guard.last.copy()
        with mock.patch.object(guard_module.Path, "read_text", return_value="MemFree: 1 kB"):
            guard.refresh()
        self.assertEqual(guard.last, previous)
        self.assertEqual(len(self.fractions), 1)

    def test_cap_never_exceeds_device_total(self):
        guard = self.make()
        self.refresh(guard, 100)
        self.assertEqual(self.fractions, [1.0])

    def test_cap_can_be_disabled_without_disabling_cache_cleanup_or_telemetry(self):
        with mock.patch.dict(guard_module.os.environ, {"TF_DS_MEM_CAP": "0"}):
            guard = self.make()
        self.refresh(guard, 1.75, 2.75)
        self.assertEqual(self.fractions, [])
        self.cuda.empty_cache.assert_called_once()
        self.assertFalse(guard.last["cap_enabled"])
        self.assertEqual(guard.last["released_bytes"], GIB)
        self.assertEqual(guard.last["host_available_bytes"], int(2.75 * GIB))

    def test_invalid_cap_setting_is_rejected(self):
        with mock.patch.dict(guard_module.os.environ, {"TF_DS_MEM_CAP": "false"}):
            with self.assertRaisesRegex(ValueError, "TF_DS_MEM_CAP"):
                self.make()

    def test_installer_refreshes_both_rank_prefill_and_decode_paths(self):
        apply_spec = importlib.util.spec_from_file_location("memory_guard_apply", ROOT / "apply.py")
        apply = importlib.util.module_from_spec(apply_spec)
        apply_spec.loader.exec_module(apply)
        import ast
        import os
        source = Path(os.environ.get("TF_DISPLAY_SOURCE", "/tmp/opencode/tfv41-display-source"))
        trees = {}
        for name in ("engine.py", "multi.py", "kernels.py"):
            relative = apply.FAMILY / name
            transformed = apply.transform(relative, (source / relative).read_text())
            trees[name] = ast.parse(transformed)
            self.assertNotIn("set_per_process_memory_fraction", transformed)
        for filename, method in (("engine.py", "prefill_steps"), ("multi.py", "_round")):
            node = next(n for n in ast.walk(trees[filename]) if isinstance(n, ast.FunctionDef) and n.name == method)
            refreshes = [n for n in ast.walk(node) if isinstance(n, ast.Call)
                         and isinstance(n.func, ast.Attribute) and n.func.attr == "refresh"]
            self.assertEqual(len(refreshes), 2)
        kernel = next(n for n in ast.walk(trees["kernels.py"])
                      if isinstance(n, ast.FunctionDef) and n.name == "topk_indices")
        self.assertTrue(any(isinstance(n, ast.ImportFrom) and n.module == "bounded_topk" for n in ast.walk(kernel)))


if __name__ == "__main__":
    unittest.main()
