"""Replay host pressure with a fake allocator. No torch import is needed."""
import importlib.util
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("memory_guard_test", ROOT / "overlay/memory_guard.py")
guard_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(guard_module)
GIB = 1 << 30


class MemoryGuardTests(unittest.TestCase):
    def setUp(self):
        patch = mock.patch.dict(os.environ, {"TF_DS_MEM_CAP": "1"})
        patch.start()
        self.addCleanup(patch.stop)
        self.reserved, self.allocated = 101 * GIB, 100 * GIB
        self.fractions = []
        self.cuda = SimpleNamespace(
            mem_get_info=mock.Mock(return_value=(5 * GIB, 122 * GIB)),
            memory_reserved=lambda: self.reserved,
            memory_allocated=lambda: self.allocated,
            set_per_process_memory_fraction=self.fractions.append,
            empty_cache=mock.Mock(side_effect=self.release))

    def release(self):
        self.reserved = self.allocated

    def make(self, floor=2):
        return guard_module.MemoryGuard(SimpleNamespace(cuda=self.cuda), rank=1, floor_gib=floor)

    def refresh(self, guard, *available):
        with mock.patch.object(guard_module.Path, "read_text", side_effect=[
                f"MemAvailable: {int(value * GIB) // 1024} kB\n" for value in available]):
            guard.refresh()

    def test_budget_recovers_and_shrinks(self):
        guard = self.make()
        for available, ceiling in ((3, 102), (6, 105), (4, 103)):
            self.refresh(guard, available)
            self.assertEqual(guard.last["ceiling_bytes"], ceiling * GIB)
        self.assertEqual(self.fractions, [102 / 122, 105 / 122, 103 / 122])
        self.cuda.empty_cache.assert_not_called()

    def test_pressure_releases_cached_peak_and_remeasures(self):
        guard = self.make()
        self.refresh(guard, 1.75, 2.75)
        self.cuda.empty_cache.assert_called_once()
        self.assertEqual(guard.last["released_bytes"], GIB)
        self.assertEqual(guard.last["reserved_bytes"], 100 * GIB)
        self.assertEqual(guard.last["ceiling_bytes"], int(100.75 * GIB))

    def test_no_growth_below_floor_or_pointless_flush(self):
        self.reserved = self.allocated
        guard = self.make()
        self.refresh(guard, 1)
        self.assertEqual(guard.last["ceiling_bytes"], self.reserved)
        self.cuda.empty_cache.assert_not_called()

    def test_floor_is_exact_and_zero_disables(self):
        self.assertEqual(self.make(0.5).floor, GIB // 2)
        guard = self.make(0)
        with mock.patch.object(guard_module.Path, "read_text") as read:
            guard.refresh()
            read.assert_not_called()
        self.assertEqual(self.fractions, [])
        for value in (-1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.make(value)

    def test_missing_meminfo_fails_initially_then_retains_budget(self):
        guard = self.make()
        with mock.patch.object(guard_module.Path, "read_text", side_effect=OSError):
            with self.assertRaisesRegex(RuntimeError, "MemAvailable"):
                guard.refresh()
        self.refresh(guard, 3)
        previous = guard.last.copy()
        for text in ("MemFree: 1 kB", "MemAvailable: bad kB"):
            with mock.patch.object(guard_module.Path, "read_text", return_value=text):
                guard.refresh()
        self.assertEqual(guard.last, previous)
        self.assertEqual(len(self.fractions), 1)

    def test_failed_second_read_keeps_first_measurement(self):
        guard = self.make()
        with mock.patch.object(guard_module.Path, "read_text", side_effect=[
                "MemAvailable: 1048576 kB", OSError()]):
            guard.refresh()
        self.assertEqual(guard.last["host_available_bytes"], GIB)
        self.assertEqual(guard.last["ceiling_bytes"], 100 * GIB)

    def test_cap_never_exceeds_device_total(self):
        guard = self.make()
        self.refresh(guard, 100)
        self.assertEqual(self.fractions, [1.0])

    def test_disabled_is_noop_without_cuda_or_proc_access(self):
        for cap in (None, "0"):
            with mock.patch.dict(os.environ, {}, clear=True):
                if cap is not None:
                    os.environ["TF_DS_MEM_CAP"] = cap
                guard = guard_module.MemoryGuard(object(), rank=0, floor_gib=2)
                with mock.patch.object(guard_module.Path, "read_text") as read:
                    guard.refresh()
                    read.assert_not_called()
                self.assertFalse(guard.cap_enabled)
                self.assertEqual(guard.last, {})
        self.cuda.mem_get_info.assert_not_called()

    def test_disabled_still_releases_idle_cache_under_pressure(self):
        with mock.patch.dict(os.environ, {"TF_DS_MEM_CAP": "0"}):
            guard = self.make()
        self.assertFalse(guard.cap_enabled)
        self.refresh(guard, 1.75, 2.75)
        self.cuda.empty_cache.assert_called_once()
        self.assertEqual(self.fractions, [])
        self.assertEqual(guard.last["cap_enabled"], False)
        self.assertEqual(guard.last["ceiling_bytes"], int(100.75 * GIB))

    def test_invalid_setting_rejected(self):
        for value in ("false", "", "2", "-1"):
            with mock.patch.dict(os.environ, {"TF_DS_MEM_CAP": value}):
                with self.assertRaisesRegex(ValueError, "TF_DS_MEM_CAP"):
                    self.make()
