"""Hermetic launch-bundle checks. No active configuration or service access."""

from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock

STAGE = Path(__file__).resolve().parents[1]
REPO = STAGE.parents[1]
spec = importlib.util.spec_from_file_location("display_prepare", STAGE / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(prepare)


class BundleTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name) / "recipe"
        self.root.mkdir()
        self.destination = self.root.parent / "bundle"
        (self.root / "scripts").mkdir()
        (self.root / "scripts/common.sh").write_text("# fixture only\n")
        for name in ("start.sh", "stop.sh", "status.sh", "verify.sh", "AGENTS.md"):
            (self.root / name).write_text((REPO / name).read_text())
        (self.root / ".env").write_text("IMAGE=ordinary:test\nSPILL_GIB=64\nPREFILL_CHUNK=2048\nBIND_HOST=127.0.0.1\n")
        self.enterContext(mock.patch.object(prepare, "ROOT", self.root))

    def snapshot(self):
        return {p.relative_to(self.root): p.read_bytes() for p in self.root.rglob("*") if p.is_file()}

    def test_private_settings_and_no_active_mutation(self):
        before = self.snapshot()
        with mock.patch("subprocess.run", side_effect=AssertionError("service command")):
            prepare.prepare(self.destination, "test:display")
        config = (self.destination / ".env").read_text()
        self.assertIn("IMAGE=test:display\n", config)
        self.assertIn("SPILL_GIB=0\n", config)
        self.assertIn("PREFILL_CHUNK=512\n", config)
        self.assertIn("BIND_HOST=127.0.0.1\n", config)
        start = (self.destination / "start.sh").read_text()
        self.assertIn("TF_DS_PREFILL_CHUNK=512", start)
        self.assertIn("TF_DS_DISPLAY_KV_MIB=1792", start)
        self.assertNotIn("TF_DS_PREFILL_CHUNK=2048", start)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.destination.stat().st_mode & 0o777, 0o700)
        self.assertEqual((self.destination / ".env").stat().st_mode & 0o777, 0o600)

    def test_missing_optional_settings_are_forced(self):
        (self.root / ".env").write_text("IMAGE=ordinary:test\n")
        prepare.prepare(self.destination)
        config = (self.destination / ".env").read_text()
        self.assertIn("SPILL_GIB=0\n", config)
        self.assertIn("PREFILL_CHUNK=512\n", config)

    def test_overwrite_refused_without_mutation(self):
        self.destination.mkdir()
        sentinel = self.destination / "keep"
        sentinel.write_text("untouched")
        with self.assertRaisesRegex(ValueError, "overwrite"):
            prepare.prepare(self.destination)
        self.assertEqual(sentinel.read_text(), "untouched")
        self.assertEqual(list(self.destination.iterdir()), [sentinel])

    def test_guard_precedes_preflight_and_container_removal(self):
        prepare.prepare(self.destination)
        start = (self.destination / "start.sh").read_text()
        guard = start.index("head_running=$(docker inspect")
        refusal = start.index("REFUSING experimental startup")
        self.assertLess(guard, refusal)
        self.assertLess(refusal, start.index("scripts/preflight.sh"))
        self.assertLess(refusal, start.index("docker rm tfv41-"))
        self.assertNotIn("docker rm -f tfv41-", start)
        self.assertNotIn("PREFLIGHT", start[guard:refusal])
        self.assertIn("|| exit 1", start[guard:refusal])

    def test_bad_anchor_creates_no_bundle(self):
        (self.root / "start.sh").write_text("#!/bin/bash\n")
        with self.assertRaisesRegex(ValueError, "anchor"):
            prepare.prepare(self.destination)
        self.assertFalse(self.destination.exists())

    def test_invalid_image_creates_no_bundle(self):
        with self.assertRaisesRegex(ValueError, "image"):
            prepare.prepare(self.destination, "bad image;command")
        self.assertFalse(self.destination.exists())

    def test_duplicate_configuration_creates_no_bundle(self):
        with (self.root / ".env").open("a") as config:
            config.write("SPILL_GIB=4\n")
        with self.assertRaisesRegex(ValueError, "SPILL_GIB"):
            prepare.prepare(self.destination)
        self.assertFalse(self.destination.exists())


if __name__ == "__main__":
    unittest.main()
