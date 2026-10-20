"""Check that only the dependent spill bundle keeps SPILL_GIB."""
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
DISPLAY = ROOT.parent / "display-kv"
spec = importlib.util.spec_from_file_location("display_prepare_spill", DISPLAY / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class SpillBundleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        shutil.copy2(ROOT.parent.parent / "start.sh", self.root / "start.sh")
        shutil.copytree(ROOT.parent.parent / "scripts", self.root / "scripts")
        shutil.copy2(ROOT.parent.parent / ".env.example", self.root / ".env")
        for name in ("stop.sh", "status.sh", "verify.sh", "AGENTS.md"):
            shutil.copy2(ROOT.parent.parent / name, self.root / name)
        (self.root / "experiments/display-spill").mkdir(parents=True)
        shutil.copy2(ROOT / "compose.py", self.root / "experiments/display-spill/compose.py")
        self.patch = mock.patch.object(prepare, "ROOT", self.root)
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def test_spill_bundle_keeps_configured_capacity(self):
        destination = self.root / "bundle"
        prepare.prepare(destination, image="fixture:display-spill", spill=True)
        config = (destination / ".env").read_text()
        self.assertIn("SPILL_GIB=64", config)
        self.assertIn("PREFILL_CHUNK=512\n", config)
        self.assertIn("IMAGE=fixture:display-spill\n", config)
        start = (destination / "start.sh").read_text()
        self.assertIn("-e TF_DS_DISPLAY_KV_MIB=1792", start)
        self.assertIn("--device /dev/dri", start)

    def test_display_only_bundle_still_disables_spill(self):
        destination = self.root / "bundle"
        prepare.prepare(destination, image="fixture:display-kv")
        self.assertIn("SPILL_GIB=0\n", (destination / ".env").read_text())


if __name__ == "__main__":
    unittest.main()
