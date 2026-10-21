"""Check that only the dependent spill bundle keeps SPILL_GIB."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
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

    def test_spill_bundle_forwards_the_tuning_values_that_env_sets(self):
        destination = self.root / "bundle"
        prepare.prepare(destination, image="fixture:display-spill", spill=True)
        start = (destination / "start.sh").read_text()
        self.assertEqual(start.count(prepare.FORWARD_SNIPPET), 1)
        for name in ("TF_DS_SPILL_MIN", "TF_DS_SPILL_MIN_FREE_GIB", "TF_DS_SPILL_KEEP_BUILDS",
                     "TF_SPILL_WRITERS", "TF_SPILL_DIRECT", "TF_SPILL_FLUSH_S"):
            self.assertIn(name, prepare.FORWARDED_VARIABLES)
        # The snippet prints only the variables that are set.
        shell = shutil.which("bash")
        if shell is None:
            self.skipTest("needs bash")
        result = subprocess.run([shell, "-c", f"set -u; printf '%s\\n' {prepare.FORWARD_SNIPPET}"], capture_output=True,
                                text=True, env={"PATH": os.environ.get("PATH", ""), "TF_SPILL_FLUSH_S": "30",
                                                "TF_SPILL_DIRECT": ""})
        self.assertEqual(result.stdout.split(), ["-e", "TF_SPILL_FLUSH_S=30"])
        syntax = subprocess.run([shell, "-n", str(destination / "start.sh")], capture_output=True, text=True)
        self.assertEqual(syntax.returncode, 0, syntax.stderr)

    def test_display_only_bundle_does_not_forward_spill_values(self):
        destination = self.root / "bundle"
        prepare.prepare(destination, image="fixture:display-kv")
        self.assertNotIn("TF_SPILL_FLUSH_S", (destination / "start.sh").read_text())

    def test_display_only_bundle_still_disables_spill(self):
        destination = self.root / "bundle"
        prepare.prepare(destination, image="fixture:display-kv")
        self.assertIn("SPILL_GIB=0\n", (destination / ".env").read_text())

    def test_the_image_build_copies_named_files_never_a_whole_folder(self):
        # The build context is the repository root, where a private bundle-launch-* folder (with a .env file) may be.
        repo = ROOT.parent.parent
        lines = [line.split() for line in (ROOT / "Dockerfile").read_text().splitlines() if line.startswith("COPY ")]
        self.assertTrue(lines)
        copied = 0
        for _, *sources, target in lines:
            for source in sources:
                path = repo / source
                self.assertTrue(path.is_file(), f"{source} is not one file of the repository")
                copied += 1
        self.assertGreaterEqual(copied, 4)


if __name__ == "__main__":
    unittest.main()
