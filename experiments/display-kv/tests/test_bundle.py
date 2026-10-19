"""Prepare launch files and syntax-check them without executing the launcher."""
import importlib.util
from pathlib import Path
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("display_prepare", HERE / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class BundleTests(unittest.TestCase):
    def test_bundle_is_separate_and_refuses_overwrite(self):
        active = {name: (prepare.ROOT / name).read_bytes() for name in (".env", "start.sh")}
        with tempfile.TemporaryDirectory() as directory:
            destination = Path(directory) / "bundle"
            prepare.prepare(destination)
            start = (destination / "start.sh").read_text()
            config = (destination / ".env").read_text()
            self.assertIn("IMAGE=" + prepare.IMAGE, config)
            root_config = (prepare.ROOT / ".env").read_text()
            spill = next(line for line in root_config.splitlines() if line.startswith("SPILL_GIB="))
            self.assertIn(spill + "\n", config)
            self.assertIn("PREFILL_CHUNK=512\n", config)
            self.assertIn("--device /dev/dri", start)
            self.assertIn("-e TF_DS_DISPLAY_KV_MIB=1792", start)
            self.assertNotIn("docker rm -f tfv41-", start)
            self.assertLess(start.index("REFUSING experimental startup"), start.index('if [ "${PREFLIGHT'))
            subprocess.run(["bash", "-n", str(destination / "start.sh")], check=True)
            for name in ("stop.sh", "status.sh", "verify.sh"):
                subprocess.run(["bash", "-n", str(destination / name)], check=True)
            with self.assertRaises(ValueError):
                prepare.prepare(destination)
        for name, original in active.items():
            self.assertEqual((prepare.ROOT / name).read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
