"""Prepare launch files and syntax-check them without executing the launcher.

The test builds the bundle from a temporary copy of the repo files and of .env.example, so it needs no .env.
"""
import importlib.util
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


HERE = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("display_prepare", HERE / "prepare.py")
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)
REPO = prepare.ROOT


def fake_root(directory: Path) -> Path:
    """A copy of the files that prepare.py reads, with .env made from .env.example."""
    root = directory / "repo"
    root.mkdir()
    shutil.copytree(REPO / "scripts", root / "scripts")
    for name in ("start.sh", "stop.sh", "status.sh", "verify.sh", "AGENTS.md"):
        shutil.copy2(REPO / name, root / name)
    shutil.copy2(REPO / ".env.example", root / ".env")
    return root


class BundleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = fake_root(Path(self.tmp.name))
        self.original_root = prepare.ROOT
        prepare.ROOT = self.root
        self.addCleanup(setattr, prepare, "ROOT", self.original_root)

    def test_bundle_is_separate_and_refuses_overwrite(self):
        active = {name: (self.root / name).read_bytes() for name in (".env", "start.sh")}
        destination = Path(self.tmp.name) / "bundle"
        prepare.prepare(destination)
        start = (destination / "start.sh").read_text()
        config = (destination / ".env").read_text()
        self.assertIn("IMAGE=" + prepare.IMAGE, config)
        root_config = (self.root / ".env").read_text()
        spill = next(line for line in root_config.splitlines() if line.startswith("SPILL_GIB="))
        self.assertIn(spill + "\n", config)
        self.assertIn("PREFILL_CHUNK=512\n", config)
        self.assertIn("PREFLIGHT_NCCL_BOOTSTRAP=1\n", config)
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
            self.assertEqual((self.root / name).read_bytes(), original)

    def test_display_settings_exist_only_in_the_bundle(self):
        destination = Path(self.tmp.name) / "bundle"
        prepare.prepare(destination)
        ordinary = (self.root / "start.sh").read_text()
        bundle = (destination / "start.sh").read_text()
        for text in ("TF_DS_MEM_CAP", "TF_SPILL_WRITERS", "TF_DS_DISPLAY_KV_MIB", "${PREFILL_CHUNK"):
            self.assertNotIn(text, ordinary)
            self.assertIn(text, bundle)
        self.assertIn("-e TF_DS_MEM_CAP=${MEM_ALLOCATOR_CAP:-0}", bundle)
        self.assertIn("-e TF_DS_PREFILL_CHUNK=${PREFILL_CHUNK:-512}", bundle)
        self.assertNotIn("PREFLIGHT_NCCL_BOOTSTRAP=1", (self.root / ".env").read_text())

    def test_bundle_forwards_only_the_settings_that_are_set(self):
        snippet = prepare.FORWARD_SNIPPET
        self.assertIn(snippet, (self.make_bundle() / "start.sh").read_text())
        script = f"cat <<EOF\n{snippet}\nEOF"
        quiet = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True,
                               env={"PATH": "/usr/bin:/bin"}).stdout
        self.assertEqual(quiet.strip(), "")
        env = {"PATH": "/usr/bin:/bin", "TF_SPILL_WRITERS": "3", "TF_SPILL_FLUSH_S": "20",
               "TF_DS_SPILL_MIN": "8192", "TF_SPILL_DIRECT": "0"}
        loud = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True, env=env).stdout
        self.assertEqual(loud.split(),
                         ["-e", "TF_DS_SPILL_MIN=8192", "-e", "TF_SPILL_WRITERS=3", "-e", "TF_SPILL_DIRECT=0",
                          "-e", "TF_SPILL_FLUSH_S=20"])
        self.assertEqual(prepare.FORWARDED_VARIABLES, ("TF_DS_SPILL_MIN", "TF_DS_SPILL_MIN_FREE_GIB",
                                                       "TF_DS_SPILL_KEEP_BUILDS", "TF_SPILL_WRITERS",
                                                       "TF_SPILL_DIRECT", "TF_SPILL_FLUSH_S"))

    def test_tool_output_guard_is_on_in_the_bundle_and_has_an_off_switch(self):
        start = (self.make_bundle() / "start.sh").read_text()
        self.assertIn("-e TF_DS_STRICT_TOOL_OUTPUT=${STRICT_TOOL_OUTPUT:-1}", start)
        self.assertNotIn("STRICT_TOOL_OUTPUT", (self.root / "start.sh").read_text())
        env = self.root / ".env"
        original = env.read_text()
        for value, accepted in (("0", True), ("1", True), ("1   # keep", True), ("true", False), ("", False)):
            with self.subTest(value=value):
                env.write_text(original + "STRICT_TOOL_OUTPUT=" + value + chr(10))
                destination = Path(self.tmp.name) / ("switch-" + str(abs(hash(value))))
                if accepted:
                    prepare.prepare(destination)
                else:
                    with self.assertRaises(ValueError):
                        prepare.prepare(destination)
        env.write_text(original)

    def test_changed_start_sh_is_refused(self):
        start = self.root / "start.sh"
        text = start.read_text()
        for old in ("docker rm -f tfv41-r0", "-e TF_DS_PREFILL_CHUNK=2048", "TF_DS_SPILL_DIR=/cache/tf-ds-spill"):
            with self.subTest(anchor=old):
                self.assertIn(old, text)
                start.write_text(text.replace(old, "true", 1))
                with self.assertRaises(ValueError):
                    prepare.prepare(Path(self.tmp.name) / "refused")
        start.write_text(text)

    def make_bundle(self) -> Path:
        destination = Path(self.tmp.name) / "snippet-bundle"
        prepare.prepare(destination)
        return destination


if __name__ == "__main__":
    unittest.main()
