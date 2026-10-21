"""Run transfer.sh with fixture docker and ssh commands only: no image, no node and no network."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

EXPERIMENT = Path(__file__).resolve().parents[1]


class TransferTests(unittest.TestCase):
    def run_transfer(self, tag="tensorfold-dsv41:fixture-tool-output", head_id="sha256:aaa", worker_id="sha256:aaa",
                     has_image=True):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "scripts").mkdir()
            (root / "experiments/tool-output").mkdir(parents=True)
            shutil.copyfile(EXPERIMENT / "transfer.sh", root / "experiments/tool-output/transfer.sh")
            (root / "scripts/common.sh").write_text(
                'WORKER_SSH=fixture-worker\n'
                'log() { echo "$*" >&2; }\n'
                'wssh() { echo "wssh $*" >> "$CALLS"; echo "$WORKER_ID"; }\n')
            bin_dir = root / "bin"
            bin_dir.mkdir()
            docker = bin_dir / "docker"
            docker.write_text('#!/bin/sh\necho "docker $*" >> "$CALLS"\n'
                              'case "$1" in\n'
                              '  image) [ "$HAS_IMAGE" = 1 ] || exit 1 ;;\n'
                              '  save) echo fixture-tarball ;;\n'
                              '  images) echo "$HEAD_ID" ;;\n'
                              'esac\n')
            ssh = bin_dir / "ssh"
            ssh.write_text('#!/bin/sh\necho "ssh $*" >> "$CALLS"\ncat > "$SENT"\n')
            for script in (docker, ssh):
                script.chmod(0o755)
            env = os.environ | {
                "PATH": f"{bin_dir}{os.pathsep}{os.environ['PATH']}", "CALLS": str(root / "calls"),
                "SENT": str(root / "sent"), "HEAD_ID": head_id, "WORKER_ID": worker_id,
                "HAS_IMAGE": "1" if has_image else "0",
            }
            result = subprocess.run(["bash", str(root / "experiments/tool-output/transfer.sh")] + ([tag] if tag else []),
                                    env=env, capture_output=True, text=True, timeout=20)
            calls = (root / "calls").read_text().splitlines() if (root / "calls").exists() else []
            sent = (root / "sent").read_text() if (root / "sent").exists() else None
            return result, calls, sent

    def test_the_built_tag_is_saved_and_loaded_on_the_worker_without_a_rebuild(self):
        result, calls, sent = self.run_transfer()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(sent, "fixture-tarball\n")
        self.assertIn("docker save tensorfold-dsv41:fixture-tool-output", calls)
        self.assertIn("ssh -o BatchMode=yes fixture-worker docker load", calls)
        self.assertFalse([call for call in calls if call.startswith("docker build")])
        self.assertIn("both nodes hold", result.stderr)

    def test_different_image_ids_fail(self):
        result, _, _ = self.run_transfer(worker_id="sha256:bbb")
        self.assertEqual(result.returncode, 1)
        self.assertIn("image IDs differ", result.stderr)

    def test_a_missing_image_on_the_head_stops_before_the_copy(self):
        result, calls, sent = self.run_transfer(has_image=False)
        self.assertEqual(result.returncode, 1)
        self.assertIn("build it first", result.stderr)
        self.assertIsNone(sent)
        self.assertFalse([call for call in calls if call.startswith("ssh")])

    def test_missing_or_unsafe_tag_is_refused(self):
        for tag in ("", "bad tag", "x;touch y", "-flag"):
            with self.subTest(tag=tag):
                result, calls, _ = self.run_transfer(tag=tag or None)
                self.assertEqual(result.returncode, 1)
                self.assertIn("usage", result.stderr)
                self.assertEqual(calls, [])


if __name__ == "__main__":
    unittest.main()
