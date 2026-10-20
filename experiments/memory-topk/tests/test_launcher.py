"""Evaluate only the switch checks and run_args, never the launcher or a service command."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
START = (ROOT / "start.sh").read_text()
FUNCTION = START[START.index("run_args() {"):START.index('\nlog "starting rank 0')]
VALIDATION = START[START.index('case "${MEM_CAP-0}"'):START.index('\nif [ "${PREFLIGHT:-1}"')]
# Fixture functions stand in for docker and wssh: each prints the label of the image on its node.
STUBS = ('IMAGE=fixture:image\n'
         'docker() { echo "${HEAD_LABEL-}"; echo "docker $*" >> "$CALLS"; }\n'
         'wssh() { echo "${WORKER_LABEL-}"; echo "wssh $*" >> "$CALLS"; }\n')


class LauncherTests(unittest.TestCase):
    def args(self, cap, rank=0, head="1", worker="1", calls=None):
        env = {"PATH": os.environ["PATH"], "HEAD_LABEL": head, "WORKER_LABEL": worker,
               "CALLS": calls or os.devnull}
        if cap is not None:
            env["MEM_CAP"] = cap
        return subprocess.run(["bash", "-c", STUBS + VALIDATION + "\n" + FUNCTION + f"\nrun_args {rank}"],
                              env=env, text=True, capture_output=True, check=False)

    def test_omitted_and_zero_keep_original_arguments(self):
        for rank in (0, 1):
            default = self.args(None, rank)
            zero = self.args("0", rank)
            self.assertEqual(default.returncode, 0, default.stderr)
            self.assertEqual(default.stdout, zero.stdout)
            self.assertNotIn("TF_DS_MEM_CAP", default.stdout)
            self.assertIn("TF_DS_MEM_FLOOR_GIB=1.5", default.stdout)
            self.assertIn("TF_DS_PREFILL_CHUNK=2048", default.stdout)

    def test_explicit_opt_in_for_both_ranks(self):
        for rank in (0, 1):
            result = self.args("1", rank)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("-e TF_DS_MEM_CAP=1"), 1)

    def test_invalid_values_fail_before_actions(self):
        for cap in ("", "2", "false", "1 extra"):
            result = self.args(cap)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("MEM_CAP must be 0 or 1", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_cap_needs_the_overlay_label_on_both_nodes(self):
        for head, worker, ok in (("1", "1", True), ("", "1", False), ("1", "", False), ("", "", False),
                                 ("0", "1", False)):
            with self.subTest(head=head, worker=worker):
                result = self.args("1", head=head, worker=worker)
                self.assertEqual(result.returncode == 0, ok, result.stderr)
                if not ok:
                    self.assertIn("MEM_CAP=1 needs the memory-topk image", result.stderr)
                    self.assertEqual(result.stdout, "")

    def test_cap_off_never_inspects_an_image(self):
        with tempfile.TemporaryDirectory() as directory:
            calls = os.path.join(directory, "calls")
            for cap in (None, "0"):
                result = self.args(cap, head="", worker="", calls=calls)
                self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse(os.path.exists(calls))

    def test_the_feature_dockerfile_sets_the_label(self):
        dockerfile = (ROOT / "experiments/memory-topk/Dockerfile").read_text()
        self.assertIn('LABEL tensorfold.memory-topk="1"', dockerfile)


if __name__ == "__main__":
    unittest.main()
