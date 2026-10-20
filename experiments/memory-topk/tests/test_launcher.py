"""Evaluate only run_args, never the launcher or a service command."""
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
START = (ROOT / "start.sh").read_text()
FUNCTION = START[START.index("run_args() {"):START.index('\nlog "starting rank 0')]
VALIDATION = START[START.index('case "${MEM_CAP-0}"'):START.index('\nif [ "${PREFLIGHT:-1}"')]


class LauncherTests(unittest.TestCase):
    def args(self, cap, rank=0):
        env = {"PATH": os.environ["PATH"]}
        if cap is not None:
            env["MEM_CAP"] = cap
        return subprocess.run(["bash", "-c", VALIDATION + "\n" + FUNCTION + f"\nrun_args {rank}"],
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
