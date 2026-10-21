"""Evaluate only the switch check and run_args of start.sh, never the launcher or a service command."""
import os
from pathlib import Path
import subprocess
import unittest

ROOT = Path(__file__).resolve().parents[3]
START = (ROOT / "start.sh").read_text()
FUNCTION = START[START.index("run_args() {"):START.index('\nlog "starting rank 0')]
VALIDATION = START[START.index('case "${STRICT_TOOL_OUTPUT-1}"'):START.index('\nif [ "${PREFLIGHT:-1}"')]


class LauncherTests(unittest.TestCase):
    def args(self, switch, rank=0):
        env = {"PATH": os.environ["PATH"]}
        if switch is not None:
            env["STRICT_TOOL_OUTPUT"] = switch
        return subprocess.run(["bash", "-c", VALIDATION + "\n" + FUNCTION + f"\nrun_args {rank}"],
                              env=env, text=True, capture_output=True, check=False)

    def test_omitted_and_one_add_no_argument(self):
        for rank in (0, 1):
            default = self.args(None, rank)
            one = self.args("1", rank)
            self.assertEqual(default.returncode, 0, default.stderr)
            self.assertEqual(default.stdout, one.stdout)
            self.assertNotIn("TF_DS_STRICT_TOOL_OUTPUT", default.stdout)

    def test_zero_turns_the_guard_off_on_both_ranks(self):
        for rank in (0, 1):
            result = self.args("0", rank)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.count("-e TF_DS_STRICT_TOOL_OUTPUT=0"), 1)

    def test_invalid_values_fail_before_actions(self):
        for switch in ("", "2", "false", "1 extra"):
            result = self.args(switch)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("STRICT_TOOL_OUTPUT must be 0 or 1", result.stderr)
            self.assertEqual(result.stdout, "")

    def test_the_example_lists_the_switch_but_leaves_it_unset(self):
        example = (ROOT / ".env.example").read_text()
        self.assertIn("#STRICT_TOOL_OUTPUT=0", example)


if __name__ == "__main__":
    unittest.main()
