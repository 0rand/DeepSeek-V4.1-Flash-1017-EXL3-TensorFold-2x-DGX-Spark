#!/usr/bin/env python3
"""Hermetic checks for start.sh's server-default thinking settings."""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys
import tempfile
import textwrap
import unittest


RECIPE = Path(__file__).resolve().parents[1]


class ThinkingSettingsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name) / "recipe"
        self.bin = self.root / "bin"
        self.commands = self.root / "docker-commands.jsonl"
        self.root.mkdir()
        self.bin.mkdir()
        (self.root / "scripts").mkdir()
        shutil.copy2(RECIPE / "start.sh", self.root / "start.sh")
        (self.root / "scripts" / "common.sh").write_text(textwrap.dedent("""\
            ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
            set -a
            . "$ROOT/.env"
            set +a
            LOGS="$ROOT/logs"
            log() { printf '%s\\n' "$*" >&2; }
            wssh() { printf '%s\\n' wssh >> "$TEST_OPERATIONS"; case "$*" in *tensorfold.memory-topk*) printf '1\\n' ;; esac; }
            healthy() { return 1; }
        """))
        (self.root / "scripts" / "build_extensions.sh").write_text(
            "#!/usr/bin/env bash\nprintf '%s\\n' build >> \"$TEST_OPERATIONS\"\n")
        (self.root / "scripts" / "build_extensions.sh").chmod(0o755)
        (self.bin / "docker").write_text(textwrap.dedent("""\
            #!%s
            import json
            import os
            from pathlib import Path
            import sys

            args = sys.argv[1:]
            with Path(os.environ["DOCKER_COMMANDS"]).open("a") as output:
                output.write(json.dumps(args) + "\\n")
            if args[:2] == ["image", "inspect"] and any("tensorfold.memory-topk" in a for a in args):
                print("1")
                sys.exit(0)
            if args[:1] == ["run"]:
                sys.exit(37)
        """ % sys.executable))
        (self.bin / "docker").chmod(0o755)

    def tearDown(self):
        self.tmp.cleanup()

    def write_env(self, **settings):
        values = {
            "HEAD_IP": "127.0.0.1",
            "WORKER_IP": "127.0.0.2",
            "WORKER_SSH": "fixture-worker",
            "RDMA_DEVICES": "fixture-rdma",
            "NCCL_IB_HCA": "fixture-hca",
            "NCCL_SOCKET_IFNAME": "fixture-nic",
            "GID_INDEX": "3",
            "MODEL_VIEW": str(self.root / "model-view"),
            "DOWNLOAD_DIR": str(self.root / "downloads"),
            "CACHE_DIR": str(self.root / "cache"),
            "IMAGE": "fixture:image",
            "BIND_HOST": "127.0.0.1",
            "PORT": "8888",
            "SERVED_NAME": "fixture-model",
            "SERVED_ALIAS": "",
            "API_KEY_FILE": "",
            "CONTEXT": "1024",
            "PARALLEL": "1",
            "PREFILL_CHUNK": "512",
            "MTP_DRAFTS": "1",
            "SPILL_GIB": "0",
            "MEM_FLOOR_GIB": "1",
            "CGROUP_PARENT": "",
            "TEMPERATURE": "1",
            "TOP_P": "1",
            "VISION": "0",
            "RESTART_POLICY": "no",
        } | {key: str(value) for key, value in settings.items()}
        (self.root / ".env").write_text("".join(
            f"{key}={shlex.quote(value)}\n" for key, value in values.items()))

    def invoke(self, **settings):
        self.write_env(**settings)
        operations = self.root / "operations.log"
        self.commands.unlink(missing_ok=True)
        operations.unlink(missing_ok=True)
        env = os.environ.copy()
        env.pop("THINKING", None)
        env.pop("REASONING_EFFORT", None)
        env |= {
            "PATH": f"{self.bin}:{os.environ['PATH']}",
            "PREFLIGHT": "0",
            "DOCKER_COMMANDS": str(self.commands),
            "TEST_OPERATIONS": str(operations),
        }
        result = subprocess.run(["bash", str(self.root / "start.sh")], cwd=self.root,
                                text=True, capture_output=True, env=env, check=False)
        return result, operations

    def docker_commands(self):
        if not self.commands.exists():
            return []
        return [json.loads(line) for line in self.commands.read_text().splitlines()]

    def assert_docker_settings(self, thinking, effort):
        result, operations = self.invoke(**thinking, REASONING_EFFORT=effort)
        commands = self.docker_commands()
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual([command[0] for command in commands], ["rm", "run"])
        args = commands[-1]
        self.assertIn(thinking["THINKING"] == "1" and "--thinking" or "--no-thinking", args)
        self.assertEqual(args[args.index("--reasoning-effort") + 1], effort)
        self.assertEqual(operations.read_text(), "wssh\nbuild\n")

    def test_missing_settings_default_to_thinking_off(self):
        result, operations = self.invoke()
        commands = self.docker_commands()
        self.assertEqual(result.returncode, 37, result.stderr)
        self.assertEqual([command[0] for command in commands], ["rm", "run"])
        args = commands[-1]
        self.assertIn("--no-thinking", args)
        self.assertNotIn("--reasoning-effort", args)
        self.assertEqual(operations.read_text(), "wssh\nbuild\n")

    def test_explicit_zero_emits_no_thinking(self):
        self.assert_docker_settings({"THINKING": "0"}, "high")

    def test_allocator_cap_switch_reaches_docker(self):
        for value in ("0", "1"):
            result, _ = self.invoke(MEM_CAP=value)
            self.assertEqual(result.returncode, 37, result.stderr)
            args = self.docker_commands()[-1]
            if value == "1":
                self.assertIn("TF_DS_MEM_CAP=1", args)
            else:
                self.assertNotIn("TF_DS_MEM_CAP=1", args)

    def test_invalid_allocator_cap_stops_before_operations(self):
        result, operations = self.invoke(MEM_CAP="invalid")
        self.assertEqual(result.returncode, 1)
        self.assertFalse(self.commands.exists())
        self.assertFalse(operations.exists())

    def test_each_valid_reasoning_effort_reaches_docker_unchanged(self):
        for effort in ("low", "medium", "high", "xhigh"):
            with self.subTest(effort=effort):
                self.commands.unlink(missing_ok=True)
                self.assert_docker_settings({"THINKING": "1"}, effort)

    def test_invalid_settings_stop_before_preflight_build_or_docker(self):
        for settings in ({"THINKING": "2"}, {"REASONING_EFFORT": "max"}):
            with self.subTest(settings=settings):
                self.commands.unlink(missing_ok=True)
                result, operations = self.invoke(**settings)
                self.assertEqual(result.returncode, 1)
                self.assertFalse(self.commands.exists())
                self.assertFalse(operations.exists())

    def test_quoted_reasoning_effort_injection_is_rejected_before_operations(self):
        result, operations = self.invoke(REASONING_EFFORT="high; touch should-not-exist")
        self.assertEqual(result.returncode, 1)
        self.assertIn("REASONING_EFFORT must be", result.stderr)
        self.assertFalse(self.commands.exists())
        self.assertFalse(operations.exists())
        self.assertFalse((self.root / "should-not-exist").exists())


class DisplayBundleThinkingSettingsTest(unittest.TestCase):
    def test_prepare_snapshot_keeps_dynamic_thinking_settings(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "recipe"
            stage = root / "experiments" / "display-kv"
            stage.mkdir(parents=True)
            shutil.copy2(RECIPE / "start.sh", root / "start.sh")
            shutil.copytree(RECIPE / "scripts", root / "scripts")
            for name in ("stop.sh", "status.sh", "verify.sh", "AGENTS.md"):
                shutil.copy2(RECIPE / name, root / name)
            shutil.copy2(RECIPE / "experiments" / "display-kv" / "apply.py", stage / "apply.py")
            prepare_path = stage / "prepare.py"
            shutil.copy2(RECIPE / "experiments" / "display-kv" / "prepare.py", prepare_path)
            (root / ".env").write_text(textwrap.dedent("""\
                IMAGE=base:image
                SPILL_GIB=64
                PREFILL_CHUNK=2048
                THINKING=0
                REASONING_EFFORT=low
            """))

            spec = importlib.util.spec_from_file_location("fixture_prepare", prepare_path)
            module = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(module)
            module.ROOT = root
            bundle = root / "bundle"
            module.prepare(bundle, "fixture:display-kv")

            start = (bundle / "start.sh").read_text()
            config = dict(line.split("=", 1) for line in (bundle / ".env").read_text().splitlines()
                          if line and not line.startswith("#"))
            self.assertIn('$( [ "${THINKING:-0}" = 1 ] && echo --thinking || echo --no-thinking )', start)
            self.assertIn('$( [ -n "${REASONING_EFFORT:-}" ] && echo --reasoning-effort $REASONING_EFFORT )', start)
            self.assertNotIn('--no-thinking --reasoning-effort low', start)
            self.assertEqual(config["IMAGE"], "fixture:display-kv")
            self.assertEqual(config["SPILL_GIB"], "64")
            self.assertEqual(config["PREFILL_CHUNK"], "512")
            self.assertEqual(config["THINKING"], "0")
            self.assertEqual(config["REASONING_EFFORT"], "low")


if __name__ == "__main__":
    unittest.main()
