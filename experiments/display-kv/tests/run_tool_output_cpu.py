#!/usr/bin/env python3
"""Install the real recipe into temporary pinned Git checkouts; CPU-only tests.

Accept --source or TF_DISPLAY_SOURCE. No downloads, installs, native extensions,
model loading or serving calls. Only the fake HTTP tests bind ephemeral loopback.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("display_apply", ROOT / "apply.py")
if spec is None or spec.loader is None:
    raise RuntimeError("installer is not loadable")
apply = importlib.util.module_from_spec(spec)
spec.loader.exec_module(apply)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def snapshot(checkout):
    return {str(path.relative_to(checkout)): digest(path)
            for path in (checkout / "src").rglob("*") if path.is_file()}


def run(command, **kwargs):
    print("+", " ".join(map(str, command)), flush=True)
    return subprocess.run(list(map(str, command)), check=True, **kwargs)


def reject(command, checkout, expected, env):
    before = snapshot(checkout)
    result = subprocess.run(list(map(str, command)), env=env, text=True, capture_output=True)
    if result.returncode == 0 or expected not in result.stderr:
        raise RuntimeError(f"expected refusal {expected!r}: {result.returncode}\n{result.stderr}")
    if snapshot(checkout) != before:
        raise RuntimeError("rejected installer wrote source files")
    print(f"PASS refusal without writes: {expected}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=os.environ.get("TF_DISPLAY_SOURCE"),
                        help="pristine local TensorFold Git checkout at the installer pin")
    args = parser.parse_args()
    if args.source is None:
        parser.error("provide --source or TF_DISPLAY_SOURCE (a pristine pinned Git checkout)")
    source = args.source.resolve()
    if apply._git_head(source) != apply.REVISION:
        parser.error(f"source HEAD must be {apply.REVISION}")
    for relative, expected in apply.FILES.items():
        if digest(source / relative) != expected:
            parser.error(f"source fingerprint differs: {relative}")
    original = snapshot(source)
    env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONOPTIMIZE", None)
    with tempfile.TemporaryDirectory(prefix="tensorfold-tool-output-") as directory:
        temp = Path(directory)
        # Match the Dockerfile COPY layout: no dependency on checkout metadata,
        # repair bundles, tests, or any paths outside this public recipe.
        stage = temp / "stage"
        stage.mkdir()
        shutil.copyfile(ROOT / "apply.py", stage / "apply.py")
        shutil.copytree(ROOT / "overlay", stage / "overlay")
        for mode, flags in (("normal", []), ("optimized", ["-O"])):
            checkout = temp / mode
            run(["git", "clone", "--shared", "--no-checkout", source, checkout])
            run(["git", "-C", checkout, "checkout", "--detach", apply.REVISION])
            command = [sys.executable, "-B", *flags, stage / "apply.py", checkout]
            pristine = {relative: (checkout / relative).read_text() for relative in apply.FILES}
            # Deliberately corrupt only this temporary checkout's HEAD to an
            # existing non-commit object. This works for shallow input clones
            # without downloading history or manufacturing a source commit.
            head_path = checkout / ".git/HEAD"
            saved_head = head_path.read_text()
            tree = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD^{tree}"], text=True).strip()
            head_path.write_text(tree + "\n")
            reject(command, checkout, "checkout HEAD must be", env)
            head_path.write_text(saved_head)
            drift = checkout / "src/tensorfold/cuda/server.py"
            drift.write_text(pristine[drift.relative_to(checkout)] + "\n# deliberate drift\n")
            reject(command, checkout, "source fingerprint differs", env)
            drift.write_text(pristine[drift.relative_to(checkout)])
            run(command, env=env)
            for relative, text in pristine.items():
                installed = (checkout / relative).read_text()
                if installed != apply.transform(relative, text):
                    raise RuntimeError(f"installed transform differs: {relative}")
                ast.parse(installed, filename=str(relative))
            guard = checkout / "src/tensorfold/cuda/tool_output.py"
            if guard.read_bytes() != (stage / "overlay/tool_output.py").read_bytes():
                raise RuntimeError("installed guard differs from Docker-stage overlay")
            reject(command, checkout, "already installed", env)
            if subprocess.run(["git", "-C", str(checkout), "diff", "--quiet"]).returncode != 1:
                raise RuntimeError("installed source must be visibly modified, not represented as pristine")
            print(f"PASS {mode}: real staged install; installed source is modified; "
                  f"guard sha256={digest(guard)}", flush=True)
            test_env = {**env, "TF_DISPLAY_SOURCE": str(checkout), "TF_TOOL_OUTPUT_INSTALLED": "1"}
            run([sys.executable, "-B", *flags, "-m", "unittest", "discover", "-s", ROOT / "tests",
                 "-p", "test_tool_output*.py", "-v"], env=test_env)
    if snapshot(source) != original:
        raise RuntimeError("read-only input source changed")
    print("PASS normal and -O: staged installer, refusal gates, parser and fake HTTP; input unchanged", flush=True)


if __name__ == "__main__":
    main()
