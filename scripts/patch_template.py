#!/usr/bin/env python3
"""Build the patched chat template for the model view.

The template comes from the MiaAI-Lab vLLM kit. The kit is AGPL-3.0, so this repo does
not hold the template. This script does these steps:
  1. It gets the kit template at a pinned commit (a download, or a local file with --from).
  2. It checks the sha256 and the size of that file.
  3. It applies our patch with exact string replacements. The patch is the one hunk of
     runs/2026-10-17/template-patch.diff. The replacement must match exactly one place.
  4. It checks the sha256 and the size of the result, and then it writes the result.

The script stops with exit code 1 on any mismatch. In that case it writes nothing.
It uses the Python standard library only.
"""

import argparse
import hashlib
import os
import re
import shlex
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

# --- Pinned input: the kit template ---
KIT_REPO = "MiaAI-Lab/DeepSeek-v4.1-Flash-EXL3-2x-DGX-Sparks"
KIT_COMMIT = "6f7d1590ad49a2b8995188e45d7b9db31e677452"
KIT_FILE = "files/chat_template.jinja"
KIT_URL = "https://raw.githubusercontent.com/%s/%s/%s" % (KIT_REPO, KIT_COMMIT, KIT_FILE)
KIT_SHA256 = "7d1356675e429ec4a55cbcc1a6fbeb40615dbeb257aca25b8c3705a42087f33e"
KIT_SIZE = 11410

# --- Pinned output: the patched template that the 2026-10-17 runs served ---
OUT_SHA256 = "defbdfc2a28ea921d9ad4c6be7244fa82c53f6d552507527a944ad4cbdd24a6d"
OUT_SIZE = 11780

MAX_DOWNLOAD = 1000000  # bytes. The kit template has 11,410 bytes. A larger file is wrong.

# The hunk of runs/2026-10-17/template-patch.diff. Each entry is (mark, line):
# " " is a context line, "-" is a line that the patch removes, "+" is a line that it adds.
# The old text is the context and the removed lines. The new text is the context and the added lines.
HUNK = [
    (" ", ""),
    (" ", r'    {%- elif role == "user" -%}'),
    (" ", r'        {#- Tools with no leading system message still have to reach the model -#}'),
    ("-", r'        {%- if idx == 0 and tools is defined and tools -%}'),
    ("+", r'        {%- set has_rf = response_format is defined and response_format -%}'),
    ("+", r'        {%- if idx == 0 and ((tools is defined and tools) or has_rf) -%}'),
    (" ", r'            {%- if not thinking -%}{%- set ns.out = ns.out + sys_tok -%}{%- endif -%}'),
    ("-", r'            {%- set ns.out = ns.out + tools_tmpl -%}'),
    ("+", r'            {%- if tools is defined and tools -%}{%- set ns.out = ns.out + "\n\n" + tools_tmpl -%}{%- endif -%}'),
    ("+", r'            {%- if has_rf -%}'),
    ("+", r'                {%- set ns.out = ns.out + "\n\n## Response Format:\n\nYou MUST strictly adhere to the following schema to reply:\n" + (response_format | tojson) -%}'),
    ("+", r'            {%- endif -%}'),
    (" ", r'        {%- endif -%}'),
    (" ", r'        {#- More than one tool result in a turn is emitted in call order, not arrival order -#}'),
    (" ", r'        {%- set tool_parts = m.parts | selectattr("tool") | list -%}'),
]


def say(message):
    print("[patch_template] " + message)


def die(message):
    print("[patch_template] ERROR: " + message, file=sys.stderr)
    sys.exit(1)


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def check(label, data, want_sha, want_size):
    """Stop with an error when the hash or the size of data is wrong."""
    got_sha = sha256(data)
    if got_sha != want_sha or len(data) != want_size:
        die(
            "%s does not match.\n"
            "  expected: sha256 %s, %d bytes\n"
            "  found:    sha256 %s, %d bytes" % (label, want_sha, want_size, got_sha, len(data))
        )
    say("%s: sha256 %s, %d bytes. OK." % (label, got_sha, len(data)))


def hunk_texts():
    """Return (old, new) as bytes. Both come from HUNK."""
    old = "".join(text + "\n" for mark, text in HUNK if mark in " -")
    new = "".join(text + "\n" for mark, text in HUNK if mark in " +")
    return old.encode("utf-8"), new.encode("utf-8")


def apply_patch(data):
    """Replace the old text with the new text. The old text must occur exactly once."""
    old, new = hunk_texts()
    count = data.count(old)
    if count != 1:
        die("the patch context occurs %d times in the kit template. It must occur once." % count)
    return data.replace(old, new)


def read_kit(local_file):
    """Return the kit template as bytes, from a local file or from GitHub."""
    if local_file:
        say("read the kit template from %s" % local_file)
        try:
            return Path(local_file).read_bytes()
        except OSError as exc:
            die("cannot read %s: %s" % (local_file, exc))
    say("download %s" % KIT_URL)
    request = urllib.request.Request(KIT_URL, headers={"User-Agent": "tfv41-patch-template"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response:
            data = response.read(MAX_DOWNLOAD + 1)
    except (urllib.error.URLError, OSError) as exc:
        die(
            "cannot download the kit template: %s\n"
            "  Save the file by hand from that URL, then run this script with --from FILE." % exc
        )
    if len(data) > MAX_DOWNLOAD:
        die("the download is larger than %d bytes. This is not the kit template." % MAX_DOWNLOAD)
    return data


# --- Settings: MODEL_VIEW and WORKER_SSH, from the environment or from the .env file of the repo ---
def expand(value, known):
    """Expand $NAME, ${NAME} and a leading ~ the way the shell does for the .env file."""

    def replace(match):
        name = match.group(1) or match.group(2)
        return known.get(name, os.environ.get(name, match.group(0)))

    return os.path.expanduser(re.sub(r"\$(?:(\w+)|\{(\w+)\})", replace, value))


def read_env_file(path):
    """Read KEY=VALUE lines of a simple .env file. Return a dict. Return {} when the file is missing."""
    values = {}
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return values
    for line in text.splitlines():
        match = re.match(r"\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$", line)
        if not match:
            continue
        key, raw = match.group(1), match.group(2).strip()
        if raw[:1] in ("'", '"'):
            end = raw.find(raw[0], 1)
            value = raw[1:end] if end > 0 else raw[1:]
            values[key] = value if raw[0] == "'" else expand(value, values)
        else:
            values[key] = expand(re.split(r"\s+#", raw, maxsplit=1)[0].strip(), values)
    return values


def setting(name, env_file_values):
    """Return a setting from the environment first, then from the .env file, else ''."""
    return os.environ.get(name) or env_file_values.get(name, "")


# --- Writing ---
def write_atomic(path, data):
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(str(tmp), str(path))


def keep_old_file(path):
    """Keep a file that is neither the kit original nor our result, so no custom edit is lost."""
    try:
        old = path.read_bytes()
    except OSError:
        return
    old_sha = sha256(old)
    if old_sha in (KIT_SHA256, OUT_SHA256):
        return
    backup = path.with_name("%s.bak-%s" % (path.name, old_sha[:8]))
    write_atomic(backup, old)
    say("the old file differs from both known versions. A copy is in %s" % backup)


def copy_to_worker(host, local, remote_path):
    """Copy the file to the same path on the worker with ssh, and check its hash there."""
    tmp = shlex.quote(remote_path + ".tmp")
    dst = shlex.quote(remote_path)
    command = "cat > %s && mv %s %s && sha256sum < %s" % (tmp, tmp, dst, dst)
    say("copy to the worker %s:%s" % (host, remote_path))
    with open(str(local), "rb") as handle:
        try:
            done = subprocess.run(
                ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10", host, command],
                stdin=handle,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=120,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            die("ssh to the worker failed: %s" % exc)
    if done.returncode != 0:
        die(
            "the copy to the worker failed (ssh exit code %d):\n  %s\n"
            "  If the folder is missing on the worker, run scripts/download.sh worker first."
            % (done.returncode, done.stderr.decode("utf-8", "replace").strip())
        )
    fields = done.stdout.decode("utf-8", "replace").split()
    if not fields or fields[0] != OUT_SHA256:
        die("the file on the worker has the wrong sha256: %s" % (fields[0] if fields else "no output"))
    say("worker: sha256 %s. OK." % fields[0])


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Download the MiaAI-Lab kit chat template at a pinned commit, apply our patch,"
        " check both sha256 values, and write the result.",
        epilog="MODEL_VIEW and WORKER_SSH come from the environment, or else from the .env file of this repo.",
    )
    parser.add_argument(
        "--out",
        metavar="PATH",
        help="write the result here (default: $MODEL_VIEW/model/chat_template.jinja)",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help="also copy the result to the same path on the worker (WORKER_SSH) and check its hash there",
    )
    parser.add_argument(
        "--from",
        dest="from_file",
        metavar="FILE",
        help="use this local copy of the kit template, not a download",
    )
    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parent.parent
    env_file = read_env_file(repo_root / ".env")
    view = setting("MODEL_VIEW", env_file)
    worker = setting("WORKER_SSH", env_file)

    if args.out:
        out = Path(os.path.expanduser(os.path.expandvars(args.out))).absolute()
    elif view:
        out = Path(os.path.expanduser(view)).absolute() / "model" / "chat_template.jinja"
        if not out.parent.is_dir():
            die("the folder %s does not exist. Run scripts/download.sh view first, or use --out." % out.parent)
    else:
        die("MODEL_VIEW is not set. Set it in the environment or in .env, or use --out PATH.")
    if args.worker and not worker:
        die("WORKER_SSH is not set. Set it in the environment or in .env.")

    existing = out.read_bytes() if out.is_file() else b""
    if existing and sha256(existing) == OUT_SHA256 and not args.from_file:
        say("%s has the patched template already. OK." % out)
    else:
        kit = read_kit(args.from_file)
        check("kit template", kit, KIT_SHA256, KIT_SIZE)
        patched = apply_patch(kit)
        check("patched template", patched, OUT_SHA256, OUT_SIZE)
        out.parent.mkdir(parents=True, exist_ok=True)
        if existing:
            keep_old_file(out)
        write_atomic(out, patched)
        say("wrote %s" % out)

    if args.worker:
        copy_to_worker(worker, out, str(out))
    elif not args.out:
        say("The worker needs the same file. Run this script again with --worker, or run scripts/download.sh worker.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
