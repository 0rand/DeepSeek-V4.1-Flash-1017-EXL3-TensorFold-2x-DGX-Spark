"""Offline scan for machine-specific values in tracked files (stdlib only; no server, no GPU).

CONTRIBUTING.md ("No secrets and no machine-specific values") forbids home paths and public IP addresses in the repo.
Use a placeholder instead: $HOME, ~, <user>, <head>, or the example addresses of .env.example (10.100.80.x).
gitleaks (CI job "secrets") checks for keys and tokens; this test checks the values that gitleaks does not know.

Run: python3 -m unittest discover -s tests -p 'test_*.py'   (or tests/run_offline.sh)
"""
import ipaddress
import re
import shutil
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Home folders that are placeholders or service accounts, not a person's name. "elevatormusic" is the maintainer's
# public GitHub name: two saved runs of 18 October show it, and old runs are records that stay unchanged.
ALLOWED_HOMES = {"user", "ubuntu", "ci", "runner", "username", "you", "your-user", "me", "nvidia", "elevatormusic"}
HOME_PATH = re.compile(r"(?:/home/|/Users/|[A-Za-z]:\\Users\\)([A-Za-z0-9._-]+)")
IPV4 = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?![\w.])")
# Public addresses that may appear: well-known resolvers in network docs.
ALLOWED_IPS = {"1.1.1.1", "8.8.8.8", "8.8.4.4"}


def tracked_text_files():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    for name in out.decode().split("\0"):
        if not name:
            continue
        path = ROOT / name
        try:
            yield name, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                                          # binary or unreadable: not text


def public_ip(text: str) -> bool:
    try:
        ip = ipaddress.IPv4Address(text)
    except ValueError:
        return False                                          # for example a version such as 1.2.300.4
    return ip.is_global and text not in ALLOWED_IPS


@unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a git checkout")
class MachineValues(unittest.TestCase):
    def test_no_personal_home_paths(self):
        hits = []
        for name, text in tracked_text_files():
            for m in HOME_PATH.finditer(text):
                if m.group(1).lower() not in ALLOWED_HOMES and not m.group(1).startswith("<"):
                    line = text.count("\n", 0, m.start()) + 1
                    hits.append(f"{name}:{line}: {m.group(0)}")
        self.assertEqual(hits, [], "home paths with a user name; use $HOME, ~ or <user>:\n" + "\n".join(hits[:40]))

    def test_no_public_ip_addresses(self):
        hits = []
        for name, text in tracked_text_files():
            for m in IPV4.finditer(text):
                if text[max(0, m.start() - 2):m.start()] in ("==", "~=", ">=", "<="):
                    continue                                  # a pinned version such as pkg==1.7.12.25
                if public_ip(m.group(1)):
                    line = text.count("\n", 0, m.start()) + 1
                    hits.append(f"{name}:{line}: {m.group(1)}")
        self.assertEqual(hits, [], "public IP addresses; use the 10.100.80.x examples or <head>:\n"
                         + "\n".join(hits[:40]))


if __name__ == "__main__":
    unittest.main()
