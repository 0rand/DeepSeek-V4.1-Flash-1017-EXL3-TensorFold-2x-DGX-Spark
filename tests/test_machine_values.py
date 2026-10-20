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

# Home folders that are placeholders or service accounts, not a person's name.
ALLOWED_HOMES = {"user", "ubuntu", "ci", "runner", "username", "you", "your-user", "me", "nvidia"}
# Old runs are records that stay unchanged. Only these files may keep the home path that they show.
ALLOWED_HOME_IN_FILE = {
    ("runs/2026-10-18/gate/suite-ht.log", "elevatormusic"),
    ("runs/2026-10-18/gate/suite-odirect2.log", "elevatormusic"),
}
# /home/<name>, /Users/<name> and C:\Users\<name>. Windows paths match in any letter case, also with the doubled
# backslashes of JSON. The POSIX forms stay case-sensitive, so a URL path such as /users/123 does not match.
# ~<name> is the home of a named user, also with no slash after it (cd ~alice, HOME=~alice).
# A plain ~ or ~/ is allowed: it names no user.
HOME_PATH = re.compile(
    r"(?:/home/|/Users/|[A-Za-z]:(?:\\\\|\\|/)(?i:users)(?:\\\\|\\|/)|(?<![\w/~])~(?=[A-Za-z]))"
    r"([A-Za-z0-9._-]+)"
)
IPV4 = re.compile(r"(?<![\w.])(\d{1,3}(?:\.\d{1,3}){3})(?![\w.])")
# An IPv6 candidate: hex groups with 2 to 7 colons. ipaddress decides if it is a valid address.
IPV6 = re.compile(r"(?<![\w:.])([0-9A-Fa-f]{0,4}(?::[0-9A-Fa-f]{0,4}){2,7})(?![\w:])")
# Public addresses that may appear: well-known resolvers in network docs.
ALLOWED_IPS = {"1.1.1.1", "8.8.8.8", "8.8.4.4", "2606:4700:4700::1111", "2001:4860:4860::8888"}


def tracked_text_files():
    out = subprocess.run(["git", "-C", str(ROOT), "ls-files", "-z"], capture_output=True, check=True).stdout
    for name in out.decode().split("\0"):
        if not name or name == "tests/test_machine_values.py":   # this file holds sample values on purpose
            continue
        path = ROOT / name
        try:
            yield name, path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue                                          # binary or unreadable: not text


def public_ip(text: str) -> bool:
    try:
        ip = ipaddress.ip_address(text)
    except ValueError:
        return False                                          # a version such as 1.2.300.4, or a time such as 12:30:00
    # Multicast addresses (224.0.0.1, ff02::1) are protocol constants, not a machine's address.
    return ip.is_global and not ip.is_multicast and str(ip) not in ALLOWED_IPS


def ip_candidates(text: str):
    for m in IPV4.finditer(text):
        if text[max(0, m.start() - 2):m.start()] in ("==", "~=", ">=", "<="):
            continue                                          # a pinned version such as pkg==1.7.12.25
        yield m
    yield from IPV6.finditer(text)


@unittest.skipUnless(shutil.which("git") and (ROOT / ".git").exists(), "needs a git checkout")
class MachineValues(unittest.TestCase):
    def test_no_personal_home_paths(self):
        hits = []
        for name, text in tracked_text_files():
            for m in HOME_PATH.finditer(text):
                user = m.group(1)
                if user.lower() in ALLOWED_HOMES or user.startswith("<") or (name, user.lower()) in ALLOWED_HOME_IN_FILE:
                    continue
                line = text.count("\n", 0, m.start()) + 1
                hits.append(f"{name}:{line}: {m.group(0)}")
        self.assertEqual(hits, [], "home paths with a user name; use $HOME, ~ or <user>:\n" + "\n".join(hits[:40]))

    def test_no_public_ip_addresses(self):
        hits = []
        for name, text in tracked_text_files():
            for m in ip_candidates(text):
                if public_ip(m.group(1)):
                    line = text.count("\n", 0, m.start()) + 1
                    hits.append(f"{name}:{line}: {m.group(1)}")
        self.assertEqual(hits, [], "public IP addresses; use the 10.100.80.x examples or <head>:\n"
                         + "\n".join(hits[:40]))


class Detection(unittest.TestCase):
    """The scan finds what it must find, and only that."""

    def found(self, text):
        return [m.group(1) for m in ip_candidates(text) if public_ip(m.group(1))]

    def test_public_addresses_found(self):
        self.assertEqual(self.found("head 203.0.113.7 ok"), [])          # documentation range: not global
        self.assertEqual(self.found("peer 34.117.59.81 up"), ["34.117.59.81"])
        self.assertEqual(self.found("addr 2a01:4f8:c17:1234::1 up"), ["2a01:4f8:c17:1234::1"])

    def test_not_addresses(self):
        for text in ("10.100.80.1", "fe80::1", "::1", "12:30:45", "2026-10-18T21:01:32Z", "aa:bb:cc:dd:ee:ff",
                     "std::vector", "ruff==0.17.1", "pkg==1.7.12.25", "driver 580.178.04", "fd00::5",
                     "224.0.0.1", "239.255.255.250", "ff02::1", "ff05::1:3"):
            with self.subTest(text=text):
                self.assertEqual(self.found(text), [])

    def test_home_paths(self):
        for text, user in (("/home/alice/x", "alice"), ("/Users/Bob/x", "Bob"), (r"C:\Users\Carol\run.log", "Carol"),
                           (r"c:\users\dave\run.log", "dave"), (r'"C:\\Users\\Erin\\run.log"', "Erin"),
                           ("C:/Users/Fay/run.log", "Fay"), ("D:\\USERS\\Gus\\x", "Gus"),
                           ("~hal/run.log", "hal"), ("cd ~ivy/deploy", "ivy"), ("cd ~jo", "jo"),
                           ("HOME=~kim", "kim"), ('"~lee"', "lee")):
            with self.subTest(text=text):
                self.assertEqual([m.group(1) for m in HOME_PATH.finditer(text)], [user])
        for text in ("GET https://api.example.com/users/123", "~/deploy/x", "about ~15 min", "a~b/c", "~~x/y", "cd ~ && ls"):
            with self.subTest(text=text):
                self.assertEqual(list(HOME_PATH.finditer(text)), [])


if __name__ == "__main__":
    unittest.main()
