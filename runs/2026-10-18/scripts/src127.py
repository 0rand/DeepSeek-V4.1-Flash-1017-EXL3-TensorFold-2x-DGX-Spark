"""Run a script with every HTTP connection bound to source 127.0.0.1 (the 172.17.0.1:8888 guard admits only loopback)."""
import http.client
import runpy
import sys

_orig = http.client.HTTPConnection.__init__


def _init(self, *a, **k):
    k.setdefault("source_address", ("127.0.0.1", 0))
    _orig(self, *a, **k)


http.client.HTTPConnection.__init__ = _init
path = sys.argv[1]
sys.argv = sys.argv[1:]
runpy.run_path(path, run_name="__main__")
