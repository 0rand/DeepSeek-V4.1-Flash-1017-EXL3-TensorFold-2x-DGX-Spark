"""Isolated stdlib test doubles. No installed TensorFold package is required."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def guard_module():
    return load("tool_guard_fixture", ROOT / "overlay/tool_output.py")


@contextmanager
def fixtures():
    parser = load("parser_fixture", ROOT / "tests/fixtures/parser.py")
    guard = guard_module()
    names = ["tensorfold", "tensorfold.cuda", "tensorfold.server", "tensorfold.cuda.health",
             "tensorfold.server.anthropic", "tensorfold.server.metrics", "tensorfold.server.responses",
             "tensorfold.server.token_routes", "tensorfold.server.cancellation", "tensorfold.server.decisions",
             "tensorfold.server.errors", "tensorfold.server.http", "tensorfold.server.request_body",
             "tensorfold.server.stacks", "tensorfold.server.auth_http"]
    modules = {name: ModuleType(name) for name in names}
    modules["tensorfold.cuda.reply_text"] = parser
    modules["tensorfold.server.tools"] = parser
    modules["tensorfold.cuda.tool_output"] = guard
    for name, module in modules.items():
        if "." in name:
            parent, attr = name.rsplit(".", 1)
            setattr(modules[parent], attr, module)
    class RequestError(Exception):
        pass
    class CapacityError(RequestError):
        pass
    class RequestCancelled(Exception):
        pass
    errors = modules["tensorfold.server.errors"]
    errors.RequestError, errors.CapacityError = RequestError, CapacityError
    errors.error_body = lambda exc, field=None: {"message": str(exc)}
    modules["tensorfold.server.cancellation"].RequestCancelled = RequestCancelled
    modules["tensorfold.server.cancellation"].socket_cancellation = lambda connection: SimpleNamespace(cancelled=False)
    modules["tensorfold.server.decisions"].DecisionError = RequestError
    modules["tensorfold.server.http"].Server = ThreadingHTTPServer
    modules["tensorfold.server.http"].wants_usage_chunk = lambda body: False
    modules["tensorfold.server.request_body"].read_body = lambda handler: handler.rfile.read(int(handler.headers["Content-Length"]))
    modules["tensorfold.server.stacks"].Rearming = BaseHTTPRequestHandler
    modules["tensorfold.server.auth_http"].handler = lambda handler, app: handler
    modules["tensorfold.server.anthropic"].route = lambda path: False
    modules["tensorfold.server.responses"].route = lambda path: None
    modules["tensorfold.server.token_routes"].ROUTES = ()
    with patch.dict(sys.modules, modules), patch.dict("os.environ", {"TF_API_KEY": "", "TF_API_KEY_FILE": ""}):
        yield guard
