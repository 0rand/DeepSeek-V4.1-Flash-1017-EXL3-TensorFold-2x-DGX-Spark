"""Checks of the spill transform on the pinned source. Set TF_SPILL_SOURCE to a pristine pinned checkout.

The spill transform runs after the display transform, so each check applies both to the pristine file.
"""
from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import types
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("TF_SPILL_SOURCE", ""))


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


display = load(ROOT.parent / "display-kv" / "apply.py", "display_apply_for_spill_integration")
spill = load(ROOT / "apply.py", "spill_apply_for_spill_integration")


@unittest.skipUnless(os.environ.get("TF_SPILL_SOURCE"), "set TF_SPILL_SOURCE to a pristine pinned checkout")
class SpillIntegrationTests(unittest.TestCase):
    def changed(self, relative) -> str:
        relative = Path(relative)
        text = (SOURCE / relative).read_text()
        if relative in display.FILES:
            text = display.transform(relative, text)
        return spill.transform(relative, text)

    def method_of(self, tree, class_name, *names):
        """The named methods of ``class_name``, without decorators and annotations."""
        owner = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        methods = [node for node in owner.body if isinstance(node, ast.FunctionDef) and node.name in names]
        self.assertEqual({method.name for method in methods}, set(names))
        for method in methods:
            method.decorator_list, method.returns = [], None
            for arg in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
                arg.annotation = None
        return methods

    def test_shutdown_drain_hook_is_installed(self):
        changed = self.changed(spill.APP)
        self.assertIn("self.on_exit = self._flush_spill", changed)
        self.assertIn("spill.shutdown_drain()", changed)
        changed = self.changed(spill.HTTP)
        self.assertIn('hook = getattr(app, "on_exit", None)', changed)

    def test_the_composed_engine_accepts_a_nonzero_spill_size(self):
        guard = "display-only KV requires TF_DS_SPILL_GIB=0"
        pristine = (SOURCE / spill.ENGINE).read_text()
        displayed = display.transform(spill.ENGINE, pristine)
        self.assertIn(guard, displayed)                      # the display-only image keeps its guard
        composed = spill.transform(spill.ENGINE, displayed)
        self.assertNotIn(guard, composed)
        self.assertIn("display KV requires --parallel greater than 1 before weights load", composed)
        self.assertIn("display KV requires TF_DS_KV=native packed storage", composed)
        ast.parse(composed)

    def test_the_spill_config_has_no_highwater_and_ids_are_not_copied_to_a_list(self):
        changed = self.changed(spill.MULTI)
        self.assertNotIn("highwater", changed)
        self.assertNotIn("tokens=list(k.host.view())", changed)

    def test_shutdown_drain_waits_for_queued_writes(self):
        tree = ast.parse(self.changed(spill.APP))
        [method] = self.method_of(tree, "DsApp", "_flush_spill")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "_flush_spill", "exec"),
             namespace)

        class Spill:
            def __init__(self):
                self.calls = 0

            def shutdown_drain(self):
                self.calls += 1
                return True

        class App:
            _flush_spill = namespace["_flush_spill"]

        app = App()
        app.engine = types.SimpleNamespace(multi=types.SimpleNamespace(spill=Spill()))
        app._flush_spill()
        self.assertEqual(app.engine.multi.spill.calls, 1)
        app.engine.multi = None
        app._flush_spill()

    def test_rank_zero_gets_its_sigterm_handler_from_the_pinned_server(self):
        """The drain hook of rank 0 runs in the KeyboardInterrupt branch, so SIGTERM must raise it. The pinned
        revision does that in serve(); if a new pin drops it, this test fails and the hook needs its own handler."""
        pinned = (SOURCE / spill.HTTP).read_text()
        self.assertIn("signal.signal(signal.SIGTERM, _terminate)", pinned)
        self.assertIn("raise KeyboardInterrupt", pinned)
        changed = self.changed(spill.HTTP)
        self.assertLess(changed.index("signal.signal(signal.SIGTERM"), changed.index('getattr(app, "on_exit"'))

    def follower(self, drain):
        tree = ast.parse(self.changed(spill.MULTI))
        methods = self.method_of(tree, "MultiDecoder", "follow", "_follow_steps")
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=methods, type_ignores=[])), "follow", "exec"), namespace)

        class Decoder:
            follow = namespace["follow"]
            _follow_steps = namespace["_follow_steps"]

        decoder = Decoder()
        decoder.spill = drain
        return decoder

    def test_follower_rank_drains_its_spill_writes_on_sigterm(self):
        import signal
        import threading

        if os.name == "nt":
            self.skipTest("needs POSIX signals")

        class Spill:
            drained = 0

            def shutdown_drain(self):
                Spill.drained += 1

        decoder = self.follower(Spill())

        class Link:
            def receive(self):
                threading.Event().wait(10)       # a follower waits here for rank 0's next step

        previous = signal.getsignal(signal.SIGTERM)
        timer = threading.Timer(0.3, os.kill, (os.getpid(), signal.SIGTERM))
        try:
            timer.start()
            decoder._follow_steps = lambda link: Link().receive()
            decoder.follow(Link())
        finally:
            timer.cancel()
            signal.signal(signal.SIGTERM, previous)
        self.assertEqual(Spill.drained, 1)

    def test_follower_drains_its_spill_writes_when_rank_0_closes_the_link_first(self):
        import signal

        class Spill:
            drained = 0

            def shutdown_drain(self):
                Spill.drained += 1

        previous = signal.getsignal(signal.SIGTERM)
        try:
            decoder = self.follower(Spill())

            def broken(link):
                raise RuntimeError("rank 0's step link closed")      # stop.sh stops rank 0 first

            decoder._follow_steps = broken
            with self.assertRaises(RuntimeError):
                decoder.follow(object())
        finally:
            signal.signal(signal.SIGTERM, previous)
        self.assertEqual(Spill.drained, 1)

    def test_follower_without_a_spill_tier_stops_cleanly_and_other_errors_pass_through(self):
        import signal

        previous = signal.getsignal(signal.SIGTERM)
        try:
            decoder = self.follower(None)

            def steps(link):
                raise KeyboardInterrupt

            decoder._follow_steps = steps
            decoder.follow(object())
            self.assertTrue(callable(signal.getsignal(signal.SIGTERM)))

            def broken(link):
                raise RuntimeError("rank 0's step link closed")

            decoder._follow_steps = broken
            with self.assertRaises(RuntimeError):
                decoder.follow(object())
        finally:
            signal.signal(signal.SIGTERM, previous)

    def test_drift_error_names_the_expected_and_the_found_hash(self):
        install_root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, install_root, True)
        for relative in spill.FILES:
            target = install_root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text((SOURCE / relative).read_text())
        changed = install_root / spill.MULTI
        changed.write_text(changed.read_text() + "# local patch" + chr(10))
        found = spill.hashlib.sha256(changed.read_text().encode()).hexdigest()
        with mock.patch.object(spill.subprocess, "check_output", return_value=spill.REVISION + chr(10)):
            with self.assertRaises(ValueError) as raised:
                spill.install(install_root)
        message = str(raised.exception)
        self.assertIn(str(spill.MULTI), message)
        self.assertIn(spill.FILES[spill.MULTI], message)
        self.assertIn(found, message)


if __name__ == "__main__":
    unittest.main()
