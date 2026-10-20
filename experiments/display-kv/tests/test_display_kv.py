import importlib.util
import os
import sys
import tempfile
import unittest
from types import SimpleNamespace


MODULE = os.path.join(os.path.dirname(__file__), "..", "overlay", "display_kv.py")


def load():
    spec = importlib.util.spec_from_file_location("display_kv_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def config():
    return SimpleNamespace(
        kv_sources=[2, 8, 14, 20],
        compress_ratios=[1] * 43,
        head_dim=128,
        idx_dim=64,
        index_sources=[2, 8, 14, 20, 24, 28, 32, 36],
    )


class DisplayKvTests(unittest.TestCase):
    def setUp(self):
        self.dk = load()

    def test_setting_validation(self):
        self.assertEqual(self.dk.enabled_bytes({}), 0)
        self.assertEqual(self.dk.enabled_bytes({self.dk.ENV: "16"}), 16 << 20)
        for value in ("15", "1793", "-16", "x"):
            with self.assertRaises(ValueError):
                self.dk.enabled_bytes({self.dk.ENV: value})

    def test_capacity_fits_and_next_does_not(self):
        cfg = config()
        cfg.compress_ratios[2], cfg.compress_ratios[8] = 1, 4
        cfg.compress_ratios[14], cfg.compress_ratios[20] = 8, 16
        display = self.dk.layout_bytes(cfg, 8192)
        cap = self.dk.pool_capacity(cfg, display)
        self.assertEqual(cap, 8192)
        self.assertLessEqual(self.dk.budget(cfg, cap), display)
        self.assertGreater(self.dk.budget(cfg, cap + self.dk.CAP_ALIGN), display)
        self.assertTrue(all(item.offset % 256 == 0 for item in self.dk.plan(cfg, cap)))
        self.assertEqual({item.layer for item in self.dk.plan(cfg, cap) if item.kind == "index"}, {2, 8, 14, 20})

    def test_real_model_compression_vector_has_uncompressed_layers(self):
        cfg = config()
        cfg.head_dim, cfg.idx_dim = 512, 128
        cfg.compress_ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
        total = 1792 << 20
        cap = self.dk.pool_capacity(cfg, total)
        self.assertGreater(cap, 2_097_152)
        self.assertLessEqual(self.dk.layout_bytes(cfg, cap), total)
        self.assertGreater(self.dk.layout_bytes(cfg, cap + 2048), total)
        self.assertAlmostEqual(self.dk.layout_bytes(cfg, 1048576) / (1 << 20), 890, places=1)

    def test_nvidia_detection_and_selected_connectors_only(self):
        with tempfile.TemporaryDirectory() as root:

            def write(path, text):
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as file:
                    file.write(text)

            write(os.path.join(root, "card0", "device", "vendor"), "0x10de\n")
            write(os.path.join(root, "card0-HDMI-A-1", "status"), "disconnected\n")
            write(os.path.join(root, "card1", "device", "uevent"), "DRIVER=nvidia\n")
            write(os.path.join(root, "card1-DP-1", "status"), "connected\n")
            write(os.path.join(root, "card1-DP-1", "enabled"), "enabled\n")
            write(os.path.join(root, "card0-HDMI-A-1", "status"), "connected\n")
            write(os.path.join(root, "card0-HDMI-A-1", "enabled"), "disabled\n")
            write(os.path.join(root, "tty0", "status"), "connected\n")
            self.assertEqual(self.dk.selected_card({}, root), "/dev/dri/card0")
            self.assertEqual(self.dk.connected_outputs(root, "/dev/dri/card0"), [])
            self.assertEqual(self.dk.connected_outputs(root, "/dev/dri/card1"), ["DP-1"])
            with self.assertRaises(RuntimeError):
                self.dk.selected_card({self.dk.CARD_ENV: "/dev/dri/card2"}, root)

    def test_mapping_failure_unwinds(self):
        calls = []

        class Fake:
            def __init__(self, card):
                pass

            def open_card(self):
                calls.append("open")
                return 4

            def create_dumb(self, fd, size):
                calls.append("create")
                return 8, size

            def map_offset(self, fd, handle):
                calls.append("offset")
                return 0

            def map_display(self, size, fd, offset):
                calls.append("map")
                return 99

            def register(self, address, size):
                calls.append("register")

            def device_pointer(self, address):
                calls.append("pointer")
                raise OSError("no pointer")

            def unregister(self, address):
                calls.append("unregister")

            def unmap(self, address, size):
                calls.append("unmap")

            def close_gem(self, fd, handle):
                calls.append("gem")

            def close_fd(self, fd):
                calls.append("close")

        self.dk.SYSTEM = Fake
        with tempfile.TemporaryDirectory() as root:
            os.makedirs(os.path.join(root, "card0", "device"))
            with open(os.path.join(root, "card0", "device", "vendor"), "w", encoding="utf-8") as file:
                file.write("0x10de")
            with self.assertRaises(OSError):
                self.dk.map_display(4096, environ={self.dk.CARD_ENV: "/dev/dri/card0"}, sysfs=root)
        self.assertEqual(calls[-4:], ["unregister", "unmap", "gem", "close"])

    def test_register_failure_and_pointer_mismatch_unwind(self):
        self.dk.selected_card = lambda *args: "/dev/dri/card0"
        self.dk.connected_outputs = lambda *args: []
        for failure, expected in (
            ("register", ["unmap", "gem", "close"]),
            ("pointer", ["unregister", "unmap", "gem", "close"]),
        ):
            calls = []

            class Fake:
                def __init__(self, card):
                    pass

                def open_card(self):
                    calls.append("open")
                    return 4

                def create_dumb(self, fd, size):
                    calls.append("create")
                    return 8, size

                def map_offset(self, fd, handle):
                    calls.append("offset")
                    return 0

                def map_display(self, size, fd, offset):
                    calls.append("map")
                    return 99

                def register(self, address, size):
                    calls.append("register")
                    if failure == "register":
                        raise OSError("register")

                def device_pointer(self, address):
                    calls.append("pointer")
                    return 100 if failure == "pointer" else address

                def unregister(self, address):
                    calls.append("unregister")

                def unmap(self, address, size):
                    calls.append("unmap")

                def close_gem(self, fd, handle):
                    calls.append("gem")

                def close_fd(self, fd):
                    calls.append("close")

            self.dk.SYSTEM = Fake
            with self.assertRaises((OSError, RuntimeError)):
                self.dk.map_display(4096)
            self.assertEqual(calls[-len(expected) :], expected)

    def test_as_tensor_pointer_mismatch_never_falls_back(self):
        class Result:
            def data_ptr(self):
                return 17

        class FakeTorch:
            calls = 0

            @staticmethod
            def as_tensor(span, device=None):
                FakeTorch.calls += 1
                return Result()

        old = sys.modules.get("torch")
        sys.modules["torch"] = FakeTorch
        owner = object()
        span = self.dk.Span(18, 32, owner)
        try:
            with self.assertRaises(RuntimeError):
                self.dk.as_tensor(span)
            self.assertIs(span._owner, owner)
            self.assertEqual(FakeTorch.calls, 1)
        finally:
            if old is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = old

    def test_copy_rows_uses_bounded_scratch_and_memmove_order(self):
        class Tensor:
            def __init__(self, values, width, indexes=None):
                self.values, self.width = values, width
                self.indexes = list(range(len(values))) if indexes is None else indexes
                self.dtype = uint8
                self.shape = (len(self.indexes), width)

            def __getitem__(self, item):
                if isinstance(item, slice):
                    return Tensor(self.values, self.width, self.indexes[item])
                return self.values[self.indexes[item]]

            def numel(self):
                return len(self.indexes) * self.width

            def element_size(self):
                return 1

        class FakeTorch:
            uint8 = object()
            calls = []

            @staticmethod
            def empty_like(source):
                return Tensor([0] * len(source.indexes), source.width)

            @classmethod
            def bitwise_or(cls, source, zero, out):
                cls.calls.append(source.shape)
                for destination, value in zip(out.indexes, (source.values[i] for i in source.indexes)):
                    out.values[destination] = value

        uint8 = FakeTorch.uint8
        old = sys.modules.get("torch")
        sys.modules["torch"] = FakeTorch
        try:
            forward = Tensor(list(range(6)), 1)
            self.dk.copy_rows(forward, 1, 0, 5)
            self.assertEqual(forward.values, [1, 2, 3, 4, 5, 5])
            backward = Tensor(list(range(6)), 1)
            self.dk.copy_rows(backward, 0, 1, 5)
            self.assertEqual(backward.values, [0, 0, 1, 2, 3, 4])
            separate = Tensor(list(range(6)), 1)
            self.dk.copy_rows(separate, 0, 3, 3)
            self.assertEqual(separate.values, [0, 1, 2, 0, 1, 2])
            wide = Tensor(list(range(6)), 600000)
            self.dk.copy_rows(wide, 0, 3, 3)
            self.assertTrue(all(rows * width <= 1 << 20 for rows, width in FakeTorch.calls))
        finally:
            if old is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = old


if __name__ == "__main__":
    unittest.main()
