"""CPU-only checks for the reused GLM spill tier (real CPU torch, no GPU)."""
import ast
import importlib.util
import os
import sys
import tempfile
import unittest
import zlib
from dataclasses import dataclass

import numpy as np
import torch

MODULE = os.path.join(os.path.dirname(__file__), "..", "overlay", "spill_tier.py")


def load():
    spec = importlib.util.spec_from_file_location("spill_tier_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    sys.modules["spill_tier_test"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Snap:
    ids: list
    ring: torch.Tensor
    raw: torch.Tensor | None = None


@dataclass
class KeptLike:
    eid: int
    base: int
    size: int
    top: int
    host: object
    snaps: dict
    replay: bool
    keys: object
    tick: int


class SpillTierTests(unittest.TestCase):
    def setUp(self):
        self.sp = load()

    def test_module_is_self_contained(self):
        tree = ast.parse(open(MODULE, encoding="utf-8").read())
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        allowed = {"__future__", "contextlib", "dataclasses", "errno", "hashlib", "importlib", "json",
                   "os", "queue", "shutil", "struct", "threading", "time", "zlib", "collections",
                   "concurrent", "typing", "pathlib", "numpy", "torch", "tensorfold"}
        self.assertEqual(imports - allowed, set())

    def test_config_validation(self):
        with self.assertRaises(ValueError):
            self.sp.SpillConfig(root="/tmp", gib=0)
        with self.assertRaises(ValueError):
            self.sp.SpillConfig(root="/tmp", gib=1, highwater=0)
        with self.assertRaises(ValueError):
            self.sp.SpillConfig(root="/tmp", gib=1, writers=0)
        cfg = self.sp.SpillConfig(root="/tmp", gib=1)
        self.assertEqual(cfg.writers, 2)
        self.assertTrue(cfg.direct)

    def test_ids_key_and_name(self):
        key = self.sp.ids_key([1, 2, 3])
        self.assertEqual(len(key), 2)
        self.assertEqual(self.sp.key_name(key, 3),
                         f"{key[0] & 0xffffffff:08x}{key[1] & 0xffffffff:08x}_3")

    def test_encode_decode_roundtrip(self):
        snap = Snap(ids=[1, 2], ring=torch.arange(6, dtype=torch.uint8).reshape(2, 3))
        items, layer = self.sp.encode(snap, skip=("ids",))
        self.assertEqual(layer["class"], f"{Snap.__module__}:{Snap.__qualname__}")
        tensors = dict(items)
        obj = self.sp.decode(layer, tensors, [f"{Snap.__module__}:{Snap.__qualname__}"])
        self.assertTrue(torch.equal(obj.ring, snap.ring))
        self.assertIsNone(obj.raw)
        self.assertFalse(hasattr(obj, "ids"))

    def test_by_kernel_bounded_copy(self):
        src = torch.arange(1000, dtype=torch.uint8)
        dst = torch.zeros_like(src)
        stage = torch.empty(64, dtype=torch.uint8)
        self.sp._by_kernel(dst, src, stage, None)
        self.assertTrue(torch.equal(dst, src))

    def test_header_roundtrip(self):
        items = [("a", torch.zeros(4, dtype=torch.uint8)), ("b", torch.ones(2, dtype=torch.int32))]
        header, data = self.sp._header(items, {"format": "x"})
        parsed, data_at = self.sp._parse_header(header)
        self.assertEqual(data_at, len(header))
        self.assertEqual(parsed["__metadata__"]["format"], "x")
        self.assertEqual(parsed["a"]["data_offsets"], [0, 4])
        self.assertEqual(parsed["b"]["data_offsets"], [4, 12])
        self.assertEqual(data, 12)

    def test_crcs(self):
        crcs = self.sp._Crcs()
        crcs.feed(b"hello")
        self.assertEqual(crcs.hex(), f"{zlib.crc32(b'hello'):08x}")

    def test_info_has_enginemetrics_fields(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu")
            info = store.info()
            for key in ("entries", "stored_bytes", "written", "written_bytes", "loaded",
                        "loaded_bytes", "loaded_tokens", "restore_ms_p50", "restore_ms_p95",
                        "write_failed", "crc_failed", "load_failed", "load_unfit", "disk_free_bytes"):
                self.assertIn(key, info)

    def test_kept_like_object_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            classes = [f"{KeptLike.__module__}:{KeptLike.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
            ids = list(range(100))
            host = np.arange(100, dtype=np.int32)
            snaps = {"32": (torch.arange(8, dtype=torch.uint8), None),
                     "64": (torch.arange(8, dtype=torch.uint8), torch.arange(4, dtype=torch.uint8))}
            obj = KeptLike(1, 0, 100, 100, host, snaps, False, np.arange(100, dtype=np.int64), 1)
            job = store.save(ids, obj, skip=("disk", "hits"))
            self.assertIsNotNone(job)
            self.assertTrue(store.drain(10))
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            loaded, rows, meta = store.finish_load(lj)
            self.assertEqual(loaded.eid, 1)
            self.assertTrue(np.array_equal(loaded.host, host))
            self.assertEqual(set(loaded.snaps), {"32", "64"})
            self.assertTrue(torch.equal(loaded.snaps["32"][0], snaps["32"][0]))
            self.assertIsNone(loaded.snaps["32"][1])
            self.assertTrue(torch.equal(loaded.snaps["64"][1], snaps["64"][1]))
            self.assertTrue(np.array_equal(loaded.keys, obj.keys))

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
            ids = list(range(100))
            ring = torch.arange(100 * 4, dtype=torch.uint8).reshape(100, 4)
            snap = Snap(ids=ids, ring=ring)
            job = store.save(ids, snap, skip=("ids",))
            self.assertIsNotNone(job)
            self.assertTrue(store.drain(10))
            self.assertTrue(store.has(ids))
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            obj, rows, meta = store.finish_load(lj)
            self.assertTrue(torch.equal(obj.ring, ring))
            self.assertEqual(store.info()["loaded"], 1)
            self.assertEqual(store.info()["loaded_tokens"], 100)


if __name__ == "__main__":
    unittest.main()
