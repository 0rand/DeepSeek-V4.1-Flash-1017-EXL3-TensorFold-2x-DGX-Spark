"""CPU-only checks for the reused GLM spill tier (real CPU torch, no GPU)."""
import ast
import importlib.util
import os
import sys
import tempfile
import unittest
import zlib
from dataclasses import dataclass

try:
    import numpy as np
    import torch
except ImportError as exc:      # CI has no torch: the CPU tests of the spill tier skip themselves
    raise unittest.SkipTest(f"needs numpy and torch: {exc}")

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
        allowed = {"__future__", "base64", "contextlib", "dataclasses", "errno", "hashlib", "importlib", "json",
                   "os", "queue", "re", "shutil", "struct", "threading", "time", "zlib", "collections",
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
            self.assertEqual(info["gib"], 1)
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

    def test_corrupt_file_is_rejected(self):
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
            with open(job.path, "r+b") as f:
                f.seek(-1, os.SEEK_END)
                last = f.read(1)
                f.seek(-1, os.SEEK_END)
                f.write(bytes([last[0] ^ 0xFF]))
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            with self.assertRaises(Exception):
                store.finish_load(lj)
            self.assertEqual(store.info()["crc_failed"], 1)

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

    def stored(self, root, ids=None, **config):
        """A store with one written entry; returns (store, ids, classes)."""
        cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1, **config)
        classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
        store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
        ids = list(range(100)) if ids is None else ids
        snap = Snap(ids=ids, ring=torch.arange(len(ids) * 4, dtype=torch.uint8).reshape(len(ids), 4))
        self.assertIsNotNone(store.save(ids, snap, skip=("ids",)))
        self.assertTrue(store.drain(10))
        return store, ids, classes

    def entry_path(self, store):
        return os.path.join(store.dir, [n for n in os.listdir(store.dir) if n.endswith(".safetensors")][0])

    def flip_in_header(self, path, old: bytes, new: bytes):
        with open(path, "r+b") as f:
            raw = f.read(1 << 16)
            self.assertEqual(raw.count(old), 1, old)
            f.seek(raw.index(old))
            f.write(new)

    def test_a_flipped_header_field_is_refused_by_the_scan_and_by_a_restore(self):
        for old, new in ((b'"head":"0"', b'"head":"1"'), (b'"rows":"[]"', b'"rows":"[x]"'),
                         (b'"n":"100"', b'"n":"101"')):
            with self.subTest(field=old), tempfile.TemporaryDirectory() as root:
                store, ids, classes = self.stored(root)
                path = self.entry_path(store)
                # A restart finds an intact file.
                cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
                again = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
                self.assertTrue(again.has(ids))
                self.flip_in_header(path, old, new)
                with self.assertRaises(Exception):
                    again.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
                self.assertEqual(again.info()["load_failed"], 1)
                # A restart drops the file.
                third = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
                self.assertFalse(third.has(ids))
                self.assertFalse(os.path.exists(path))

    def test_header_without_its_own_crc_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            self.flip_in_header(self.entry_path(store), b'"hcrc":', b'"hxrc":')
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            self.assertFalse(self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes).has(ids))

    def test_missing_file_counts_one_failed_restore(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, _ = self.stored(root)
            os.remove(self.entry_path(store))
            with self.assertRaises(OSError):
                store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            self.assertEqual(store.info()["load_failed"], 1)
            self.assertEqual(store.info()["crc_failed"], 0)
            self.assertEqual(store.info()["loaded"], 0)

    def test_unknown_key_counts_one_failed_restore(self):
        with tempfile.TemporaryDirectory() as root:
            store, _, _ = self.stored(root)
            with self.assertRaises(FileNotFoundError):
                store.load_async((1, 2), lambda name, shape, dt: None)
            self.assertEqual(store.info()["load_failed"], 1)

    def test_truncated_file_counts_one_failed_restore_not_a_crc_failure(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, _ = self.stored(root)
            path = self.entry_path(store)
            with open(path, "r+b") as f:
                f.truncate(os.path.getsize(path) - 100)
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            with self.assertRaises(Exception):
                store.finish_load(lj)
            with self.assertRaises(Exception):          # the same job again: still one count
                store.finish_load(lj)
            info = store.info()
            self.assertEqual(info["load_failed"], 1)
            self.assertEqual(info["crc_failed"], 0)
            self.assertEqual(info["loaded"], 0)

    def test_crc_mismatch_counts_in_both_counters(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, _ = self.stored(root)
            with open(self.entry_path(store), "r+b") as f:
                f.seek(-1, os.SEEK_END)
                last = f.read(1)
                f.seek(-1, os.SEEK_END)
                f.write(bytes([last[0] ^ 0xFF]))
            lj = store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            with self.assertRaises(ValueError):
                store.finish_load(lj)
            self.assertEqual(store.info()["load_failed"], 1)
            self.assertEqual(store.info()["crc_failed"], 1)

    def test_a_rank_local_write_failure_does_not_change_the_candidate(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, _ = self.stored(root)
            before = store.find(ids, whole_needs_head=False)
            self.assertIsNotNone(before)
            before.bad = True                      # this rank failed to write it; the other rank did not
            after = store.find(ids, whole_needs_head=False)
            self.assertIs(after, before)           # both ranks must pick it, then the read fails here and both forget
            with self.assertRaises(Exception):
                store.load_async(before.key, lambda name, shape, dt: None)
            self.assertEqual(store.info()["load_failed"], 1)
            store.forget(before.key)
            self.assertIsNone(store.find(ids, whole_needs_head=False))

    def test_header_with_a_shape_larger_than_its_span_is_refused(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, classes = self.stored(root)
            path = self.entry_path(store)
            with open(path, "rb") as f:
                raw = f.read()
            self.assertEqual(raw.count(b'"shape":[100,4]'), 1)
            with open(path, "wb") as f:                  # same length, so the file size and the CRC bytes still fit
                f.write(raw.replace(b'"shape":[100,4]', b'"shape":[100,5]'))
            with self.assertRaises(ValueError):
                store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None)
            self.assertEqual(store.info()["load_failed"], 1)
            again = self.sp.SpillStore(store.cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            self.assertEqual(again.info()["entries"], 0)        # the start scan does not trust the header either

    def test_descriptor_check_refuses_gaps_overlaps_negative_numbers_and_unknown_dtypes(self):
        good = {"__metadata__": {}, "a": {"dtype": next(iter(self.sp._TD)), "shape": [2], "data_offsets": [0, 0]}}
        code = good["a"]["dtype"]
        size = torch.empty((), dtype=self.sp._TD[code]).element_size()
        good["a"]["data_offsets"] = [0, 2 * size]
        self.assertEqual(self.sp._check_table(good), 2 * size)
        second = {"dtype": code, "shape": [1], "data_offsets": [2 * size, 3 * size]}
        self.assertEqual(self.sp._check_table({**good, "b": second}), 3 * size)
        bad = [
            {**good, "b": {**second, "data_offsets": [3 * size, 4 * size]}},          # a gap
            {**good, "b": {**second, "data_offsets": [size, 2 * size]}},               # an overlap
            {"a": {**good["a"], "shape": [-2], "data_offsets": [0, 2 * size]}},         # a negative shape
            {"a": {**good["a"], "dtype": "nope"}},                                      # an unknown dtype
            {"a": {**good["a"], "shape": [10 ** 12], "data_offsets": [0, 2 * size]}},   # a huge shape
        ]
        for header in bad:
            with self.subTest(header=header):
                with self.assertRaises(ValueError):
                    self.sp._check_table(header)

    def test_good_restore_counts_no_failure(self):
        with tempfile.TemporaryDirectory() as root:
            store, ids, _ = self.stored(root)
            store.finish_load(store.load_async(self.sp.ids_key(ids), lambda name, shape, dt: None))
            self.assertEqual(store.info()["load_failed"], 0)
            self.assertEqual(store.info()["loaded"], 1)

    def test_shutdown_budget_follows_the_flush_setting_below_the_docker_stop_limit(self):
        with tempfile.TemporaryDirectory() as root:
            for flush, budget in ((10.0, 10.0), (60.0, 55.0), (100.0, 55.0), (0.0, 0.0)):
                with self.subTest(flush=flush):
                    cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, flush_s=flush)
                    store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", quiet=True)
                    self.assertEqual(store.shutdown_budget(), budget)
            self.assertLess(self.sp.SHUTDOWN_CAP_S, 60.0)

    def test_shutdown_drain_uses_the_budget_and_reports_unfinished_writes(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, flush_s=7.0)
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", quiet=True)
            seen = []
            store.drain = lambda timeout=None: seen.append(timeout) or True
            self.assertTrue(store.shutdown_drain())
            self.assertEqual(seen, [7.0])
            store.drain = lambda timeout=None: False
            self.assertFalse(store.shutdown_drain())

    def test_token_ids_are_stored_as_base64_and_survive_a_restart(self):
        with tempfile.TemporaryDirectory() as root:
            ids = np.arange(5000, dtype=np.int32) * 7 + 3
            store, _, classes = self.stored(root, ids=ids)
            with open(self.entry_path(store), "rb") as f:
                header, _ = self.sp._read_header_file(f)
            tokens = header["__metadata__"]["tokens"]
            self.assertFalse(tokens.startswith("["))
            self.assertTrue(np.array_equal(self.sp._decode_ids(tokens), ids.astype(np.int64)))
            again = self.sp.SpillStore(store.cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            self.assertTrue(again.has(ids))
            self.assertEqual(again.info()["entries"], 1)

    def test_image_keyed_prompt_keeps_both_id_lists(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = self.sp.SpillConfig(root=root, gib=1, min_free_gib=0, min_tokens=1)
            classes = [f"{Snap.__module__}:{Snap.__qualname__}"]
            store = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes)
            keys = np.array([5, -(1 << 40), 7, 1 << 50], dtype=np.int64)
            tokens = np.array([1, 2, 3, 4], dtype=np.int32)
            snap = Snap(ids=[], ring=torch.zeros((4, 4), dtype=torch.uint8))
            self.assertIsNotNone(store.save(keys, snap, skip=("ids",), tokens=tokens))
            self.assertTrue(store.drain(10))
            again = self.sp.SpillStore(cfg, rank=0, world=1, device="cpu", classes=classes, quiet=True)
            [entry] = again.index.values()
            self.assertTrue(np.array_equal(entry.ids, keys))

    def test_one_million_token_ids_are_cheap_to_encode(self):
        ids = np.arange(1_000_000, dtype=np.int32)
        start = self.sp.time.perf_counter()
        text = self.sp._encode_ids(ids)
        elapsed = self.sp.time.perf_counter() - start
        self.assertTrue(np.array_equal(self.sp._decode_ids(text), ids.astype(np.int64)))
        self.assertLess(elapsed, 0.25, "encoding 1M ids must stay far below a JSON list's cost")


if __name__ == "__main__":
    unittest.main()
