"""CPU-only checks for the offline display-KV source installer."""

from __future__ import annotations

import ast
import importlib.util
import os
from pathlib import Path
import hashlib
import tempfile
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ["TF_DISPLAY_SOURCE"]) if os.environ.get("TF_DISPLAY_SOURCE") else None
APPLY_PATH = ROOT / "apply.py"
spec = importlib.util.spec_from_file_location("display_kv_apply", APPLY_PATH)
apply = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(apply)


@unittest.skipUnless(SOURCE is not None and SOURCE.is_dir(), "set TF_DISPLAY_SOURCE to a pinned checkout")
class DisplayKvIntegrationTest(unittest.TestCase):
    def source(self, name: str) -> tuple[Path, str]:
        path = SOURCE / apply.FAMILY / name
        return path.relative_to(SOURCE), path.read_text()

    def test_all_pinned_sources_transform_and_compile(self):
        for relative, expected in apply.FILES.items():
            text = (SOURCE / relative).read_text()
            self.assertEqual(apply.hashlib.sha256(text.encode()).hexdigest(), expected)
            changed = apply.transform(relative, text)
            ast.parse(changed, filename=str(relative))

    def test_disabled_path_keeps_ordinary_cache_allocators(self):
        relative, text = self.source("model.py")
        changed = apply.transform(relative, text)
        self.assertIn("display_bytes: int = 0", changed)
        self.assertIn("arena = display_kv.CacheArena(c, cap, display_bytes) if display_bytes else None", changed)
        self.assertIn('else lambda shape, dtype: torch.zeros(shape, dtype=dtype, device="cuda")', changed)
        self.assertIn("sc.ring = [torch.zeros", changed)
        self.assertIn("sc.tokens = torch.zeros", changed)

    def test_display_pool_is_independent_of_request_limit_and_rank_flags_match(self):
        relative, text = self.source("engine.py")
        changed = apply.transform(relative, text)
        self.assertIn("display_kv.pool_capacity(cfg, self.display_bytes, max_rows=0)", changed)
        self.assertIn("self.display_pool_cap < self.limit + self.max_rows + 8", changed)
        self.assertIn("display KV requires --parallel greater than 1 before weights load", changed)
        self.assertIn("display-only KV requires TF_DS_SPILL_GIB=0", changed)
        self.assertIn("display KV requires TF_DS_KV=native packed storage", changed)
        self.assertIn("world * 7", changed)
        self.assertIn("self.display_bytes, self.display_pool_cap, self.limit", changed)
        relative, text = self.source("model.py")
        changed = apply.transform(relative, text)
        self.assertIn("self.display_arena = arena", changed)
        self.assertIn('getattr(self, "display_arena", None) is not None', changed)
        relative, text = self.source("multi.py")
        changed = apply.transform(relative, text)
        self.assertIn("display_kv.pool_capacity(m.cfg, engine.display_bytes, max_rows=0)", changed)
        self.assertIn("admit_cap = min(self.extents.total, self.e.limit)", changed)
        self.assertIn("display_kv.copy_rows(t, a, b, k)", changed)

    def test_transform_rejects_modified_or_installed_source(self):
        relative, text = self.source("model.py")
        with self.assertRaises(ValueError):
            apply.transform(relative, text.replace("def new_cache", "def changed_cache", 1))
        with self.assertRaises(ValueError):
            apply.transform(relative, text + "\n# TF_DS_DISPLAY_KV_MIB\n")

    def test_display_model_rope_cap_applies_to_round_cache_without_an_arena(self):
        relative, text = self.source("model.py")
        tree = ast.parse(apply.transform(relative, text), filename=str(relative))
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
        method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "_cs")
        method.decorator_list, method.returns = [], None
        for arg in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
            arg.annotation = None
        namespace = {}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "_cs", "exec"), namespace)

        class Rope:
            def __init__(self):
                self.calls = []

            def cs(self, compressed, cap):
                self.calls.append((compressed, cap))
                return self.calls[-1]

        class Model:
            _cs = namespace["_cs"]

        class SeqCache:
            cap = 2_100_000

        display = Model()
        display.cfg = type("Cfg", (), {"compress_ratios": [1]})()
        display.rope = Rope()
        display.rope_cap = 1_000_008
        display.display_arena = object()
        self.assertEqual(display._cs(0, SeqCache()), (True, 1_000_008))

        ordinary = Model()
        ordinary.cfg, ordinary.rope, ordinary.rope_cap = display.cfg, Rope(), display.rope_cap
        self.assertEqual(ordinary._cs(0, SeqCache()), (True, 2_100_000))

    def test_new_cache_places_only_packed_planes_in_display_arena(self):
        overlay_spec = importlib.util.spec_from_file_location("display_kv_overlay", ROOT / "overlay" / "display_kv.py")
        overlay = importlib.util.module_from_spec(overlay_spec)
        assert overlay_spec.loader is not None
        overlay_spec.loader.exec_module(overlay)

        class Cfg:
            kv_sources = [2, 8, 14, 20]
            index_sources = [2, 8, 14, 20]
            compress_ratios = [0, 0] + [2] * 18 + [1] * 20 + [0] * 3
            head_dim = 512
            idx_dim = 128
            window = 128

        cfg = Cfg()
        budget = overlay.MAX_MIB * overlay.MIB
        cap = overlay.pool_capacity(cfg, budget, max_rows=0)
        expected = overlay.layout_bytes(cfg, cap)
        self.assertLessEqual(expected, budget)

        class Tensor:
            def __init__(self, shape, dtype, origin, fill=0):
                self.shape, self.dtype, self.origin, self.fill = tuple(shape), dtype, origin, fill

        class FakeArena:
            instances = []

            def __init__(self, got_cfg, got_cap, display_bytes):
                self.cfg, self.cap, self.size, self.offset, self.calls = got_cfg, got_cap, display_bytes, 0, []
                if overlay.layout_bytes(got_cfg, got_cap) > display_bytes:
                    raise ValueError("display arena too small")
                self.instances.append(self)

            def allocate(self, shape, dtype, fill=0):
                width = 1 if dtype == "u8" else 0
                self.offset = overlay._up(self.offset)
                count = 1
                for dimension in shape:
                    count *= dimension
                self.offset += count * width
                tensor = Tensor(shape, dtype, "display", fill)
                self.calls.append(tensor)
                return tensor

            def zeros(self, shape, dtype):
                return self.allocate(shape, dtype)

            def full(self, shape, fill, dtype):
                return self.allocate(shape, dtype, fill)

        ordinary = []

        def zeros(shape, dtype, device):
            tensor = Tensor(shape, dtype, "ordinary")
            ordinary.append(tensor)
            return tensor

        def full(shape, fill, dtype, device):
            tensor = Tensor(shape, dtype, "ordinary", fill)
            ordinary.append(tensor)
            return tensor

        relative, text = self.source("model.py")
        tree = ast.parse(apply.transform(relative, text), filename=str(relative))
        model = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "Model")
        method = next(node for node in model.body if isinstance(node, ast.FunctionDef) and node.name == "new_cache")
        method.decorator_list, method.returns = [], None
        for arg in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
            arg.annotation = None
        namespace = {
            "KV_QUANT": True,
            "RING_EXTRA": 16,
            "RAW": 64,
            "BF16": "bf16",
            "F32": "f32",
            "torch": types.SimpleNamespace(uint8="u8", int64="i64", zeros=zeros, full=full),
            "display_kv": types.SimpleNamespace(CacheArena=FakeArena),
        }

        class SeqCache:
            def __init__(self, **values):
                self.__dict__.update(values)
                self.ring, self.comp, self.index_k, self.comp_raw, self.tokens = [], {}, {}, {}, None

        namespace["SeqCache"] = SeqCache
        exec(
            compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "new_cache", "exec"),
            namespace,
        )

        class Model:
            new_cache = namespace["new_cache"]

        model = Model()
        model.cfg = cfg
        model.w = types.SimpleNamespace(layers=[object() for _ in range(40)])
        cache = model.new_cache(cap, display_bytes=budget)
        arena = FakeArena.instances[-1]
        self.assertIs(model.display_arena, arena)
        self.assertEqual(overlay._up(arena.offset), expected)
        self.assertTrue(all(value.origin == "display" for pair in cache.comp.values() for value in pair))
        self.assertTrue(all(value.origin == "display" for pair in cache.index_k.values() for value in pair))
        self.assertTrue(all(value.origin == "ordinary" for value in cache.ring))
        self.assertTrue(all(value.origin == "ordinary" for pair in cache.comp_raw.values() for value in pair))
        self.assertEqual(cache.tokens.origin, "ordinary")
        self.assertEqual(sum(value.fill == 127 for value in arena.calls), len(cfg.index_sources))


class InstallerValidationTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.checkout = Path(self.temp.name)
        self.originals = {path: f"# {path.name}\nvalue = 1\n" for path in apply.FILES}
        for path, text in self.originals.items():
            target = self.checkout / path
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text)
        hashes = {path: hashlib.sha256(text.encode()).hexdigest() for path, text in self.originals.items()}
        self.enterContext(mock.patch.object(apply, "FILES", hashes))
        self.head = self.enterContext(mock.patch.object(apply, "_git_head", return_value=apply.REVISION))

    def snapshot(self):
        return {
            path.relative_to(self.checkout): path.read_bytes() for path in self.checkout.rglob("*") if path.is_file()
        }

    def test_install_copies_only_display_overlay_and_refuses_reinstall(self):
        with mock.patch.object(apply, "transform", side_effect=lambda path, text: text + "# transformed\n"):
            apply.install(self.checkout)
        added = set(self.snapshot()) - set(self.originals)
        self.assertEqual(added, {apply.FAMILY / "display_kv.py"})
        self.assertEqual(
            (self.checkout / apply.FAMILY / "display_kv.py").read_text(), (ROOT / "overlay/display_kv.py").read_text()
        )
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "already installed"):
            apply.install(self.checkout)
        self.assertEqual(self.snapshot(), before)

    def test_revision_rejection_leaves_checkout_unchanged(self):
        self.head.return_value = "wrong"
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "HEAD"):
            apply.install(self.checkout)
        self.assertEqual(self.snapshot(), before)

    def test_hash_rejection_leaves_checkout_unchanged(self):
        (self.checkout / next(iter(self.originals))).write_text("changed = True\n")
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            apply.install(self.checkout)
        self.assertEqual(self.snapshot(), before)

    def test_bad_anchor_leaves_checkout_unchanged(self):
        before = self.snapshot()
        with self.assertRaisesRegex(ValueError, "anchor"):
            apply.install(self.checkout)
        self.assertEqual(self.snapshot(), before)

    def test_late_transform_rejection_leaves_checkout_unchanged(self):
        before = self.snapshot()
        with mock.patch.object(
            apply, "transform", side_effect=["valid = 1\n", "valid = 2\n", ValueError("bad anchor")]
        ):
            with self.assertRaisesRegex(ValueError, "anchor"):
                apply.install(self.checkout)
        self.assertEqual(self.snapshot(), before)

    def test_ast_rejection_leaves_checkout_unchanged(self):
        before = self.snapshot()
        with mock.patch.object(apply, "transform", side_effect=["valid = 1\n", "valid = 2\n", "def broken("]):
            with self.assertRaises(SyntaxError):
                apply.install(self.checkout)
        self.assertEqual(self.snapshot(), before)

    def test_overlay_ast_rejection_leaves_checkout_unchanged(self):
        before = self.snapshot()
        with mock.patch.object(apply, "transform", side_effect=lambda path, text: text):
            with mock.patch.object(apply, "overlays", return_value={apply.FAMILY / "display_kv.py": "def broken("}):
                with self.assertRaises(SyntaxError):
                    apply.install(self.checkout)
        self.assertEqual(self.snapshot(), before)

    def test_only_three_display_sources_are_supported(self):
        self.assertEqual({path.name for path in apply.FILES}, {"model.py", "engine.py", "multi.py"})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            apply.transform("src/tensorfold/cuda/http.py", "pass\n")
        self.assertEqual(set(apply.overlays()), {apply.FAMILY / "display_kv.py"})


if __name__ == "__main__":
    unittest.main()
