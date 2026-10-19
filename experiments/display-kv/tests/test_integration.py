"""CPU-only checks for the offline display-KV source installer."""
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
SOURCE = Path(os.environ.get("TF_DISPLAY_SOURCE", "/tmp/opencode/tfv41-display-source"))
APPLY_PATH = ROOT / "apply.py"
spec = importlib.util.spec_from_file_location("display_kv_apply", APPLY_PATH)
apply = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(apply)


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
        self.assertIn("else lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=\"cuda\")", changed)
        self.assertIn("sc.ring = [torch.zeros", changed)
        self.assertIn("sc.tokens = torch.zeros", changed)

    def test_display_pool_is_independent_of_request_limit_and_rank_flags_match(self):
        relative, text = self.source("engine.py")
        changed = apply.transform(relative, text)
        self.assertIn("display_kv.pool_capacity(cfg, self.display_bytes, max_rows=0)", changed)
        self.assertIn("self.display_pool_cap < self.limit + self.max_rows + 8", changed)
        self.assertIn("display KV requires --parallel greater than 1 before weights load", changed)
        self.assertNotIn("display-only KV staging requires TF_DS_SPILL_GIB=0", changed)
        self.assertIn("display KV requires TF_DS_KV=native packed storage", changed)
        self.assertIn("world * 7", changed)
        self.assertIn("self.display_bytes, self.display_pool_cap, self.limit", changed)
        relative, text = self.source("model.py")
        changed = apply.transform(relative, text)
        self.assertIn("self.display_arena = arena", changed)
        self.assertIn("getattr(self, \"display_arena\", None) is not None", changed)
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
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "new_cache", "exec"), namespace)

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

    def test_shutdown_drain_hook_is_installed(self):
        relative, text = self.source("app.py")
        changed = apply.transform(relative, text)
        self.assertIn("self.on_exit = self._flush_spill", changed)
        self.assertIn("def _flush_spill(self)", changed)
        relative = Path("src/tensorfold/cuda/http.py")
        changed = apply.transform(relative, (SOURCE / relative).read_text())
        self.assertIn('hook = getattr(app, "on_exit", None)', changed)
        self.assertIn("hook()", changed)

    def test_shutdown_drain_waits_for_queued_writes(self):
        relative, text = self.source("app.py")
        tree = ast.parse(apply.transform(relative, text), filename=str(relative))
        app = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DsApp")
        method = next(node for node in app.body if isinstance(node, ast.FunctionDef) and node.name == "_flush_spill")
        method.decorator_list, method.returns = [], None
        for arg in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
            arg.annotation = None
        namespace = {"time": types.SimpleNamespace(perf_counter=lambda: 0.0)}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "_flush_spill", "exec"),
             namespace)

        class Spill:
            def __init__(self):
                self.calls = []

            def drain(self, timeout):
                self.calls.append(timeout)
                return True

        class Engine:
            def __init__(self):
                self.multi = types.SimpleNamespace(spill=Spill())

        class App:
            _flush_spill = namespace["_flush_spill"]

        app = App()
        app.engine = Engine()
        with mock.patch("builtins.print"):
            app._flush_spill()
        self.assertEqual(app.engine.multi.spill.calls, [55.0])
        app.engine.multi = None
        with mock.patch("builtins.print"):
            app._flush_spill()

    def test_spill_signature_ignores_the_driver_version(self):
        relative, text = self.source("multi.py")
        changed = apply.transform(relative, text)
        self.assertIn('build_id(getattr(engine, "model_dir", ".")).split("|driver=")[0]', changed)
        self.assertNotIn('"dsv41-multi", build_id(getattr(engine, "model_dir", ".")),', changed)

    def test_spill_calls_match_the_reused_tier(self):
        relative, text = self.source("multi.py")
        changed = apply.transform(relative, text)
        tree = ast.parse(changed, filename=str(relative))
        calls = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
                value = node.func.value
                if (isinstance(value, ast.Attribute) and isinstance(value.value, ast.Name)
                        and value.value.id == "self" and value.attr == "spill"):
                    calls.add(node.func.attr)
        self.assertNotIn("lineage", calls)
        self.assertTrue({"save", "find", "load_async", "finish_load", "release", "info", "forget"} <= calls)
        self.assertIn("self.spill.forget(key)", changed)

    def test_spill_share_callback_is_installed(self):
        relative, text = self.source("multi.py")
        changed = apply.transform(relative, text)
        self.assertIn("share=self._share", changed)
        self.assertIn("def _share(self, values", changed)

    def test_spill_share_returns_rank_zero_values_on_every_rank(self):
        relative, text = self.source("multi.py")
        tree = ast.parse(apply.transform(relative, text), filename=str(relative))
        multi = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "MultiDecoder")
        method = next(node for node in multi.body if isinstance(node, ast.FunctionDef) and node.name == "_share")
        method.decorator_list, method.returns = [], None
        for arg in [*method.args.posonlyargs, *method.args.args, *method.args.kwonlyargs]:
            arg.annotation = None

        class Tensor:
            def __init__(self, values):
                self.values = list(values)

            def numel(self):
                return len(self.values)

        class Torch:
            int64 = "int64"

            @staticmethod
            def tensor(values, dtype, device):
                return Tensor(values)

            @staticmethod
            def zeros(n, dtype, device):
                return Tensor([0] * n)

            @staticmethod
            def cat(tensors):
                return Tensor([value for tensor in tensors for value in tensor.values])

        class Rows:
            def __init__(self, rows):
                self.rows = rows

            def tolist(self):
                return self.rows

        class Comm:
            def __init__(self, length=3):
                self.length = length

            def gather(self, tensor):
                if len(tensor.values) == 1:
                    return Rows([[self.length], [0]])
                return Rows([[1, 2, 3], [0, 0, 0]])

        namespace = {"torch": Torch}
        exec(compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "_share", "exec"),
             namespace)

        class Multi:
            _share = namespace["_share"]

        multi = Multi()
        multi.e = types.SimpleNamespace(world=2)
        multi.m = types.SimpleNamespace(comm=Comm())
        self.assertEqual(multi._share([1, 2, 3]), [1, 2, 3])
        self.assertEqual(multi._share(None), [1, 2, 3])
        multi.m.comm = Comm(length=0)
        self.assertEqual(multi._share(None), [])
        multi.e.world = 1
        self.assertEqual(multi._share([4, 5]), [4, 5])
        self.assertEqual(multi._share(None), [])

    def test_install_transforms_minimal_checkout_and_refuses_reinstall(self):
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            for relative in apply.FILES:
                target = checkout / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(SOURCE / relative, target)
            with mock.patch.object(apply, "_git_head", return_value=apply.REVISION):
                apply.install(checkout)
                for relative in apply.FILES:
                    changed = (checkout / relative).read_text()
                    ast.parse(changed, filename=str(relative))
                installed = checkout / apply.FAMILY / "display_kv.py"
                self.assertEqual(installed.read_text(), (ROOT / "overlay" / "display_kv.py").read_text())
                with self.assertRaisesRegex(ValueError, "already installed"):
                    apply.install(checkout)


if __name__ == "__main__":
    unittest.main()
