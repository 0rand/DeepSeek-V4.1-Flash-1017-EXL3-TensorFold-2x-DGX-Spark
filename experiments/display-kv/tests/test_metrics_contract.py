"""CPU-only contract checks for display-KV cache telemetry and its dashboard adapter."""
from __future__ import annotations

import importlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import types
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
SOURCE = Path(os.environ.get("TF_DISPLAY_SOURCE", "/tmp/opencode/tfv41-display-source"))
# A checkout of EngineMetrics (set ENGINE_METRICS_DIR); the adapter test skips itself without it.
ENGINE_METRICS = Path(os.environ.get("ENGINE_METRICS_DIR") or "/nonexistent/EngineMetrics")


def load_file(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


apply = load_file("display_kv_metrics_apply", ROOT / "apply.py")
cache_metrics = load_file("display_kv_cache_metrics", ROOT / "overlay" / "cache_metrics.py")


class Tensor:
    def __init__(self, count: int, width: int) -> None:
        self.count, self.width = count, width

    def numel(self) -> int:
        return self.count

    def element_size(self) -> int:
        return self.width


def fake_engine(*, display: bool = True, live: bool = True, spill: bool = False):
    shared = Tensor(7, 4)
    kept = types.SimpleNamespace(size=65_536, snaps={"a": (shared, Tensor(3, 8)), "b": [shared]})
    pool = types.SimpleNamespace(
        comp={0: (Tensor(10, 1), Tensor(3, 1))},
        index_k={0: (Tensor(6, 2), Tensor(1, 2))},
        display_arena=types.SimpleNamespace(size=1792 * 2**20) if display else None,
    )
    decoding = {}
    filling = []
    if live:
        decoding["decode"] = types.SimpleNamespace(st=types.SimpleNamespace(sc=types.SimpleNamespace(length=524_288)))
        # The prompt is deliberately larger: telemetry must use partial prefill progress.
        filling.append(types.SimpleNamespace(filled=262_144, prompt=[0] * 1_048_576))
    spill_info = {
        "entries": 3, "pending": 1, "stored_bytes": 1234, "written": 2, "written_bytes": 1000,
        "loaded": 1, "loaded_bytes": 500, "loaded_tokens": 64,
        "restore_ms_p50": 12.5, "restore_ms_p95": 20.0,
        "write_failed": 0, "crc_failed": 0, "load_failed": 0, "load_unfit": 0,
        "disk_free_bytes": 9999, "highwater": 0.7, "writers": 2, "direct": True,
        "stored_gib": 0.0, "written_gib": 0.0, "loaded_gib": 0.0,
    }
    multi = types.SimpleNamespace(
        extents=types.SimpleNamespace(total=2_109_440, gaps=[(0, 500_000), (700_000, 1_000_000)]),
        streams=decoding,
        filling=filling,
        kept={"retained": kept},
        slots=[None] * 4,
        pool=pool,
        spill=types.SimpleNamespace(info=lambda: spill_info) if spill else None,
    )
    scheduler = types.SimpleNamespace(decoder=multi, max_streams=4)
    engine = types.SimpleNamespace(multi=multi, scheduler=scheduler, limit=1_048_576)
    engine.cache_metrics_snapshot = lambda: cache_metrics.snapshot(engine)
    return engine


@unittest.skipUnless(SOURCE.is_dir(), "needs TF_DISPLAY_SOURCE: a checkout of the pinned TensorFold revision (docs/DISPLAY-KV.md, Offline tests)")
class MetricsContractTests(unittest.TestCase):
    def load_transformed_modules(self):
        metrics_source = apply.transform("src/tensorfold/server/metrics.py",
                                         (SOURCE / "src/tensorfold/server/metrics.py").read_text())
        health_source = apply.transform("src/tensorfold/cuda/health.py",
                                        (SOURCE / "src/tensorfold/cuda/health.py").read_text())
        compile(metrics_source, "metrics.py", "exec")
        compile(health_source, "health.py", "exec")
        before = set(sys.modules)
        saved = {name: sys.modules.pop(name) for name in list(sys.modules) if name.startswith("tensorfold")}
        sys.path.insert(0, str(SOURCE / "src"))
        try:
            metrics = types.ModuleType("tensorfold.server.metrics")
            metrics.__file__ = "metrics.py"
            sys.modules[metrics.__name__] = metrics
            exec(compile(metrics_source, metrics.__file__, "exec"), metrics.__dict__)
            health = types.ModuleType("tensorfold.cuda.health")
            health.__file__ = "health.py"
            sys.modules[health.__name__] = health
            exec(compile(health_source, health.__file__, "exec"), health.__dict__)
            self.assertFalse(any(name == "torch" or name.startswith("torch.") for name in set(sys.modules) - before))
            return metrics, health
        finally:
            sys.path.pop(0)
            for name in list(sys.modules):
                if name.startswith("tensorfold"):
                    del sys.modules[name]
            sys.modules.update(saved)

    def test_snapshot_and_render_report_display_cache_without_confusing_reservations_and_streams(self):
        metrics, _health = self.load_transformed_modules()
        engine = fake_engine()
        snapshot = engine.cache_metrics_snapshot()
        self.assertEqual(snapshot["pool_tokens"], 2_109_440)
        self.assertEqual(snapshot["pool_free_tokens"], 800_000)
        self.assertEqual(snapshot["pool_used_tokens"], 1_309_440)
        self.assertEqual(snapshot["pool_kept_tokens"], 65_536)
        self.assertEqual(snapshot["kept_bytes"], 52)
        self.assertEqual(snapshot["streams"], {"decoding": 1, "prefilling": 1, "max": 4})
        self.assertEqual(snapshot["stream_kv_usage_ratio"], 0.5)
        self.assertEqual(snapshot["kv_cache_backing"], "display")
        self.assertEqual(snapshot["kv_cache_display_bytes"], 1792 * 2**20)
        self.assertEqual(snapshot["kv_cache_ordinary_bytes"], 0)

        body = metrics.render(types.SimpleNamespace(engine=engine))
        for name in ("pool_tokens", "pool_free_tokens", "pool_kept_tokens", "kv_cache_payload_bytes",
                     "kv_cache_capacity_bytes", "kv_cache_display_bytes", "kv_cache_ordinary_bytes"):
            self.assertIn(f"tensorfold:{name} ", body)
        self.assertIn('tensorfold:kv_cache_backing_info{backing="display"} 1', body)
        self.assertIn('tensorfold:kv_cache_usage_ratio{pool="0"} 0.5', body)
        self.assertIn("tensorfold:kv_pool_usage_ratio 0.620752", body)

    def test_idle_and_ordinary_backing_keep_stream_and_pool_occupancy_distinct(self):
        metrics, _health = self.load_transformed_modules()
        engine = fake_engine(display=False, live=False)
        snapshot = engine.cache_metrics_snapshot()
        self.assertGreater(snapshot["pool_used_tokens"], 0)
        self.assertEqual(snapshot["stream_kv_usage_ratio"], 0.0)
        self.assertEqual(snapshot["kv_cache_backing"], "cuda")
        self.assertEqual(snapshot["kv_cache_display_bytes"], 0)
        self.assertEqual(snapshot["kv_cache_ordinary_bytes"], snapshot["kv_cache_payload_bytes"])
        body = metrics.render(types.SimpleNamespace(engine=engine))
        self.assertIn('tensorfold:kv_cache_usage_ratio{pool="0"} 0', body)
        self.assertIn('tensorfold:kv_cache_backing_info{backing="cuda"} 1', body)

    def test_spill_info_reaches_health_and_prometheus_gauges(self):
        metrics, health = self.load_transformed_modules()
        engine = fake_engine(spill=True)
        app = types.SimpleNamespace(engine=engine, effective_context_window=engine.limit)
        tracker = health.Health()
        body = tracker.snapshot(app)
        self.assertIn("spill", body)
        self.assertEqual(body["spill"]["entries"], 3)
        self.assertEqual(body["spill"]["stored_bytes"], 1234)
        self.assertEqual(body["spill"]["restore_ms_p50"], 12.5)

        rendered = metrics.render(app)
        self.assertIn("tensorfold:spill_entries 3", rendered)
        self.assertIn("tensorfold:spill_stored_bytes 1234", rendered)
        self.assertIn("tensorfold:spill_disk_free_bytes 9999", rendered)
        self.assertIn("tensorfold:spill_restore_seconds_p50 0.0125", rendered)
        self.assertIn("tensorfold:spill_written_total 2", rendered)
        self.assertIn("tensorfold:spill_loaded_tokens_total 64", rendered)
        self.assertNotIn("tensorfold:spill_stored_gib", rendered)

        plain = metrics.render(types.SimpleNamespace(engine=fake_engine(spill=False)))
        self.assertNotIn("tensorfold:spill_", plain)

    def test_health_preserves_request_counters_and_metrics_without_hook_stays_unfabricated(self):
        metrics, health = self.load_transformed_modules()
        engine = fake_engine()
        app = types.SimpleNamespace(engine=engine, effective_context_window=engine.limit)
        tracker = health.Health()
        tracker.totals.update(requests_total=3, prompt_tokens_total=99, completion_tokens_total=7)
        result = tracker.snapshot(app)
        self.assertEqual(result["requests_total"], 3)
        self.assertEqual(result["prompt_tokens_total"], 99)
        self.assertEqual(result["completion_tokens_total"], 7)
        self.assertEqual(result["pool_tokens"], 2_109_440)
        self.assertEqual(result["streams"], {"decoding": 1, "prefilling": 1, "max": 4})

        body = metrics.render(types.SimpleNamespace(engine=types.SimpleNamespace(context_window=100)))
        self.assertIn('tensorfold:kv_cache_usage_ratio{pool="0"} 0', body)
        self.assertNotIn("tensorfold:pool_tokens", body)
        self.assertNotIn("tensorfold:kv_cache_payload_bytes", body)
        self.assertNotIn("tensorfold:kv_cache_backing_info", body)

    @unittest.skipUnless((ENGINE_METRICS / "engines" / "tensorfold.py").is_file(), "EngineMetrics adapter is absent")
    def test_actual_enginemetrics_adapter_consumes_rendered_metrics_and_health(self):
        metrics, health = self.load_transformed_modules()
        engine = fake_engine()
        app = types.SimpleNamespace(engine=engine, effective_context_window=engine.limit)
        tracker = health.Health()
        tracker.totals.update(requests_total=4, prompt_tokens_total=12, completion_tokens_total=8,
                              cached_tokens_total=3, rounds_total=2, drafted_total=5, accepted_total=4,
                              prefill_seconds_total=1.5, decode_seconds_total=0.5)
        health_body = tracker.snapshot(app)
        rendered = metrics.render(app)

        inserted = str(ENGINE_METRICS) not in sys.path
        if inserted:
            sys.path.insert(0, str(ENGINE_METRICS))
        try:
            adapter = importlib.import_module("engines.tensorfold")
            def http_get(url, timeout=5):
                return rendered if url.endswith("/metrics") else json.dumps(health_body)
            with mock.patch("engines.common.http_get", side_effect=http_get):
                sample = adapter.sample("unused", 0)
                curmetric, _model = adapter.fetch_metrics("unused", 0)
        finally:
            if inserted:
                sys.path.remove(str(ENGINE_METRICS))

        for key in adapter.HEALTH_COUNTERS:
            self.assertEqual(sample[key], float(health_body.get(key, 0) or 0), key)
        self.assertEqual(sample["kept_bytes"], 52.0)
        self.assertEqual(sample["streams"], {"decoding": 1, "prefilling": 1, "max": 4})
        self.assertEqual(adapter.get_value(curmetric, "tensorfold:kv_cache_usage_ratio"), 0.5)
        self.assertNotIn("nan", rendered.lower())


if __name__ == "__main__":
    unittest.main()
