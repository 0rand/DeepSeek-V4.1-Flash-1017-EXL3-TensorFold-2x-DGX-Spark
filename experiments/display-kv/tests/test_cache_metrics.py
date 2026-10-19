import importlib.util
import os
import unittest
from types import SimpleNamespace


MODULE = os.path.join(os.path.dirname(__file__), "..", "overlay", "cache_metrics.py")


def load():
    spec = importlib.util.spec_from_file_location("cache_metrics_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class Tensor:
    def __init__(self, count, width):
        self.count, self.width = count, width

    def numel(self):
        return self.count

    def element_size(self):
        return self.width


def engine(*, display=False, gaps=((0, 2_100_000),), decoding=(), filling=(), kept=()):
    comp = {0: (Tensor(10, 1), Tensor(3, 1))}
    index = {0: (Tensor(6, 2), Tensor(1, 2))}
    pool = SimpleNamespace(comp=comp, index_k=index,
                           display_arena=SimpleNamespace(size=1234) if display else None)
    decoder = SimpleNamespace(extents=SimpleNamespace(total=2_100_000, gaps=list(gaps)), pool=pool,
                              streams={index: stream for index, stream in enumerate(decoding)}, filling=list(filling),
                              kept={index: item for index, item in enumerate(kept)}, slots=[None, None, None])
    return SimpleNamespace(multi=decoder, limit=1_000_000)


class CacheMetricsTests(unittest.TestCase):
    def setUp(self):
        self.metrics = load()

    def test_single_sequence_engine_is_unsupported(self):
        self.assertEqual(self.metrics.snapshot(SimpleNamespace()), {})

    def test_idle_pool_and_fragmentation(self):
        result = self.metrics.snapshot(engine(gaps=((0, 500_000), (700_000, 1_000_000))))
        self.assertEqual(result["pool_tokens"], 2_100_000)
        self.assertEqual(result["pool_free_tokens"], 800_000)
        self.assertEqual(result["pool_used_tokens"], 1_300_000)
        self.assertEqual(result["pool_largest_free_tokens"], 500_000)
        self.assertEqual(result["pool_free_extents"], 2)
        self.assertEqual(result["stream_kv_usage_ratio"], 0.0)
        self.assertEqual(result["streams"], {"decoding": 0, "prefilling": 0, "max": 3})

    def test_prefilling_uses_filled_and_decoding_uses_cache_length(self):
        decoding = SimpleNamespace(st=SimpleNamespace(sc=SimpleNamespace(length=1_200_000)), context=[])
        filling = SimpleNamespace(filled=500_000, context=[0] * 2)
        result = self.metrics.snapshot(engine(decoding=(decoding,), filling=(filling,)))
        self.assertEqual(result["streams"], {"decoding": 1, "prefilling": 1, "max": 3})
        self.assertEqual(result["stream_kv_usage_ratio"], 1.0)

    def test_decoding_falls_back_to_context_without_inventing_missing_values(self):
        fallback = SimpleNamespace(st=SimpleNamespace(sc=SimpleNamespace()), context=[0] * 500_000)
        unknown = SimpleNamespace(st=None)
        result = self.metrics.snapshot(engine(decoding=(fallback, unknown)))
        self.assertEqual(result["stream_kv_usage_ratio"], 0.5)

    def test_retained_snapshot_bytes_are_unique_and_exclude_pool_kv(self):
        shared = Tensor(7, 4)
        other = Tensor(3, 8)
        kept = (SimpleNamespace(size=2048, snaps={1: (shared, None), 2: {"raw": [shared, other]}}),
                SimpleNamespace(size=4096, snaps={3: (shared,)}))
        result = self.metrics.snapshot(engine(kept=kept))
        self.assertEqual(result["kept_prompts"], 2)
        self.assertEqual(result["pool_kept_tokens"], 6144)
        self.assertEqual(result["kept_bytes"], 52)
        self.assertEqual(result["kv_cache_payload_bytes"], 27)

    def test_cuda_and_display_backings_are_reported_separately(self):
        cuda = self.metrics.snapshot(engine())
        display = self.metrics.snapshot(engine(display=True))
        self.assertEqual(cuda["kv_cache_backing"], "cuda")
        self.assertEqual(cuda["kv_cache_payload_bytes"], 27)
        self.assertEqual(cuda["kv_cache_ordinary_bytes"], 27)
        self.assertEqual(cuda["kv_cache_capacity_bytes"], 27)
        self.assertEqual(display["kv_cache_backing"], "display")
        self.assertEqual(display["kv_cache_display_bytes"], 1234)
        self.assertEqual(display["kv_cache_ordinary_bytes"], 0)
        self.assertEqual(display["kv_cache_capacity_bytes"], 1234)
        self.assertEqual(display["kv_cache_bytes_scope"], "compressed_kv_and_indexer_per_rank")

    def test_nonpacked_cache_bytes_are_not_silently_omitted(self):
        value = engine()
        value.multi.pool.comp = {0: Tensor(100, 2)}
        value.multi.pool.index_k = {0: Tensor(20, 2)}
        self.assertEqual(self.metrics.snapshot(value)["kv_cache_payload_bytes"], 240)


if __name__ == "__main__":
    unittest.main()
