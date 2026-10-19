import importlib.util
from pathlib import Path
import unittest


try:
    import torch
except ImportError:
    torch = None


MODULE = Path(__file__).parent.parent / "overlay" / "bounded_topk.py"


def load():
    spec = importlib.util.spec_from_file_location("bounded_topk_test", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(score, k):
    bits = (score + 0.0).view(torch.int32)
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)
    keys = (ordered << 32) | (0xFFFFFFFF - torch.arange(score.shape[-1], device=score.device, dtype=torch.int64))
    top = keys.topk(k, dim=-1, sorted=False).values
    return (0xFFFFFFFF - (top & 0xFFFFFFFF)).sort(dim=-1).values


@unittest.skipIf(torch is None, "torch is not installed")
class BoundedTopkTests(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def assert_matches(self, score, k):
        self.assertTrue(torch.equal(self.module.topk_indices(score, k), reference(score, k)))

    def test_random_cpu_equivalence(self):
        generator = torch.Generator().manual_seed(17)
        score = torch.randn((11, 257), generator=generator, dtype=torch.float32)
        for k in (0, 1, 7, 257):
            self.assert_matches(score, k)

    def test_special_values_and_lower_index_ties(self):
        score = torch.tensor([
            [float("nan"), float("inf"), float("inf"), 1.0, 1.0, 0.0, -0.0, float("-inf")],
            [float("-inf"), -3.0, -3.0, -0.0, 0.0, float("inf"), float("nan"), float("nan")],
        ], dtype=torch.float32)
        for k in range(score.shape[-1] + 1):
            self.assert_matches(score, k)

    def test_extra_leading_dimensions(self):
        score = torch.tensor([[[1.0, 1.0, -0.0, 0.0], [float("nan"), -1.0, float("inf"), -1.0]]])
        self.assert_matches(score, 3)

    def test_wide_rows_tile_under_patched_scratch_limit(self):
        old = self.module.KEY_SCRATCH_BYTES
        self.module.KEY_SCRATCH_BYTES = 128
        try:
            score = torch.arange(5 * 33, dtype=torch.float32).reshape(5, 33)
            self.assert_matches(score, 9)
        finally:
            self.module.KEY_SCRATCH_BYTES = old

    def test_invalid_k(self):
        for k in (-1, 5):
            with self.assertRaises(ValueError):
                self.module.topk_indices(torch.empty((0, 4)), k)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_equivalence(self):
        score = torch.tensor([[float("nan"), -0.0, 0.0, float("inf"), -4.0, -4.0]], device="cuda")
        self.assert_matches(score, 4)

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_long_context_peak_is_reduced(self):
        score = torch.randn((512, 32768), device="cuda")
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        expected = reference(score, 256)
        torch.cuda.synchronize()
        original_peak = torch.cuda.max_memory_allocated() - baseline
        torch.cuda.reset_peak_memory_stats()
        baseline = torch.cuda.memory_allocated()
        actual = self.module.topk_indices(score, 256)
        torch.cuda.synchronize()
        bounded_peak = torch.cuda.max_memory_allocated() - baseline
        self.assertTrue(torch.equal(actual, expected))
        self.assertLess(bounded_peak, original_peak // 2)
        print(f"top-k CUDA scratch peak: {original_peak / 2**20:.1f} -> {bounded_peak / 2**20:.1f} MiB")

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_graph_capture(self):
        score = torch.randn((4, 257), device="cuda")
        self.module.topk_indices(score, 7)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = self.module.topk_indices(score, 7)
        graph.replay()
        self.assertTrue(torch.equal(actual, reference(score, 7)))
