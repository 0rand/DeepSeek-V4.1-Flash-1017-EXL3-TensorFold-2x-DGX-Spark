"""Optional CPU-only numerical checks. No CUDA probe or initialization."""
import importlib.util
import os
from pathlib import Path
import unittest


torch = None
if os.environ.get("TF_MEMORY_CPU_TESTS") == "1" and importlib.util.find_spec("torch"):
    import torch as candidate
    if candidate.version.cuda is None and candidate.version.hip is None:
        torch = candidate


def load():
    path = Path(__file__).resolve().parents[1] / "overlay/bounded_topk.py"
    spec = importlib.util.spec_from_file_location("bounded_topk_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reference(score, k):
    bits = (score + 0.0).view(torch.int32)
    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)
    keys = (ordered << 32) | (0xFFFFFFFF - torch.arange(score.shape[-1], device=score.device, dtype=torch.int64))
    top = keys.topk(k, dim=-1, sorted=False).values
    return (0xFFFFFFFF - (top & 0xFFFFFFFF)).sort(dim=-1).values


@unittest.skipUnless(torch is not None, "set TF_MEMORY_CPU_TESTS=1 with CPU-only torch installed")
class BoundedTopkTests(unittest.TestCase):
    def setUp(self):
        self.module = load()

    def assert_matches(self, score, k):
        original = score.clone()
        actual = self.module.topk_indices(score, k)
        self.assertTrue(torch.equal(actual, reference(score, k)))
        self.assertEqual(actual.dtype, torch.int64)
        self.assertEqual(actual.shape, (*score.shape[:-1], k))
        self.assertTrue(torch.equal(score.view(torch.int32), original.view(torch.int32)))

    def test_random_and_tile_boundaries(self):
        generator = torch.Generator().manual_seed(17)
        for budget in (1, 128, 264, 528, 16 << 20):
            self.module.KEY_SCRATCH_BYTES = budget
            for rows in (0, 1, 2, 3, 5, 11):
                for width in (1, 4, 33, 257):
                    score = torch.randn((rows, width), generator=generator, dtype=torch.float32)
                    for k in sorted({0, 1, width // 2, width}):
                        self.assert_matches(score, k)

    def test_special_values_and_lower_index_ties(self):
        score = torch.tensor([
            [float("nan"), float("inf"), float("inf"), 1.0, 1.0, 0.0, -0.0, float("-inf")],
            [float("-inf"), -3.0, -3.0, -0.0, 0.0, float("inf"), float("nan"), float("nan")],
        ], dtype=torch.float32)
        for k in range(score.shape[-1] + 1):
            self.assert_matches(score, k)
        self.assertTrue(torch.equal(self.module.topk_indices(torch.ones((2, 9)), 3),
                                    torch.tensor([[0, 1, 2], [0, 1, 2]])))

    def test_leading_dimensions_and_noncontiguous_input(self):
        self.assert_matches(torch.arange(24, dtype=torch.float32).reshape(2, 3, 4), 3)
        self.assert_matches(torch.arange(24, dtype=torch.float32).reshape(4, 6).t(), 2)
        self.assert_matches(torch.tensor([1.0, 1.0, -0.0, 0.0]), 2)
        self.assertEqual(self.module.topk_indices(torch.empty(3, 0), 0).shape, (3, 0))

    def test_invalid_k_dimension_and_dtype(self):
        for k in (-1, 5):
            with self.assertRaises(ValueError):
                self.module.topk_indices(torch.empty((0, 4)), k)
        with self.assertRaises(TypeError):
            self.module.topk_indices(torch.zeros(4), 1.5)
        with self.assertRaises(ValueError):
            self.module.topk_indices(torch.tensor(1.0), 0)
        for dtype in (torch.float16, torch.bfloat16, torch.float64, torch.int32):
            with self.assertRaises(TypeError):
                self.module.topk_indices(torch.ones(4, dtype=dtype), 2)
