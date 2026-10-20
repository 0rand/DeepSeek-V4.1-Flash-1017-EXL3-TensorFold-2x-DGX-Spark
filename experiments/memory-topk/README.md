# Memory and top-k experiment

This is an independent image overlay for pinned TensorFold source. It does not
use display KV, SSD spill changes, cache metrics, or tool-output changes.
See [the operator guide](../../docs/MEMORY-TOPK.md) for settings and limits.

The installer exports `REVISION`, `FILES`, `transform(relative, text)`, and
`overlays()`. `FILES` contains only the three edited source paths and their
original SHA-256 hashes. `overlays()` returns relative paths and source text.
The installer checks the revision, pristine bytes, overlay absence, replacement
anchors, and Python ASTs before its first write. It refuses a second install.
Validation errors leave the checkout unchanged. This is not a transaction for
an I/O error during writes. Use a disposable build checkout.

Run the standard-library checks with:

```bash
python3 -m unittest discover -s experiments/memory-topk/tests -v
```

Set `TF_MEMORY_SOURCE` to a pristine checkout at the pinned revision to test the
real source transforms. The test copies only the three input files to a temporary
folder. It does not modify the source checkout. Without this setting, that test
skips. Set `TF_MEMORY_CPU_TESTS=1` to run numerical tests with an installed CPU-only
PyTorch package. CUDA and ROCm builds skip these tests. The tests do not probe or
initialize CUDA. All allocator tests use mocks.

The overlays were extracted from the memory/top-k work in the
`0rand/DeepSeek-V4.1-Flash-1017-EXL3-TensorFold-2x-DGX-Spark` fork, snapshot
`172cf78`. They use this repository's Apache-2.0 licence. The packed-key order
comes from the pinned TensorFold kernel. No fork run logs are included.
