# Attribution

The display-memory mapping in `overlay/display_kv.py` is adapted from
Gilesco's `0072-glm-display-kv.patch`:
https://github.com/gilesco/GLM-5.3-Flash-EXL3-2x-DGX-Sparks-TensorFold/blob/main/patches/0072-glm-display-kv.patch

The allocation technique is also described in
coolbho3k's DeepSeek display-reserve allocator:
https://github.com/coolbho3k/DeepSeek-v4.1-Flash-2x-DGX-Spark

The local Gilesco recipe and this recipe carry Apache-2.0 licenses. The separate
NVFP4 C allocator/toolkit carries AGPL notices; its binaries and C source are
not included in this overlay. Retain applicable upstream copyright and license
notices when distributing derived code or an image, and review any additional
toolkit copied into it. See the repository's `LICENSE` for this recipe.

The TensorFold source installer targets the exact upstream revision
`07aa02585f64cfb04c60a96de6554c0d4708ed46` and preserves its source headers.

The native tool-output validator extends this recipe's existing strict guard
and installer hooks, with CPU-reviewed atomic protocol/schema validation and
bounded evaluation. Its recognized wrapper formats follow TensorFold's pinned
native tool-call path. It remains readable source in `overlay/tool_output.py`,
not a replacement engine source archive. The input pin and fingerprints describe
upstream inputs only; installed engine files are modified derivatives. This
hardening does not import the separate allocator/toolkit or change its licensing.
