#!/usr/bin/env python3
"""Destructive integrity check of the display mapping. Run only with ranks stopped."""
import argparse
from pathlib import Path
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm-idle", action="store_true", required=True)
    parser.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent / "overlay"))
    import display_kv
    import torch

    torch.cuda.set_device(0)
    torch.zeros(1, device="cuda")
    size = 1792 << 20
    span = display_kv.map_display(size)
    words = display_kv.as_tensor(span).view(torch.int32)
    block = 1 << 18
    offsets = torch.arange(block, dtype=torch.int32, device="cuda")
    expected = torch.empty_like(offsets)
    # Fill the whole reservation first, then verify it to detect aliased pages.
    for salt in (0x5A5A5A5A, 0x25A525A5):
        for start in range(0, words.numel(), block):
            part = words[start:start + block]
            torch.add(offsets[:part.numel()], start, out=expected[:part.numel()])
            torch.bitwise_xor(expected[:part.numel()], salt, out=expected[:part.numel()])
            torch.bitwise_or(expected[:part.numel()], 0, out=part)
        for start in range(0, words.numel(), block):
            part = words[start:start + block]
            torch.add(offsets[:part.numel()], start, out=expected[:part.numel()])
            torch.bitwise_xor(expected[:part.numel()], salt, out=expected[:part.numel()])
            if not (part == expected[:part.numel()]).all().item():
                raise RuntimeError(f"display integrity mismatch at byte {start * 4}")
        torch.cuda.synchronize()
    print(f"PASS: {display_kv.selected_card()}, {size} bytes, two full SM write/read sweeps", flush=True)


if __name__ == "__main__":
    main()
