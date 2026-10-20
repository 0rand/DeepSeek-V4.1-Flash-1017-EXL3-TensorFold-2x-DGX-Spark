#!/usr/bin/env python3
"""Install only display-backed KV into the exact pinned TensorFold source.

No tool-output, memory-policy, telemetry or replacement spill implementation is
installed. Validation and compilation finish before the first source write.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import subprocess

REVISION = "07aa02585f64cfb04c60a96de6554c0d4708ed46"
FAMILY = Path("src/tensorfold/families/deepseek_v41/cuda")
FILES = {
    FAMILY / "model.py": "a5a2d643970856e662a04eabe665994696084d80bbfcd30b9b253ebe18abf843",
    FAMILY / "engine.py": "019f616166d3ed72f117461e4978df37d36c85c28cef9c7f42a51b5a3824fee5",
    FAMILY / "multi.py": "0e19663c2b2f994ef7c160d21fa444624af012c3f756a9265fae5a082c24213d",
}
HERE = Path(__file__).resolve().parent


def _replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{name}: expected one unmodified anchor, found {count}")
    return text.replace(old, new)


def transform(relative: str | Path, text: str) -> str:
    """Transform one source without depending on any other optional feature."""
    relative = Path(relative)
    if relative not in FILES:
        raise ValueError(f"unsupported source file: {relative}")
    name = relative.name
    if "display_arena" in text or "TF_DS_DISPLAY_KV_MIB" in text:
        raise ValueError(f"{relative}: display-KV appears to be installed already")
    if name == "model.py":
        text = _replace_once(text, "from . import kernels as K\n", "from . import display_kv\nfrom . import kernels as K\n", name)
        text = _replace_once(text, "    ring_size: int = 0\n\n\n@dataclass\nclass PoolCache:",
            "    ring_size: int = 0\n    display_arena: object | None = None\n\n\n@dataclass\nclass PoolCache:", name)
        text = _replace_once(text, "    tokens: torch.Tensor | None = None\n\n\ndef _engram_io():",
            "    tokens: torch.Tensor | None = None\n    display_arena: object | None = None\n\n\ndef _engram_io():", name)
        text = _replace_once(text, "    def new_cache(self, cap: int) -> SeqCache:\n        c = self.cfg\n        ring = c.window + RING_EXTRA\n        sc = SeqCache(cap=cap, ring_size=ring)\n",
            "    def new_cache(self, cap: int, display_bytes: int = 0) -> SeqCache:\n        c = self.cfg\n        if display_bytes and not KV_QUANT:\n            raise ValueError(\"display KV requires TF_DS_KV=native packed storage\")\n        arena = display_kv.CacheArena(c, cap, display_bytes) if display_bytes else None\n        if arena is not None:\n            self.display_arena = arena\n        alloc_zeros = arena.zeros if arena is not None else lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=\"cuda\")\n        alloc_full = arena.full if arena is not None else lambda shape, fill, dtype: torch.full(shape, fill, dtype=dtype, device=\"cuda\")\n        ring = c.window + RING_EXTRA\n        sc = SeqCache(cap=cap, ring_size=ring, display_arena=arena)\n", name)
        for shape in ("c.head_dim // 2", "c.head_dim // 16", "c.idx_dim // 2"):
            text = _replace_once(text, f'torch.zeros((rows, {shape}), dtype=torch.uint8, device="cuda")',
                                 f"alloc_zeros((rows, {shape}), torch.uint8)", name)
        text = _replace_once(text, 'torch.full((rows, c.idx_dim // 32), 127, dtype=torch.uint8, device="cuda")',
                            "alloc_full((rows, c.idx_dim // 32), 127, torch.uint8)", name)
        text = _replace_once(text, '    def new_pool(self, slots: int, cap: int) -> "PoolCache":',
                            '    def new_pool(self, slots: int, cap: int, display_bytes: int = 0) -> "PoolCache":', name)
        text = _replace_once(text, "        one = self.new_cache(cap)\n        pool = PoolCache(slots=slots, cap=cap, ring_size=one.ring_size)\n",
                            "        one = self.new_cache(cap, display_bytes=display_bytes)\n        pool = PoolCache(slots=slots, cap=cap, ring_size=one.ring_size, display_arena=one.display_arena)\n", name)
        text = _replace_once(text, "        v.tokens = pool.tokens[base:base + size]\n        return v\n",
                            "        v.tokens = pool.tokens[base:base + size]\n        v.display_arena = pool.display_arena\n        return v\n", name)
        return _replace_once(text, 'return self.rope.cs(self.cfg.compress_ratios[layer] > 0, max(sc.cap, getattr(self, "rope_cap", 0)))',
                            'return self.rope.cs(self.cfg.compress_ratios[layer] > 0,\n                                  getattr(self, "rope_cap", 0) if getattr(self, "display_arena", None) is not None\n                                  else max(sc.cap, getattr(self, "rope_cap", 0)))', name)
    if name == "engine.py":
        text = _replace_once(text, '        self.model_dir = Path(model_dir)                 # (multi\'s spilled kept prompts: part of their engine identity)\n',
            '        self.model_dir = Path(model_dir)                 # (multi\'s spilled kept prompts: part of their engine identity)\n'
            '        from . import display_kv\n'
            '        self.display_bytes = display_kv.enabled_bytes()\n'
            '        if self.display_bytes:\n'
            '            if int(parallel) <= 1:\n'
            '                raise ValueError("display KV requires --parallel greater than 1 before weights load")\n'
            '            if os.environ.get("TF_DS_KV", "native") != "native":\n'
            '                raise ValueError("display KV requires TF_DS_KV=native packed storage")\n'
            '            if float(os.environ.get("TF_DS_SPILL_GIB") or 0) != 0:\n'
            '                raise ValueError("display-only KV requires TF_DS_SPILL_GIB=0")\n', name)
        text = _replace_once(text, "        self.limit = int(context or 65536)\n        self.max_rows = drafts + 1\n", "", name)
        text = _replace_once(text, "        cfg = self.w.cfg\n",
            "        cfg = self.w.cfg\n        self.limit = int(context or 65536)\n        self.max_rows = drafts + 1\n"
            "        self.display_pool_cap = (display_kv.pool_capacity(cfg, self.display_bytes, max_rows=0)\n"
            "                                 if self.display_bytes else 0)\n"
            "        if self.display_bytes and self.display_pool_cap < self.limit + self.max_rows + 8:\n"
            '            raise ValueError("display KV pool is smaller than the requested context plus decode scratch rows")\n', name)
        text = _replace_once(text, "mine = torch.tensor([int(self.vcfg is not None), int(parallel), int(markov.ON), int(markov.SPLIT)],",
            "mine = torch.tensor([int(self.vcfg is not None), int(parallel), int(markov.ON), int(markov.SPLIT),\n                                self.display_bytes, self.display_pool_cap, self.limit],", name)
        text = _replace_once(text, "world * 4", "world * 7", name)
        text = _replace_once(text, "every.view(world, 4)", "every.view(world, 7)", name)
        text = _replace_once(text, "--vision, --parallel, TF_DS_MARKOV and TF_DS_MARKOV_SPLIT must be the same on every ",
                            "--vision, --parallel, TF_DS_MARKOV, TF_DS_MARKOV_SPLIT and display-KV settings must be the same on every ", name)
        text = _replace_once(text, "streams share one window of ",
                            "streams share {'display KV pool' if self.display_bytes else 'one window'} of ", name)
        return _replace_once(text, 'tokens (an extent each)", flush=True)',
                             'tokens{f\'; per-request context {self.limit}\' if self.display_bytes else \' (an extent each)\'}", flush=True)', name)
    text = _replace_once(text, "from . import constrain\n", "from . import constrain\nfrom . import display_kv\n", name)
    text = _replace_once(text, "        self.m, self.cap = m, int(cap)\n        self.pool = m.new_pool(slots, self.cap)\n",
        "        self.m, self.cap = m, int(cap)\n"
        "        self.cap = (display_kv.pool_capacity(m.cfg, engine.display_bytes, max_rows=0)\n"
        "                    if engine.display_bytes else int(cap))\n"
        "        self.pool = m.new_pool(slots, self.cap, display_bytes=engine.display_bytes)\n", name)
    text = _replace_once(text, "                    t[b:b + k].copy_(t[a:a + k].clone() if overlap else t[a:a + k])",
        "                    if self.pool.display_arena is not None:\n                        display_kv.copy_rows(t, a, b, k)\n                    else:\n                        t[b:b + k].copy_(t[a:a + k].clone() if overlap else t[a:a + k])", name)
    return _replace_once(text, "        room = self.extents.total - len(s.prompt) - self.max_rows - 8 - 4\n",
        "        admit_cap = min(self.extents.total, self.e.limit)\n        room = admit_cap - len(s.prompt) - self.max_rows - 8 - 4\n", name)


def overlays() -> dict[Path, str]:
    return {FAMILY / "display_kv.py": (HERE / "overlay/display_kv.py").read_text()}


def _git_head(checkout: Path) -> str:
    return subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()


def install(checkout: Path) -> None:
    checkout = checkout.resolve()
    if _git_head(checkout) != REVISION:
        raise ValueError(f"checkout HEAD must be {REVISION}")
    additions = overlays()
    for destination in additions:
        if (checkout / destination).exists():
            raise ValueError(f"{destination}: display-KV is already installed")
    originals = {path: (checkout / path).read_text() for path in FILES}
    for path, text in originals.items():
        actual = hashlib.sha256(text.encode()).hexdigest()
        if actual != FILES[path]:
            raise ValueError(f"{path}: source fingerprint differs from pinned revision "
                             f"(expected sha256 {FILES[path]}, found sha256 {actual})")
    edits = {path: transform(path, text) for path, text in originals.items()}
    for path, text in (edits | additions).items():
        ast.parse(text, filename=str(path))
    for path, text in (edits | additions).items():
        (checkout / path).write_text(text)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    install(parser.parse_args().checkout)
