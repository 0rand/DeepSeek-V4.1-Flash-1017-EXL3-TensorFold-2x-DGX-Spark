#!/usr/bin/env python3
"""Install only memory/top-k changes into pristine pinned TensorFold source."""
from __future__ import annotations

import argparse
import ast
import hashlib
from pathlib import Path
import subprocess

REVISION = "07aa02585f64cfb04c60a96de6554c0d4708ed46"
FAMILY = Path("src/tensorfold/families/deepseek_v41/cuda")
FILES = {
    FAMILY / "engine.py": "019f616166d3ed72f117461e4978df37d36c85c28cef9c7f42a51b5a3824fee5",
    FAMILY / "multi.py": "0e19663c2b2f994ef7c160d21fa444624af012c3f756a9265fae5a082c24213d",
    FAMILY / "kernels.py": "1db323f4553f729b5a7281990dc6086fbcf1503921a6bdf30e6218e71d178367",
}
HERE = Path(__file__).resolve().parent


def _replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{name}: expected one unmodified anchor, found {count}")
    return text.replace(old, new)


def transform(relative: str | Path, text: str) -> str:
    """Transform one supported file. The installer checks pristine hashes first."""
    relative = Path(relative)
    if relative not in FILES:
        raise ValueError(f"unsupported source file: {relative}")
    name = relative.name
    if "from .memory_guard import MemoryGuard" in text or 'guard = getattr(self.e, "memory_guard", None)' in text:
        raise ValueError(f"{relative}: memory policy is already installed")
    if name == "kernels.py":
        return _replace_once(text,
            '    bits = (score + 0.0).view(torch.int32)                       # + 0.0: -0 becomes +0\n'
            '    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)\n'
            '    keys = (ordered << 32) | (0xFFFFFFFF - torch.arange(score.shape[-1], device=score.device, dtype=torch.int64))\n'
            '    top = keys.topk(k, dim=-1, sorted=False).values\n'
            '    return (0xFFFFFFFF - (top & 0xFFFFFFFF)).sort(dim=-1).values\n',
            '    from .bounded_topk import topk_indices as bounded_topk\n\n'
            '    return bounded_topk(score, k)\n', name)
    if name == "engine.py":
        # Insert an opt-in branch, rather than replacing the original ceiling.
        # The original body below this branch stays byte-for-byte intact.
        text = _replace_once(text, '        floor = float(os.environ.get("TF_DS_MEM_FLOOR_GIB") or 2.0)\n',
            '        cap = os.environ.get("TF_DS_MEM_CAP", "0")\n'
            '        if cap not in ("0", "1"):\n'
            '            raise ValueError("TF_DS_MEM_CAP must be 0 or 1")\n'
            '        if cap == "1":\n'
            '            from .memory_guard import MemoryGuard\n\n'
            '            self.memory_guard = MemoryGuard(torch, rank=self.rank,\n'
            '                                            floor_gib=float(os.environ.get("TF_DS_MEM_FLOOR_GIB") or 2.0))\n'
            '            self.memory_guard.refresh()\n'
            '            return\n\n'
            '        floor = float(os.environ.get("TF_DS_MEM_FLOOR_GIB") or 2.0)\n', name)
        text = _replace_once(text, '        for s in range(start, len(prompt), PREFILL_CHUNK):\n',
            '        for s in range(start, len(prompt), PREFILL_CHUNK):\n'
            '            guard = getattr(self, "memory_guard", None)\n'
            '            if guard is not None:\n'
            '                guard.refresh()\n', name)
        return _replace_once(text, '            if e < len(prompt):\n                yield e\n',
            '            if guard is not None:\n'
            '                guard.refresh()\n'
            '            if e < len(prompt):\n                yield e\n', name)
    text = _replace_once(text,
        '        why = f"a step failed on rank {self.e.rank} ({type(exc).__name__}: {str(exc)[:160]})"\n',
        '        guard = getattr(self.e, "memory_guard", None)\n'
        '        if guard is not None:\n'
        '            print(f"[tensorfold] rank {self.e.rank}: last allocator budget {guard.last}", flush=True)\n'
        '        why = f"a step failed on rank {self.e.rank} ({type(exc).__name__}: {str(exc)[:160]})"\n', name)
    text = _replace_once(text, '        e = self.e\n        t0 = time.perf_counter()\n        self._step(True)\n',
        '        e = self.e\n'
        '        guard = getattr(e, "memory_guard", None)\n'
        '        if guard is not None:\n'
        '            guard.refresh()\n'
        '        t0 = time.perf_counter()\n        self._step(True)\n', name)
    return _replace_once(text, '            self.round_end = time.perf_counter()\n            return done\n',
        '            if guard is not None:\n'
        '                guard.refresh()\n'
        '            self.round_end = time.perf_counter()\n            return done\n', name)


def overlays() -> dict[Path, str]:
    """Return only this feature's overlay destinations and text."""
    return {FAMILY / name: (HERE / "overlay" / name).read_text()
            for name in ("bounded_topk.py", "memory_guard.py")}


def _git_head(checkout: Path) -> str:
    return subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()


def install(checkout: Path) -> None:
    checkout = checkout.resolve()
    if _git_head(checkout) != REVISION:
        raise ValueError(f"checkout HEAD must be {REVISION}")
    additions = overlays()
    for path in additions:
        if (checkout / path).exists() or (checkout / path).is_symlink():
            raise ValueError(f"{path}: overlay already exists")
    source = {}
    for path, expected in FILES.items():
        raw = (checkout / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{path}: source fingerprint differs from pinned revision")
        source[path] = raw.decode()
    edits = {path: transform(path, text) for path, text in source.items()}
    for path, text in (edits | additions).items():
        ast.parse(text, filename=str(path))
    # No writes until every input, anchor and generated module passes validation.
    for path, text in (edits | additions).items():
        (checkout / path).write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    install(parser.parse_args().checkout)


if __name__ == "__main__":
    main()
