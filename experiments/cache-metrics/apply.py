#!/usr/bin/env python3
"""Install opt-in cache metrics into the pristine pinned TensorFold source."""
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
    Path("src/tensorfold/cuda/health.py"): "3f5befbbef7764c420209a9f1b29f2d2d28ea98af51d8e5b31e92f131f2c8851",
    Path("src/tensorfold/server/metrics.py"): "a3b7956c3b82513006c6631b58df9aac8516cec140984b28c77bf53ed2036881",
}
HERE = Path(__file__).resolve().parent


def _replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{name}: expected one unmodified anchor, found {count}")
    return text.replace(old, new)


GAUGES = (
    ("pool_tokens", "Positions in the shared KV pool on this rank."),
    ("pool_free_tokens", "Pool positions not reserved by streams or kept prompts."),
    ("pool_used_tokens", "Pool positions reserved by streams or kept prompts."),
    ("pool_largest_free_tokens", "Largest contiguous free extent in positions."),
    ("pool_free_extents", "Number of free extents."),
    ("pool_kept_tokens", "Pool positions reserved for retained prompts."),
    ("kept_prompts", "Retained prompts in memory."),
    ("kept_bytes", "Unique retained snapshot tensor bytes outside the shared pool."),
    ("kv_cache_payload_bytes", "Compressed KV and indexer tensor bytes on this rank."),
    ("kv_cache_capacity_bytes", "Backing capacity for compressed KV and indexer tensors on this rank."),
    ("kv_cache_display_bytes", "Display reservation mapped for KV on this rank."),
    ("kv_cache_ordinary_bytes", "Ordinary CUDA backing for compressed KV and indexer tensors on this rank."),
)


def transform(relative: Path | str, text: str) -> str:
    """Transform one supported path, without reading or writing the checkout."""
    relative = Path(relative)
    if relative not in FILES:
        raise ValueError(f"unsupported source file: {relative}")
    name = str(relative)
    if "def cache_metrics_snapshot(" in text or "def _cache_snapshot(" in text or "body.update(reader())" in text:
        raise ValueError(f"{relative}: cache metrics already installed")
    if relative == FAMILY / "engine.py":
        return _replace_once(text, "    def _memory_ceiling(self) -> None:\n",
            '    def cache_metrics_snapshot(self) -> dict:\n'
            '        """Read local metadata without allocating or synchronizing the GPU."""\n'
            '        from .cache_metrics import snapshot\n\n'
            '        return snapshot(self)\n\n'
            '    def _memory_ceiling(self) -> None:\n', name)
    if relative.name == "health.py":
        return _replace_once(text, "        return body\n",
            '        reader = getattr(getattr(app, "engine", None), "cache_metrics_snapshot", None)\n'
            '        if callable(reader):\n            body.update(reader())\n'
            '        return body\n', name)
    text = _replace_once(text, "    pools = _pools(app)\n",
                         "    cache = _cache_snapshot(app)\n    pools = _pools(app, cache)\n", name)
    block = f'''    for key, help_text in {GAUGES!r}:
        if key in cache:
            _family(lines, key, "gauge", help_text, [f"{{PREFIX}}{{key}} {{cache[key]}}"])
    if cache.get("pool_tokens", 0) > 0:
        ratio = cache["pool_used_tokens"] / cache["pool_tokens"]
        _family(lines, "kv_pool_usage_ratio", "gauge", "Reserved shared pool positions divided by total positions.",
                [f"{{PREFIX}}kv_pool_usage_ratio {{_num(ratio)}}"])
    if "kv_cache_backing" in cache:
        backing = cache["kv_cache_backing"]
        _family(lines, "kv_cache_backing_info", "gauge", "Backing for this rank's compressed KV and indexer tensors.",
                [f'{{PREFIX}}kv_cache_backing_info{{{{backing="{{backing}}"}}}} 1'])
'''
    text = _replace_once(text, '    _family(lines, "mtp_drafted_total",',
                         block + '    _family(lines, "mtp_drafted_total",', name)
    text = _replace_once(text,
        '            "Tokens in a stream cache divided by that stream\'s context window.",',
        '            "Live stream cache / request context; DeepSeek reports the maximum live stream ratio.",', name)
    text = _replace_once(text, 'def _pools(app: Any) -> list[tuple[str, float]]:',
        '''def _cache_snapshot(app: Any) -> dict:
    reader = getattr(getattr(app, "engine", None), "cache_metrics_snapshot", None)
    return reader() if callable(reader) else {}


def _pools(app: Any, cache: dict | None = None) -> list[tuple[str, float]]:''', name)
    return _replace_once(text, '    window = _window(app)\n',
        '''    cache = _cache_snapshot(app) if cache is None else cache
    if "stream_kv_usage_ratio" in cache:
        return [("0", cache["stream_kv_usage_ratio"])]
    window = _window(app)
''', name)


def overlays() -> dict[Path, str]:
    """Return relative target paths and contents for external composition."""
    return {FAMILY / "cache_metrics.py": (HERE / "overlay/cache_metrics.py").read_text()}


def install(checkout: Path) -> None:
    checkout = Path(checkout).resolve()
    revision = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"checkout HEAD must be {REVISION}")
    additions = overlays()
    for path in additions:
        target = checkout / path
        if target.exists() or target.is_symlink():
            raise ValueError(f"{path}: overlay already exists")
    source = {}
    for path, expected in FILES.items():
        raw = (checkout / path).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected:
            raise ValueError(f"{path}: source fingerprint differs from pinned revision")
        source[path] = raw.decode()
    edits = {path: transform(path, text) for path, text in source.items()}
    for path, text in {**edits, **additions}.items():
        ast.parse(text, filename=str(path))
    # No writes occur before all source, overlay and syntax checks pass.
    for path, text in {**edits, **additions}.items():
        (checkout / path).write_text(text)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkout", type=Path)
    install(parser.parse_args().checkout)


if __name__ == "__main__":
    main()
