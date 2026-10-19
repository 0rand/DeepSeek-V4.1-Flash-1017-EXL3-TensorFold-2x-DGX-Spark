#!/usr/bin/env python3
"""Install the display-KV experiment into the pinned TensorFold checkout.

This is deliberately an offline, fail-closed source transformer.  It only
accepts the exact upstream revision and refuses both drift and re-installation.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import subprocess
from pathlib import Path


REVISION = "07aa02585f64cfb04c60a96de6554c0d4708ed46"
FAMILY = Path("src/tensorfold/families/deepseek_v41/cuda")
FILES = {
    FAMILY / "model.py": "a5a2d643970856e662a04eabe665994696084d80bbfcd30b9b253ebe18abf843",
    FAMILY / "engine.py": "019f616166d3ed72f117461e4978df37d36c85c28cef9c7f42a51b5a3824fee5",
    FAMILY / "multi.py": "0e19663c2b2f994ef7c160d21fa444624af012c3f756a9265fae5a082c24213d",
    Path("src/tensorfold/cuda/health.py"): "3f5befbbef7764c420209a9f1b29f2d2d28ea98af51d8e5b31e92f131f2c8851",
    Path("src/tensorfold/server/metrics.py"): "a3b7956c3b82513006c6631b58df9aac8516cec140984b28c77bf53ed2036881",
    FAMILY / "app.py": "426b9463b5a254d96895a8c38f2f7cb5cc6c949bc8b607326e09af3a28d1de33",
    Path("src/tensorfold/cuda/server.py"): "933dbd468f057a1a9f592171ee6f7bdc823231237b1fcc4962ac671cf727e0a9",
    Path("src/tensorfold/cuda/http.py"): "bf8cb5e428c372358ff1e8676f3f92a6f354f1dc35999adfd8487cf6ff1aa1cb",
}
HERE = Path(__file__).resolve().parent
OVERLAY = HERE / "overlay" / "display_kv.py"


def _replace_once(text: str, old: str, new: str, name: str) -> str:
    count = text.count(old)
    if count != 1:
        raise ValueError(f"{name}: expected one unmodified anchor, found {count}")
    return text.replace(old, new)


def transform(relative: str | Path, text: str) -> str:
    """Return the exact display-KV edit for one pristine source file."""
    name = Path(relative).name
    if "display_arena" in text or "TF_DS_DISPLAY_KV_MIB" in text:
        raise ValueError(f"{relative}: display-KV appears to be installed already")
    if name == "model.py":
        text = _replace_once(text, "from . import kernels as K\n", "from . import display_kv\nfrom . import kernels as K\n", name)
        text = _replace_once(text, "    ring_size: int = 0\n\n\n@dataclass\nclass PoolCache:",
            "    ring_size: int = 0\n    display_arena: object | None = None\n\n\n@dataclass\nclass PoolCache:", name)
        text = _replace_once(text, "    tokens: torch.Tensor | None = None\n\n\ndef _engram_io():",
            "    tokens: torch.Tensor | None = None\n    display_arena: object | None = None\n\n\ndef _engram_io():", name)
        text = _replace_once(text, "    def new_cache(self, cap: int) -> SeqCache:\n        c = self.cfg\n        ring = c.window + RING_EXTRA\n        sc = SeqCache(cap=cap, ring_size=ring)\n",
            "    def new_cache(self, cap: int, display_bytes: int = 0) -> SeqCache:\n        c = self.cfg\n        if display_bytes and not KV_QUANT:\n            raise ValueError(\"display KV requires TF_DS_KV=native packed storage\")\n        arena = display_kv.CacheArena(c, cap, display_bytes) if display_bytes else None\n        alloc_zeros = arena.zeros if arena is not None else lambda shape, dtype: torch.zeros(shape, dtype=dtype, device=\"cuda\")\n        alloc_full = arena.full if arena is not None else lambda shape, fill, dtype: torch.full(shape, fill, dtype=dtype, device=\"cuda\")\n        ring = c.window + RING_EXTRA\n        sc = SeqCache(cap=cap, ring_size=ring, display_arena=arena)\n", name)
        text = _replace_once(text, "arena = display_kv.CacheArena(c, cap, display_bytes) if display_bytes else None\n        alloc_zeros",
            "arena = display_kv.CacheArena(c, cap, display_bytes) if display_bytes else None\n        if arena is not None:\n            self.display_arena = arena\n        alloc_zeros", name)
        text = _replace_once(text, "torch.zeros((rows, c.head_dim // 2), dtype=torch.uint8, device=\"cuda\")",
            "alloc_zeros((rows, c.head_dim // 2), torch.uint8)", name)
        text = _replace_once(text, "torch.zeros((rows, c.head_dim // 16), dtype=torch.uint8, device=\"cuda\")",
            "alloc_zeros((rows, c.head_dim // 16), torch.uint8)", name)
        text = _replace_once(text, "torch.zeros((rows, c.idx_dim // 2), dtype=torch.uint8, device=\"cuda\")",
            "alloc_zeros((rows, c.idx_dim // 2), torch.uint8)", name)
        text = _replace_once(text, "torch.full((rows, c.idx_dim // 32), 127, dtype=torch.uint8, device=\"cuda\")",
            "alloc_full((rows, c.idx_dim // 32), 127, torch.uint8)", name)
        text = _replace_once(text, "    def new_pool(self, slots: int, cap: int) -> \"PoolCache\":", "    def new_pool(self, slots: int, cap: int, display_bytes: int = 0) -> \"PoolCache\":", name)
        text = _replace_once(text, "        one = self.new_cache(cap)\n        pool = PoolCache(slots=slots, cap=cap, ring_size=one.ring_size)\n",
            "        one = self.new_cache(cap, display_bytes=display_bytes)\n        pool = PoolCache(slots=slots, cap=cap, ring_size=one.ring_size, display_arena=one.display_arena)\n", name)
        text = _replace_once(text, "        v.tokens = pool.tokens[base:base + size]\n        return v\n",
            "        v.tokens = pool.tokens[base:base + size]\n        v.display_arena = pool.display_arena\n        return v\n", name)
        text = _replace_once(text, "return self.rope.cs(self.cfg.compress_ratios[layer] > 0, max(sc.cap, getattr(self, \"rope_cap\", 0)))",
            "return self.rope.cs(self.cfg.compress_ratios[layer] > 0,\n                                  getattr(self, \"rope_cap\", 0) if getattr(self, \"display_arena\", None) is not None\n                                  else max(sc.cap, getattr(self, \"rope_cap\", 0)))", name)
        return text
    if name == "engine.py":
        text = _replace_once(text, "    def _memory_ceiling(self) -> None:\n",
            '    def cache_metrics_snapshot(self) -> dict:\n        """Read local cache metadata without allocating or synchronizing the GPU."""\n        from .cache_metrics import snapshot\n\n        return snapshot(self)\n\n    def _memory_ceiling(self) -> None:\n', name)
        text = _replace_once(text, "        self.model_dir = Path(model_dir)                 # (multi's spilled kept prompts: part of their engine identity)\n",
            "        self.model_dir = Path(model_dir)                 # (multi's spilled kept prompts: part of their engine identity)\n        from . import display_kv\n        self.display_bytes = display_kv.enabled_bytes()\n        if self.display_bytes:\n            if int(parallel) <= 1:\n                raise ValueError(\"display KV requires --parallel greater than 1 before weights load\")\n            if os.environ.get(\"TF_DS_KV\", \"native\") != \"native\":\n                raise ValueError(\"display KV requires TF_DS_KV=native packed storage\")\n            if float(os.environ.get(\"TF_DS_SPILL_GIB\") or 0) > 0:\n                raise ValueError(\"display-only KV staging requires TF_DS_SPILL_GIB=0; mapped spill transfers are not validated\")\n", name)
        text = _replace_once(text, "        self.limit = int(context or 65536)\n        self.max_rows = drafts + 1\n", "", name)
        text = _replace_once(text, "        cfg = self.w.cfg\n", "        cfg = self.w.cfg\n        self.limit = int(context or 65536)\n        self.max_rows = drafts + 1\n        self.display_pool_cap = (display_kv.pool_capacity(cfg, self.display_bytes, max_rows=0)\n                                 if self.display_bytes else 0)\n        if self.display_bytes and self.display_pool_cap < self.limit + self.max_rows + 8:\n            raise ValueError(\"display KV pool is smaller than the requested context plus decode scratch rows\")\n", name)
        text = _replace_once(text, "mine = torch.tensor([int(self.vcfg is not None), int(parallel), int(markov.ON), int(markov.SPLIT)],",
            "mine = torch.tensor([int(self.vcfg is not None), int(parallel), int(markov.ON), int(markov.SPLIT),\n                                self.display_bytes, self.display_pool_cap, self.limit],", name)
        text = _replace_once(text, "world * 4", "world * 7", name)
        text = _replace_once(text, "every.view(world, 4)", "every.view(world, 7)", name)
        text = _replace_once(text, "--vision, --parallel, TF_DS_MARKOV and TF_DS_MARKOV_SPLIT must be the same on every ",
            "--vision, --parallel, TF_DS_MARKOV, TF_DS_MARKOV_SPLIT and display-KV settings must be the same on every ", name)
        text = _replace_once(text, "streams share one window of ",
            "streams share {'display KV pool' if self.display_bytes else 'one window'} of ", name)
        text = _replace_once(text, "tokens (an extent each)\", flush=True)",
            "tokens{f'; per-request context {self.limit}' if self.display_bytes else ' (an extent each)'}\", flush=True)", name)
        return text
    if name == "multi.py":
        text = _replace_once(text, "from . import constrain\n", "from . import constrain\nfrom . import display_kv\n", name)
        text = _replace_once(text, "        self.m, self.cap = m, int(cap)\n        self.pool = m.new_pool(slots, self.cap)\n",
            "        self.m = m\n        self.cap = (display_kv.pool_capacity(m.cfg, engine.display_bytes, max_rows=0)\n                    if engine.display_bytes else int(cap))\n        self.pool = m.new_pool(slots, self.cap, display_bytes=engine.display_bytes)\n", name)
        text = _replace_once(text, "                    t[b:b + k].copy_(t[a:a + k].clone() if overlap else t[a:a + k])",
            "                    if self.pool.display_arena is not None:\n                        display_kv.copy_rows(t, a, b, k)\n                    else:\n                        t[b:b + k].copy_(t[a:a + k].clone() if overlap else t[a:a + k])", name)
        text = _replace_once(text, "        room = self.extents.total - len(s.prompt) - self.max_rows - 8 - 4\n",
            "        admit_cap = min(self.extents.total, self.e.limit)\n        room = admit_cap - len(s.prompt) - self.max_rows - 8 - 4\n", name)
        return text
    if name == "app.py":
        return _replace_once(text, 'class DsApp(App):\n    reads_ignore_eos = True\n',
            'class DsApp(App):\n    reads_ignore_eos = True\n    strict_tool_output = True\n', name)
    if name == "server.py":
        text = _replace_once(text, 'from tensorfold.cuda.turns import Turns, Yield\n',
            'from tensorfold.cuda.turns import Turns, Yield\nfrom tensorfold.cuda.tool_output import ToolOutputGuard\n', name)
        text = _replace_once(text, '        policy = ToolCallPolicy(body)\n',
            '        policy = ToolCallPolicy(body)\n        tool_output_guard = (ToolOutputGuard(tools) if chat and getattr(self, "strict_tool_output", False)\n                             and (tools or "tools" in body) else None)\n', name)
        text = _replace_once(text,
            '        calls_stream = ToolCallStreamer(tools) if tools and not policy.single else None\n',
            '        calls_stream = (ToolCallStreamer(tools) if tools and not policy.single\n                        and tool_output_guard is None else None)\n', name)
        text = _replace_once(text, '            answer_raw[0] = answer\n',
            '            answer_raw[0] = answer\n            if tool_output_guard is not None:\n                # Validate the whole answer before releasing content or executable call deltas.\n                return reasoning, ""\n', name)
        text = _replace_once(text,
            '        content, calls = parse_tool_calls(raw_answer, tools, max_calls=policy.max_calls) if tools else (answer, None)\n        content = policy.content(content) if tools else content\n',
            '        if tool_output_guard is not None:\n            content, calls = tool_output_guard.parse(raw_answer, max_calls=policy.max_calls)\n        else:\n            content, calls = parse_tool_calls(raw_answer, tools, max_calls=policy.max_calls) if tools else (answer, None)\n            content = policy.content(content) if tools else content\n', name)
        return text
    if name == "http.py":
        text = _replace_once(text, 'from tensorfold.cuda import health\n',
            'from tensorfold.cuda import health\nfrom tensorfold.cuda.tool_output import ModelOutputError\n', name)
        text = _replace_once(text,
            '                except RequestError as exc:\n                    return self._stream_error(error_body(exc, field))\n',
            '                except RequestError as exc:\n                    return self._stream_error(error_body(exc, field))\n                except ModelOutputError as exc:\n                    print(f"[tensorfold] model output rejected: {exc.code}", flush=True)\n                    return self._stream_error(exc.error)\n', name)
        text = _replace_once(text,
            '                result = app.run(body, chat, lambda delta: True, prepared=prepared, cancelled=cancelled)\n            except RequestCancelled:\n',
            '                result = app.run(body, chat, lambda delta: True, prepared=prepared, cancelled=cancelled)\n            except ModelOutputError as exc:\n                print(f"[tensorfold] model output rejected: {exc.code}", flush=True)\n                return self._json(500, {"error": exc.error})\n            except RequestCancelled:\n', name)
        return text
    if name == "health.py":
        return _replace_once(text, '        return body\n',
            '        reader = getattr(getattr(app, "engine", None), "cache_metrics_snapshot", None)\n        if callable(reader):\n            body.update(reader())\n        return body\n', name)
    if name == "metrics.py":
        text = _replace_once(text, '    pools = _pools(app)\n',
            '    cache = _cache_snapshot(app)\n    pools = _pools(app, cache)\n', name)
        text = _replace_once(text, '    _family(lines, "mtp_drafted_total",',
            '''    for key, help_text in (
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
    ):
        if key in cache:
            _family(lines, key, "gauge", help_text, [f"{PREFIX}{key} {cache[key]}"])
    if cache.get("pool_tokens", 0) > 0:
        ratio = cache["pool_used_tokens"] / cache["pool_tokens"]
        _family(lines, "kv_pool_usage_ratio", "gauge", "Reserved shared pool positions divided by total positions.",
                [f"{PREFIX}kv_pool_usage_ratio {_num(ratio)}"])
    if "kv_cache_backing" in cache:
        backing = cache["kv_cache_backing"]
        _family(lines, "kv_cache_backing_info", "gauge", "Backing for this rank's compressed KV and indexer tensors.",
                [f'{PREFIX}kv_cache_backing_info{{backing="{backing}"}} 1'])
    _family(lines, "mtp_drafted_total",''', name)
        text = _replace_once(text,
            '            "Tokens in a stream cache divided by that stream\'s context window.",',
            '            "Live stream cache / request context; DeepSeek reports the maximum live stream ratio.",', name)
        text = _replace_once(text, 'def _pools(app: Any) -> list[tuple[str, float]]:',
            '''def _cache_snapshot(app: Any) -> dict:
    reader = getattr(getattr(app, "engine", None), "cache_metrics_snapshot", None)
    return reader() if callable(reader) else {}


def _pools(app: Any, cache: dict | None = None) -> list[tuple[str, float]]:''', name)
        text = _replace_once(text, '    window = _window(app)\n',
            '''    cache = _cache_snapshot(app) if cache is None else cache
    if "stream_kv_usage_ratio" in cache:
        return [("0", cache["stream_kv_usage_ratio"])]
    window = _window(app)
''', name)
        return text
    raise ValueError(f"unsupported source file: {relative}")


def _git_head(checkout: Path) -> str:
    return subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()


def install(checkout: Path) -> None:
    checkout = checkout.resolve()
    if _git_head(checkout) != REVISION:
        raise ValueError(f"checkout HEAD must be {REVISION}")
    overlays = {checkout / FAMILY / name: (HERE / "overlay" / name).read_text()
                for name in ("display_kv.py", "cache_metrics.py")}
    overlays[checkout / "src/tensorfold/cuda/tool_output.py"] = (HERE / "overlay/tool_output.py").read_text()
    source = {path: (checkout / path).read_text() for path in FILES}
    for destination in overlays:
        if destination.exists():
            raise ValueError(f"{destination}: display-KV is already installed")
    for path, text in source.items():
        actual = hashlib.sha256(text.encode()).hexdigest()
        if actual != FILES[path]:
            raise ValueError(f"{path}: source fingerprint differs from pinned revision")
    edits = {path: transform(path, text) for path, text in source.items()}
    for path, text in edits.items():
        ast.parse(text, filename=str(path))
    for destination, overlay in overlays.items():
        ast.parse(overlay, filename=str(destination))
    # All reads and validation complete. These are the first writes.
    for path, text in edits.items():
        (checkout / path).write_text(text)
    for destination, overlay in overlays.items():
        destination.write_text(overlay)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("CHECKOUT", type=Path, help="pinned TensorFold source checkout")
    args = parser.parse_args()
    install(args.CHECKOUT)


if __name__ == "__main__":
    main()
