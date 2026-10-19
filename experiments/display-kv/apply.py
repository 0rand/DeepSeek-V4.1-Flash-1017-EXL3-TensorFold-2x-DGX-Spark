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
    FAMILY / "kernels.py": "1db323f4553f729b5a7281990dc6086fbcf1503921a6bdf30e6218e71d178367",
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
    if name == "kernels.py":
        return _replace_once(text,
            '    bits = (score + 0.0).view(torch.int32)                       # + 0.0: -0 becomes +0\n'
            '    ordered = torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits).to(torch.int64)\n'
            '    keys = (ordered << 32) | (0xFFFFFFFF - torch.arange(score.shape[-1], device=score.device, dtype=torch.int64))\n'
            '    top = keys.topk(k, dim=-1, sorted=False).values\n'
            '    return (0xFFFFFFFF - (top & 0xFFFFFFFF)).sort(dim=-1).values\n',
            '    from .bounded_topk import topk_indices as bounded_topk\n\n'
            '    return bounded_topk(score, k)\n', name)
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
        start = text.index("    def _memory_ceiling(self) -> None:\n")
        end = text.index("    def warm(self) -> None:\n", start)
        text = _replace_once(text, text[start:end],
            '    def _memory_ceiling(self) -> None:\n'
            '        from .memory_guard import MemoryGuard\n\n'
            '        self.memory_guard = MemoryGuard(torch, rank=self.rank,\n'
            '                                        floor_gib=float(os.environ.get("TF_DS_MEM_FLOOR_GIB") or 2.0))\n'
            '        self.memory_guard.refresh()\n\n', name)
        text = _replace_once(text, "        for s in range(start, len(prompt), PREFILL_CHUNK):\n",
            '        for s in range(start, len(prompt), PREFILL_CHUNK):\n'
            '            guard = getattr(self, "memory_guard", None)\n'
            '            if guard is not None:\n'
            '                guard.refresh()\n', name)
        text = _replace_once(text, "            if e < len(prompt):\n                yield e\n",
            '            if guard is not None:\n'
            '                guard.refresh()\n'
            '            if e < len(prompt):\n                yield e\n', name)
        text = _replace_once(text, "    def _memory_ceiling(self) -> None:\n",
            '    def cache_metrics_snapshot(self) -> dict:\n        """Read local cache metadata without allocating or synchronizing the GPU."""\n        from .cache_metrics import snapshot\n\n        return snapshot(self)\n\n    def _memory_ceiling(self) -> None:\n', name)
        text = _replace_once(text, "        self.model_dir = Path(model_dir)                 # (multi's spilled kept prompts: part of their engine identity)\n",
            "        self.model_dir = Path(model_dir)                 # (multi's spilled kept prompts: part of their engine identity)\n        from . import display_kv\n        self.display_bytes = display_kv.enabled_bytes()\n        if self.display_bytes:\n            if int(parallel) <= 1:\n                raise ValueError(\"display KV requires --parallel greater than 1 before weights load\")\n            if os.environ.get(\"TF_DS_KV\", \"native\") != \"native\":\n                raise ValueError(\"display KV requires TF_DS_KV=native packed storage\")\n", name)
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
        text = _replace_once(text,
            '        why = f"a step failed on rank {self.e.rank} ({type(exc).__name__}: {str(exc)[:160]})"\n',
            '        guard = getattr(self.e, "memory_guard", None)\n'
            '        if guard is not None:\n'
            '            print(f"[tensorfold] rank {self.e.rank}: last allocator budget {guard.last}", flush=True)\n'
            '        why = f"a step failed on rank {self.e.rank} ({type(exc).__name__}: {str(exc)[:160]})"\n', name)
        text = _replace_once(text, "        e = self.e\n        t0 = time.perf_counter()\n        self._step(True)\n",
            '        e = self.e\n'
            '        guard = getattr(e, "memory_guard", None)\n'
            '        if guard is not None:\n'
            '            guard.refresh()\n'
            '        t0 = time.perf_counter()\n        self._step(True)\n', name)
        text = _replace_once(text, "            self.round_end = time.perf_counter()\n            return done\n",
            '            if guard is not None:\n'
            '                guard.refresh()\n'
            '            self.round_end = time.perf_counter()\n            return done\n', name)
        text = _replace_once(text, "from . import constrain\n", "from . import constrain\nfrom . import display_kv\n", name)
        text = _replace_once(text, "        self.m, self.cap = m, int(cap)\n        self.pool = m.new_pool(slots, self.cap)\n",
            "        self.m, self.cap = m, int(cap)\n"
            "        self.spill = None\n"
            "        self.spill_jobs: list = []\n"
            "        self.spill_nonce = os.urandom(4).hex() if engine.rank == 0 else None\n"
            "        spill_cfg = None\n"
            "        if KEEP:\n"
            "            from tensorfold.cuda.spill import SpillConfig, preallocate\n"
            "            gib = float(os.environ.get(\"TF_DS_SPILL_GIB\") or 0)\n"
            "            if gib > 0:\n"
            "                spill_cfg = SpillConfig(\n"
            "                    root=os.path.expanduser(os.environ.get(\"TF_DS_SPILL_DIR\") or \"~/.cache/tensorfold\"),\n"
            "                    gib=gib,\n"
            "                    highwater=float(os.environ.get(\"TF_DS_SPILL_HIGHWATER\") or 0.70),\n"
            "                    min_tokens=int(os.environ.get(\"TF_DS_SPILL_MIN\") or 4096),\n"
            "                    min_free_gib=float(os.environ.get(\"TF_DS_SPILL_MIN_FREE_GIB\") or 50.0),\n"
            "                    keep_builds=int(os.environ.get(\"TF_DS_SPILL_KEEP_BUILDS\") or 1),\n"
            "                    writers=int(os.environ.get(\"TF_SPILL_WRITERS\") or 2),\n"
            "                    direct=os.environ.get(\"TF_SPILL_DIRECT\", \"1\") != \"0\",\n"
            "                    flush_s=float(os.environ.get(\"TF_SPILL_FLUSH_S\") or 60),\n"
            "                )\n"
            "                preallocate(spill_cfg)\n"
            "        self.cap = (display_kv.pool_capacity(m.cfg, engine.display_bytes, max_rows=0)\n"
            "                    if engine.display_bytes else int(cap))\n"
            "        self.pool = m.new_pool(slots, self.cap, display_bytes=engine.display_bytes)\n", name)
        text = _replace_once(text, "        self.spill = None                                # TF_DS_SPILL_GIB: kept prompts on this rank's disk\n"
            "        self.spill_nonce = os.urandom(4).hex() if engine.rank == 0 else None   # names this run's spilled prompts\n"
            "        if KEEP:\n"
            "            from . import spill as sp\n"
            "\n"
            "            if sp.SPILL_GIB > 0:\n"
            "                try:\n"
            "                    ratios = {i: m.cfg.compress_ratios[i] for i in set(self.pool.comp) | set(self.pool.index_k)}\n"
            "                    ident = sp.engine_id(getattr(engine, \"model_dir\", \".\"), world=engine.world, pool=self.pool,\n"
            "                                         dpool=self.dpool)\n"
            "                    self.spill = sp.Spill(sp.SPILL_DIR, engine=ident, rank=engine.rank, pool=self.pool, ratios=ratios,\n"
            "                                          budget=int(sp.SPILL_GIB * 2**30), max_age=sp.SPILL_MAX_AGE_H * 3600.0)\n"
            "                    self.spill.recycle()\n"
            "                    if engine.rank == 0:\n"
            "                        st = self.spill.stats()\n"
            "                        print(f\"[tensorfold] spilled kept prompts: {self.spill.dir} ({st['manifests']} on disk, \"\n"
            "                              f\"{st['bytes'] / 2**30:.2f} of {sp.SPILL_GIB:g} GiB)\", flush=True)\n"
            "                except Exception as exc:                 # noqa: BLE001  (no disk: kept prompts stay in the window only)\n"
            "                    print(f\"[tensorfold] rank {engine.rank}: spilled kept prompts off ({type(exc).__name__}: {exc})\",\n"
            "                          flush=True)\n"
            "                    self.spill = None\n",
            "        if spill_cfg is not None:\n"
            "            try:\n"
            "                from tensorfold.cuda.spill import SpillStore, build_id, weights_id\n"
            "                planes = []\n"
            "                for kind, planes_dict in ((\"comp\", self.pool.comp), (\"index_k\", self.pool.index_k)):\n"
            "                    for i, x in sorted(planes_dict.items()):\n"
            "                        for j, t in enumerate(x if isinstance(x, tuple) else (x,)):\n"
            "                            planes.append((kind, i, j, tuple(t.shape[1:]), str(t.dtype)))\n"
            "                signature = \"|\".join([\n"
            "                    \"dsv41-multi\", build_id(getattr(engine, \"model_dir\", \".\")),\n"
            "                    os.environ.get(\"TF_DS_KV\", \"native\"),\n"
            "                    f\"world{engine.world}\",\n"
            "                    f\"display{getattr(engine, 'display_bytes', 0)}\",\n"
            "                    f\"replay{int(self._replay())}\",\n"
            "                    repr(planes),\n"
            "                ])\n"
            "                spanned = []\n"
            "                if getattr(self.pool, \"display_arena\", None) is not None:\n"
            "                    for kind, planes_dict in ((\"comp\", self.pool.comp), (\"index_k\", self.pool.index_k)):\n"
            "                        for i, x in planes_dict.items():\n"
            "                            for j in range(len(x) if isinstance(x, tuple) else 1):\n"
            "                                spanned.append(f\"{kind}.{i}.{j}\")\n"
            "                self.spill = SpillStore(spill_cfg, rank=engine.rank, world=engine.world, device=\"cuda\",\n"
            "                                        signature=signature,\n"
            "                                        model_id=os.path.basename(str(getattr(engine, \"model_dir\", \".\"))),\n"
            "                                        share=self._share, gather=self._gather_ints,\n"
            "                                        weights=weights_id(getattr(engine, \"model_dir\", \".\")),\n"
            "                                        classes=[\"tensorfold.families.deepseek_v41.cuda.multi:Kept\"],\n"
            "                                        spanned=spanned)\n"
            "                if engine.rank == 0:\n"
            "                    st = self.spill.info()\n"
            "                    print(f\"[tensorfold] spilled kept prompts: {self.spill.dir} ({st['entries']} on disk, \"\n"
            "                          f\"{st['stored_bytes'] / 2**30:.2f} of {spill_cfg.gib:g} GiB)\", flush=True)\n"
            "            except Exception as exc:                     # noqa: BLE001  (no disk: kept prompts stay in the window only)\n"
            "                print(f\"[tensorfold] rank {engine.rank}: spilled kept prompts off ({type(exc).__name__}: {exc})\",\n"
            "                      flush=True)\n"
            "                self.spill = None\n", name)
        text = _replace_once(text, "                    t[b:b + k].copy_(t[a:a + k].clone() if overlap else t[a:a + k])",
            "                    if self.pool.display_arena is not None:\n                        display_kv.copy_rows(t, a, b, k)\n                    else:\n                        t[b:b + k].copy_(t[a:a + k].clone() if overlap else t[a:a + k])", name)
        text = _replace_once(text, "        room = self.extents.total - len(s.prompt) - self.max_rows - 8 - 4\n",
            "        admit_cap = min(self.extents.total, self.e.limit)\n        room = admit_cap - len(s.prompt) - self.max_rows - 8 - 4\n", name)
        text = _replace_once(text, "    def _forget(self, k: Kept, give: bool = True) -> None:\n        self.kept.pop(k.eid, None)\n",
            "    def _release_rows(self, base: int, keep: int = 0) -> None:\n"
            "        \"\"\"Wait for pending writes that read rows past ``keep`` of the extent at ``base``.\"\"\"\n"
            "\n"
            "        if self.spill is None or not self.spill_jobs:\n"
            "            return\n"
            "        self.spill_jobs = [j for j in self.spill_jobs if not j[0].copied.is_set()]\n"
            "        for job, jbase, jtop in self.spill_jobs:\n"
            "            if jbase == base and jtop > keep:\n"
            "                self.spill.release(job)\n"
            "\n"
            "    def _spill_obj(self, k: Kept):\n"
            "        \"\"\"A shallow copy of ``k`` whose fields the spill tier can encode (host as an array, snaps string-keyed).\"\"\"\n"
            "\n"
            "        obj = Kept.__new__(Kept)\n"
            "        obj.eid, obj.base, obj.size, obj.top = k.eid, k.base, k.size, k.top\n"
            "        obj.host = k.host.view()\n"
            "        obj.snaps = {str(b): v for b, v in k.snaps.items()}\n"
            "        obj.replay, obj.keys, obj.tick = k.replay, k.keys, k.tick\n"
            "        return obj\n"
            "\n"
            "    def _spill_rows(self, k: Kept):\n"
            "        \"\"\"The pool rows of ``k``'s prompt, named for the spill tier (``comp.<layer>.<part>``).\"\"\"\n"
            "\n"
            "        rows = []\n"
            "        for kind, planes in ((\"comp\", self.pool.comp), (\"index_k\", self.pool.index_k)):\n"
            "            for i, x in sorted(planes.items()):\n"
            "                r = self.m.cfg.compress_ratios[i]\n"
            "                a, b = k.base // r, (k.base + k.top) // r\n"
            "                for j, t in enumerate(x if isinstance(x, tuple) else (x,)):\n"
            "                    rows.append((f\"{kind}.{i}.{j}\", t[a:b]))\n"
            "        return rows\n"
            "\n"
            "    def _dst_for(self, base: int):\n"
            "        \"\"\"Place a stored row into this rank's pool at ``base`` (display-backed rows go through the tier's kernel).\"\"\"\n"
            "\n"
            "        def dst(name, shape, dt):\n"
            "            if name.startswith(\"rows.comp.\") or name.startswith(\"rows.index_k.\"):\n"
            "                kind, layer, part = name[5:].split(\".\")\n"
            "                planes = self.pool.comp if kind == \"comp\" else self.pool.index_k\n"
            "                x = planes[int(layer)]\n"
            "                t = x[int(part)] if isinstance(x, tuple) else x\n"
            "                r = self.m.cfg.compress_ratios[int(layer)]\n"
            "                a = base // r\n"
            "                return t[a:a + shape[0]]\n"
            "            return None\n"
            "        return dst\n"
            "\n"
            "    def _gather_ints(self, values: list[int]) -> list[list[int]]:\n"
            "        \"\"\"Every rank's ``values`` (the spill tier's agreement).\"\"\"\n"
            "\n"
            "        if self.e.world < 2:\n"
            "            return [list(values)]\n"
            "        t = torch.tensor(values, dtype=torch.int64, device=\"cuda\")\n"
            "        return self.m.comm.gather(t).tolist()\n"
            "\n"
            "    def _forget(self, k: Kept, give: bool = True) -> None:\n"
            "        self._release_rows(k.base, 0)\n"
            "        self.kept.pop(k.eid, None)\n", name)
        text = _replace_once(text, "        if self.spill is not None:                       # write-through: its rows past what its lineage holds\n"
            "            k.disk = self.spill.persist(f\"{self.spill_nonce or 'run'}-{k.eid:06d}\", base=k.base, top=k.top,\n"
            "                                        replay=k.replay, snaps=k.snaps, host=k.host,\n"
            "                                        lineage=getattr(s, \"lineage\", None), keys=k.keys)\n",
            "        if self.spill is not None:                       # write-through: its rows and snapshots to disk\n"
            "            ids = k.keys if k.keys is not None else k.host.view()\n"
            "            job = self.spill.save(ids, self._spill_obj(k), self._spill_rows(k),\n"
            "                                  skip=(\"disk\", \"hits\"),\n"
            "                                  tokens=list(k.host.view()) if k.keys is not None else None)\n"
            "            if job is not None:\n"
            "                k.disk = job\n"
            "                self.spill_jobs.append((job, k.base, k.top))\n", name)
        text = _replace_once(text, "        if self.spill is not None and s.keys is not None:   # a spilled boundary past every kept prompt's\n"
            "            far = self.spill.best(s.keys, len(s.prompt), window=self.m.cfg.window, replay=self._replay(),\n"
            "                                  above=int(reuse[1]) if reuse else 0)\n"
            "            if far is not None:\n"
            "                reuse = [\"disk\", far[0], far[1]]\n",
            "        if self.spill is not None and s.keys is not None:   # a stored prompt longer than any kept one\n"
            "            far = self.spill.find(s.keys, grid=0, whole_needs_head=True,\n"
            "                                  usable=lambda e: bool(e.fields.get(\"replay\")) == self._replay())\n"
            "            if far is not None and (not reuse or far.n > int(reuse[1])):\n"
            "                reuse = [\"disk\", far.key, far.n]\n", name)
        text = _replace_once(text, "            disk = bool(reuse) and reuse[0] == \"disk\"\n"
            "            if disk and not self._all(self.spill is not None and self.spill.check(str(reuse[1]), int(reuse[2]))):\n"
            "                if e.rank == 0:\n"
            "                    print(f\"[tensorfold] spilled prompt {reuse[1]}: not on every rank, the prompt fills fresh\",\n"
            "                          flush=True)\n"
            "                reuse, disk = None, False\n",
            "            disk = bool(reuse) and reuse[0] == \"disk\"\n", name)
        text = _replace_once(text, "            if disk:\n"
            "                name, at, t0 = str(reuse[1]), int(reuse[2]), time.perf_counter()\n"
            "                try:\n"
            "                    snaps, host, lineage = self.spill.load(name, at, base)\n"
            "                    loaded = at in snaps\n"
            "                except Exception as exc:                 # noqa: BLE001  (every rank agrees on the outcome below)\n"
            "                    print(f\"[tensorfold] rank {e.rank}: spilled prompt {name} did not load ({type(exc).__name__}: \"\n"
            "                          f\"{exc})\", flush=True)\n"
            "                    loaded = False\n"
            "                if self._all(loaded):                    # else the rows written so far get filled over, fresh\n"
            "                    cut, how = at, \"disk\"\n"
            "                    self._restore(index, snaps[cut])\n"
            "                    slot.sc.host = HostIds(host)\n"
            "                    s.snaps, s.lineage = snaps, lineage\n"
            "                    self.keep_stats[how] = self.keep_stats.get(how, 0) + 1\n"
            "                    if e.rank == 0:\n"
            "                        print(f\"[tensorfold] spilled prompt {name}: continued at {cut} of {len(s.prompt)} (disk, \"\n"
            "                              f\"{time.perf_counter() - t0:.2f}s)\", flush=True)\n"
            "                elif e.rank == 0:\n"
            "                    print(f\"[tensorfold] spilled prompt {name}: a rank could not read it, the prompt fills fresh\",\n"
            "                          flush=True)\n",
            "            if disk:\n"
            "                key, at, t0 = tuple(reuse[1]), int(reuse[2]), time.perf_counter()\n"
            "                try:\n"
            "                    lj = self.spill.load_async(key, self._dst_for(base))\n"
            "                    obj, _rows, _meta = self.spill.finish_load(lj)\n"
            "                    loaded = obj is not None\n"
            "                except Exception as exc:                 # noqa: BLE001  (every rank agrees on the outcome below)\n"
            "                    print(f\"[tensorfold] rank {e.rank}: spilled prompt {key} did not load ({type(exc).__name__}: \"\n"
            "                          f\"{exc})\", flush=True)\n"
            "                    loaded = False\n"
            "                if self._all(loaded):                    # else the rows written so far get filled over, fresh\n"
            "                    cut, how = at, \"disk\"\n"
            "                    snaps = {int(b): v for b, v in obj.snaps.items()}\n"
            "                    self._restore(index, snaps[cut])\n"
            "                    slot.sc.host = HostIds(obj.host)\n"
            "                    s.snaps, s.lineage = snaps, None\n"
            "                    self.keep_stats[how] = self.keep_stats.get(how, 0) + 1\n"
            "                    if e.rank == 0:\n"
            "                        print(f\"[tensorfold] spilled prompt {key}: continued at {cut} of {len(s.prompt)} (disk, \"\n"
            "                              f\"{time.perf_counter() - t0:.2f}s)\", flush=True)\n"
            "                elif e.rank == 0:\n"
            "                    print(f\"[tensorfold] spilled prompt {key}: a rank could not read it, the prompt fills fresh\",\n"
            "                          flush=True)\n", name)
        text = _replace_once(text, "        sp = self.spill\n"
            "        spilled = sp is not None and all(k.disk is not None for k in kept_now if k.top >= sp.min_top)\n"
            "        for eid in drops:\n"
            "            if int(eid) in self.kept:\n"
            "                k = self.kept[int(eid)]\n"
            "                if spilled and k.disk is not None:\n"
            "                    sp.forget(k.disk[\"name\"])\n"
            "                self._forget(k)\n"
            "        while len(self.kept) > KEEP_ENTRIES:\n"
            "            self._forget(min(self.kept.values(), key=lambda k: k.tick))\n"
            "        if sp is not None and kept_now:\n"
            "            sp.recycle(pinned={k.disk[\"name\"] for k in self.kept.values() if k.disk is not None})\n",
            "        sp = self.spill\n"
            "        spilled = sp is not None and all(k.disk is not None for k in kept_now if k.top >= sp.cfg.min_tokens)\n"
            "        for eid in drops:\n"
            "            if int(eid) in self.kept:\n"
            "                k = self.kept[int(eid)]\n"
            "                if spilled and k.disk is not None:\n"
            "                    sp.forget(k.disk.entry.key)\n"
            "                self._forget(k)\n"
            "        while len(self.kept) > KEEP_ENTRIES:\n"
            "            self._forget(min(self.kept.values(), key=lambda k: k.tick))\n", name)
        text = _replace_once(text, "            if any(b - a >= trial.size(need) for a, b in trial.gaps):\n"
            "                self.extents.give(k.base + size, k.size - size)\n",
            "            if any(b - a >= trial.size(need) for a, b in trial.gaps):\n"
            "                self._release_rows(k.base, m)\n"
            "                self.extents.give(k.base + size, k.size - size)\n", name)
        text = _replace_once(text, "        if src == dst:\n            return\n        c = self.m.cfg\n",
            "        if src == dst:\n            return\n        self._release_rows(src, 0)\n        c = self.m.cfg\n", name)
        text = _replace_once(text, "            if src is not None:\n                self.extents.give(src.base, src.size)\n",
            "            if src is not None:\n                self._release_rows(src.base, 0)\n                self.extents.give(src.base, src.size)\n", name)
        return text
    if name == "app.py":
        text = _replace_once(text, 'class DsApp(App):\n    reads_ignore_eos = True\n',
            'class DsApp(App):\n    reads_ignore_eos = True\n    strict_tool_output = True\n', name)
        text = _replace_once(text, 'import dataclasses\n', 'import dataclasses\nimport time\n', name)
        text = _replace_once(text, '        template.render = render_effort\n',
            '        template.render = render_effort\n'
            '        self.on_exit = self._flush_spill\n', name)
        text = _replace_once(text, '    @staticmethod\n    def _response_format(',
            '    def _flush_spill(self) -> None:\n'
            '        """Wait for queued spill writes before the process exits (docker stop sends SIGTERM)."""\n'
            '\n'
            '        multi = getattr(self.engine, "multi", None)\n'
            '        spill = getattr(multi, "spill", None)\n'
            '        if spill is None:\n'
            '            return\n'
            '        t = time.perf_counter()\n'
            '        ok = spill.drain(55.0)\n'
            '        print(f"[tensorfold] spill: shutdown drain {\'done\' if ok else \'not finished\'} "\n'
            '              f"in {time.perf_counter() - t:.1f}s", flush=True)\n'
            '\n'
            '    @staticmethod\n    def _response_format(', name)
        return text
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
        text = _replace_once(text,
            '    except KeyboardInterrupt:\n        pass\n    finally:\n        server.server_close()\n',
            '    except KeyboardInterrupt:\n'
            '        hook = getattr(app, "on_exit", None)\n'
            '        if hook is not None:\n'
            '            try:\n'
            '                hook()\n'
            '            except Exception as exc:\n'
            '                print(f"[tensorfold] shutdown hook failed: {exc!r}", flush=True)\n'
            '    finally:\n        server.server_close()\n', name)
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
    spill = cache.get("spill")
    if isinstance(spill, dict):
        for key in sorted(spill):
            value = spill[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            if key.endswith("_gib"):
                continue
            if key.endswith(("_ms_p50", "_ms_p95")):
                name, kind, value = f"spill_{key.replace('_ms_', '_seconds_')}", "gauge", value / 1000.0
            elif key in {"entries", "pending", "stored_bytes", "disk_free_bytes", "highwater", "writers"}:
                name, kind = f"spill_{key}", "gauge"
            elif key.endswith("_s"):
                name, kind = f"spill_{key[:-2]}_seconds_total", "counter"
            else:
                name, kind = f"spill_{key}_total", "counter"
            _family(lines, name, kind, f"Spill tier: /health's spill.{key}.", [f"{PREFIX}{name} {_num(value)}"])
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
                for name in ("display_kv.py", "cache_metrics.py", "memory_guard.py", "bounded_topk.py")}
    overlays[checkout / "src/tensorfold/cuda/tool_output.py"] = (HERE / "overlay/tool_output.py").read_text()
    overlays[checkout / "src/tensorfold/cuda/spill.py"] = (HERE / "overlay/spill_tier.py").read_text()
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
