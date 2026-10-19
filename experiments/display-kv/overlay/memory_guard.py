"""Live allocator headroom for the unified-memory DeepSeek serving lane."""
from __future__ import annotations

import math
import os
from pathlib import Path


GIB = 1 << 30


class MemoryGuard:
    def __init__(self, torch, *, rank: int, floor_gib: float):
        if not math.isfinite(floor_gib) or floor_gib < 0:
            raise ValueError("TF_DS_MEM_FLOOR_GIB must be finite and nonnegative")
        self.cuda, self.rank = torch.cuda, rank
        cap = os.environ.get("TF_DS_MEM_CAP", "1")
        if cap not in ("0", "1"):
            raise ValueError("TF_DS_MEM_CAP must be 0 or 1")
        self.cap_enabled = cap == "1"
        # earlyoom sends SIGTERM at 1 GiB; smaller positive floors cannot protect the lane.
        self.floor = int(max(1.5, floor_gib) * GIB) if floor_gib else 0
        self.total = int(self.cuda.mem_get_info()[1])
        self.last = {}
        self.warned = False

    def refresh(self):
        if not self.floor:
            return
        try:
            info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
            available = int(info["MemAvailable"].split()[0]) * 1024
        except (OSError, KeyError, ValueError):
            if not self.last:
                raise RuntimeError("cannot initialize live allocator guard: MemAvailable is unreadable") from None
            if not self.warned:
                print(f"[tensorfold] rank {self.rank}: MemAvailable unreadable; retaining last allocator budget", flush=True)
                self.warned = True
            return
        self.warned = False
        reserved = int(self.cuda.memory_reserved())
        allocated = int(self.cuda.memory_allocated())
        released = 0
        if available < self.floor + GIB // 2 and reserved - allocated >= 64 << 20:
            # A cached prefill peak is real host RAM on GB10. Return idle blocks before earlyoom sees them.
            self.cuda.empty_cache()
            after = int(self.cuda.memory_reserved())
            released = max(0, reserved - after)
            reserved = after
            # Re-read rather than assuming every freed CUDA byte immediately becomes MemAvailable.
            try:
                info = dict(line.split(":", 1) for line in Path("/proc/meminfo").read_text().splitlines())
                available = int(info["MemAvailable"].split()[0]) * 1024
            except (OSError, KeyError, ValueError):
                pass
        ceiling = min(self.total, reserved + max(0, available - self.floor))
        if self.cap_enabled:
            self.cuda.set_per_process_memory_fraction(ceiling / self.total)
        if not self.last:
            print(f"[tensorfold] rank {self.rank}: live allocator guard, floor {self.floor / GIB:g} GiB, "
                  + (f"initial ceiling {ceiling / GIB:.2f} GiB" if self.cap_enabled else "hard cap DISABLED; cleanup only")
                  + " (refreshed per prefill chunk/decode round)", flush=True)
        self.last = dict(host_available_bytes=available, reserved_bytes=reserved,
                         allocated_bytes=allocated, ceiling_bytes=ceiling,
                         floor_bytes=self.floor, released_bytes=released, cap_enabled=self.cap_enabled)
