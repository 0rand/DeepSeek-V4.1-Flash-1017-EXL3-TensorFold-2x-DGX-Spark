"""Display-only packed KV storage.

Adapted from Gilesco patch 0072-glm-display-kv.patch (Apache-2.0 recipe).
Unlike that patch this module never creates an ordinary-memory prefix: every
byte mapped here is the selected NVIDIA card's display allocation.
"""

from __future__ import annotations

import os
from collections import namedtuple

ENV = "TF_DS_DISPLAY_KV_MIB"
CARD_ENV = "TF_DS_DISPLAY_KV_CARD"
SYSFS = "/sys/class/drm"
MIB = 1 << 20
MAX_MIB = 1792
ALIGN = 256
CAP_ALIGN = 2048

Allocation = namedtuple("Allocation", "kind layer part shape fill offset")


def enabled_bytes(environ=None) -> int:
    """Return the requested display bytes, or zero when display KV is off."""
    raw = ((os.environ if environ is None else environ).get(ENV) or "").strip()
    if not raw:
        return 0
    if not raw.isascii() or not raw.isdecimal():
        raise ValueError(f"{ENV}={raw!r}: expected a number of MiB")
    value = int(raw)
    if value == 0:
        return 0
    if value % 16 or value > MAX_MIB:
        raise ValueError(f"{ENV}={value}: expected a multiple of 16 from 16 to {MAX_MIB}")
    return value * MIB


def _read(path: str) -> str:
    with open(path, encoding="utf-8") as file:
        return file.read().strip()


def _card_name(card: str) -> str:
    name = os.path.basename(card.rstrip("/"))
    if not name.startswith("card") or not name[4:].isdigit():
        raise ValueError(f"{CARD_ENV}={card!r}: expected a DRM card path such as /dev/dri/card0")
    return name


def is_nvidia_card(sysfs: str, card: str) -> bool:
    """Whether one DRM card belongs to NVIDIA, without touching other cards."""
    base = os.path.join(sysfs, _card_name(card), "device")
    try:
        if _read(os.path.join(base, "vendor")).lower() == "0x10de":
            return True
    except OSError:
        pass
    try:
        return any(line.strip() == "DRIVER=nvidia" for line in _read(os.path.join(base, "uevent")).splitlines())
    except OSError:
        return False


def selected_card(environ=None, sysfs: str = SYSFS) -> str:
    """Select an NVIDIA DRM card, validating an explicit choice."""
    environ = os.environ if environ is None else environ
    requested = (environ.get(CARD_ENV) or "").strip()
    if requested:
        if not is_nvidia_card(sysfs, requested):
            raise RuntimeError(f"{CARD_ENV}={requested!r} is not an NVIDIA DRM card")
        return requested
    try:
        entries = sorted(os.listdir(sysfs))
    except OSError as exc:
        raise RuntimeError(f"{ENV}: cannot inspect {sysfs}: {exc}") from exc
    for card in entries:
        if card.startswith("card") and card[4:].isdigit() and is_nvidia_card(sysfs, card):
            return "/dev/dri/" + card
    raise RuntimeError(f"{ENV}: no NVIDIA DRM card found")


def connected_outputs(sysfs: str, card: str) -> list[str]:
    """Return connected outputs with an enabled scanout encoder on this card.

    TTY and simple-framebuffer entries do not use a DRM connector namespace and
    are deliberately ignored.
    """
    name = _card_name(card)
    entries = sorted(os.listdir(sysfs))
    found = []
    for entry in entries:
        if not entry.startswith(name + "-"):
            continue
        status = os.path.join(sysfs, entry, "status")
        if _read(status) == "connected":
            enabled = _read(os.path.join(sysfs, entry, "enabled"))
            if enabled not in ("enabled", "disabled"):
                raise RuntimeError(f"{ENV}: unknown encoder state for {entry}: {enabled!r}")
            if enabled == "enabled":
                found.append(entry[len(name) + 1 :])
    return found


def _field(cfg, name):
    try:
        return cfg[name]
    except (TypeError, KeyError):
        try:
            return getattr(cfg, name)
        except AttributeError:
            raise ValueError(f"display KV config is missing {name}") from None


def _config(cfg):
    sources = tuple(int(source) for source in _field(cfg, "kv_sources"))
    ratios = tuple(int(r) for r in _field(cfg, "compress_ratios"))
    index_sources = frozenset(int(source) for source in _field(cfg, "index_sources"))
    head_dim, idx_dim = int(_field(cfg, "head_dim")), int(_field(cfg, "idx_dim"))
    if (
        not sources
        or len(set(sources)) != len(sources)
        or any(source < 0 or source >= len(ratios) for source in sources)
    ):
        raise ValueError("invalid packed KV source IDs")
    if any(ratios[source] < 1 for source in sources) or head_dim < 16 or idx_dim < 32 or head_dim % 16 or idx_dim % 32:
        raise ValueError("invalid packed KV dimensions or compression ratios")
    return sources, ratios, index_sources, head_dim, idx_dim


def _up(value: int, alignment: int = ALIGN) -> int:
    return (value + alignment - 1) // alignment * alignment


def plan(cfg, cap: int) -> list[Allocation]:
    """Packed KV/index planes for ``cap`` tokens; each plane begins on 256 B."""
    if cap < 0:
        raise ValueError("capacity must not be negative")
    sources, ratios, index_sources, head_dim, idx_dim = _config(cfg)
    result, offset = [], 0
    for layer in sources:
        rows = cap // ratios[layer] + 2
        for part, width in (("data", head_dim // 2), ("scale", head_dim // 16)):
            offset = _up(offset)
            result.append(Allocation("kv", layer, part, (rows, width), 0, offset))
            offset += rows * width
        if layer in index_sources:
            for part, width in (("data", idx_dim // 2), ("scale", idx_dim // 32)):
                offset = _up(offset)
                result.append(Allocation("index", layer, part, (rows, width), 0, offset))
                offset += rows * width
    return result


def layout_bytes(cfg, cap: int) -> int:
    """Total display bytes required by the packed allocations in :func:`plan`."""
    entries = plan(cfg, cap)
    return 0 if not entries else _up(max(item.offset + item.shape[0] * item.shape[1] for item in entries))


def budget(cfg, cap: int) -> int:
    """Alias for the display-memory budget of a full packed allocation."""
    return layout_bytes(cfg, cap)


def pool_capacity(cfg, display_bytes: int, max_rows: int = 0) -> int:
    """Largest 2048-token pool whose packed KV layout fits display RAM."""
    if display_bytes < 0 or max_rows < 0:
        raise ValueError("display_bytes and max_rows must not be negative")
    lo, hi = 0, CAP_ALIGN
    while layout_bytes(cfg, hi + max_rows) <= display_bytes:
        lo, hi = hi, hi * 2
    while hi - lo > CAP_ALIGN:
        middle = ((lo + hi) // (2 * CAP_ALIGN)) * CAP_ALIGN
        if layout_bytes(cfg, middle + max_rows) <= display_bytes:
            lo = middle
        else:
            hi = middle
    return lo


class System:
    """DRM/CUDA calls, isolated so CPU tests can replace this factory."""

    CREATE_DUMB = 0xC02064B2
    MAP_DUMB = 0xC01064B3
    GEM_CLOSE = 0x40086409
    DEVICEMAP, IOMEMORY = 0x02, 0x04
    PITCH, MAP_FAILED = 16384, (1 << 64) - 1

    def __init__(self, card: str) -> None:
        import ctypes

        self.card = card
        self.libc = ctypes.CDLL("libc.so.6", use_errno=True)
        self.libc.mmap.restype = ctypes.c_void_p
        self.libc.mmap.argtypes = [
            ctypes.c_void_p,
            ctypes.c_size_t,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_long,
        ]
        self.libc.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.libc.munmap.restype = ctypes.c_int
        self.cuda = ctypes.CDLL("libcuda.so.1")
        self.cuda.cuMemHostRegister_v2.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
        self.cuda.cuMemHostRegister_v2.restype = ctypes.c_int
        self.cuda.cuMemHostGetDevicePointer_v2.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_uint,
        ]
        self.cuda.cuMemHostGetDevicePointer_v2.restype = ctypes.c_int
        self.cuda.cuMemHostUnregister.argtypes = [ctypes.c_void_p]
        self.cuda.cuMemHostUnregister.restype = ctypes.c_int
        self.cuda.cuGetErrorName.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]

    def open_card(self):
        return os.open(self.card, os.O_RDWR | os.O_CLOEXEC)

    def create_dumb(self, fd, size):
        import fcntl
        import struct

        layout = "=6IQ"
        result = fcntl.ioctl(
            fd, self.CREATE_DUMB, struct.pack(layout, (size + self.PITCH - 1) // self.PITCH, 4096, 32, 0, 0, 0, 0)
        )
        fields = struct.unpack(layout, result)
        return fields[4], fields[6]

    def map_offset(self, fd, handle):
        import fcntl
        import struct

        return struct.unpack("=2IQ", fcntl.ioctl(fd, self.MAP_DUMB, struct.pack("=2IQ", handle, 0, 0)))[2]

    def map_display(self, size, fd, offset):
        return self._mmap(size, 3, 1, fd, offset)

    def register(self, address, size):
        import ctypes

        self._check(
            self.cuda.cuMemHostRegister_v2(
                ctypes.c_void_p(address), ctypes.c_size_t(size), ctypes.c_uint(self.DEVICEMAP | self.IOMEMORY)
            )
        )

    def device_pointer(self, address):
        import ctypes

        pointer = ctypes.c_void_p()
        self._check(self.cuda.cuMemHostGetDevicePointer_v2(ctypes.byref(pointer), ctypes.c_void_p(address), 0))
        return int(pointer.value or 0)

    def unregister(self, address):
        import ctypes

        self._check(self.cuda.cuMemHostUnregister(ctypes.c_void_p(address)))

    def unmap(self, address, size):
        import ctypes

        if self.libc.munmap(ctypes.c_void_p(address), ctypes.c_size_t(size)):
            raise OSError(ctypes.get_errno(), "munmap")

    def close_gem(self, fd, handle):
        import fcntl
        import struct

        fcntl.ioctl(fd, self.GEM_CLOSE, struct.pack("=2I", handle, 0))

    def close_fd(self, fd):
        os.close(fd)

    def _mmap(self, size, prot, flags, fd, offset):
        import ctypes

        ctypes.set_errno(0)
        value = int(self.libc.mmap(None, size, prot, flags, fd, offset) or 0)
        if value in (0, self.MAP_FAILED):
            raise OSError(ctypes.get_errno(), "mmap")
        return value

    def _check(self, result):
        if result:
            import ctypes

            name = ctypes.c_char_p()
            self.cuda.cuGetErrorName(result, ctypes.byref(name))
            raise OSError(f"CUDA driver error {result} ({(name.value or b'?').decode()})")


SYSTEM = System
_OWNERS = []


class Span:
    def __init__(self, pointer: int, size: int, owner) -> None:
        self.pointer, self.size, self._owner = pointer, size, owner

    @property
    def __cuda_array_interface__(self):
        return {"shape": (self.size,), "typestr": "|u1", "data": (self.pointer, False), "version": 3}


def _quiet(call, *args):
    try:
        call(*args)
    except Exception:
        pass


def map_display(display_bytes: int, *, environ=None, sysfs: str = SYSFS) -> Span:
    """Map only a display dumb buffer; failures unwind every acquired resource."""
    if display_bytes <= 0:
        raise ValueError("display_bytes must be positive")
    card = selected_card(environ, sysfs)
    busy = connected_outputs(sysfs, card)
    if busy:
        raise RuntimeError(f"{ENV}: {', '.join(_card_name(card) + '-' + item for item in busy)} is connected")
    system, fd, handle, address, registered = SYSTEM(card), None, None, None, False
    try:
        fd = system.open_card()
        handle, got = system.create_dumb(fd, display_bytes)
        if got < display_bytes:
            raise RuntimeError(f"{ENV}: display allocation is {got} B, short of {display_bytes} B")
        address = system.map_display(display_bytes, fd, system.map_offset(fd, handle))
        system.register(address, display_bytes)
        registered = True
        pointer = system.device_pointer(address)
        if pointer != address:
            raise RuntimeError(f"{ENV}: CUDA device pointer does not match the display mapping")
        span = Span(pointer, display_bytes, (system, fd, handle, address))
        _OWNERS.append(span)
        return span
    except BaseException:
        if registered:
            _quiet(system.unregister, address)
        if address is not None:
            _quiet(system.unmap, address, display_bytes)
        if handle is not None:
            _quiet(system.close_gem, fd, handle)
        if fd is not None:
            _quiet(system.close_fd, fd)
        raise


def as_tensor(span, device="cuda"):
    """Convert a span without copying and reject any CUDA-pointer substitution."""
    import torch

    tensor = torch.as_tensor(span, device=device)
    if tensor.data_ptr() != span.pointer:
        raise RuntimeError(f"{ENV}: torch did not retain the display mapping pointer")
    return tensor


def copy_rows(t, src: int, dst: int, count: int) -> None:
    """Move uint8 display rows through bounded ordinary CUDA scratch storage."""
    import torch

    if t.dtype != torch.uint8:
        raise TypeError("display KV row copies require uint8 tensors")
    if len(t.shape) < 1:
        raise ValueError("display KV tensor must have rows")
    rows = int(t.shape[0])
    if min(src, dst, count) < 0 or src + count > rows or dst + count > rows:
        raise ValueError("display KV row copy is out of bounds")
    if not count or src == dst:
        return
    row_bytes = t.element_size()
    for dimension in t.shape[1:]:
        row_bytes *= int(dimension)
    if not 0 < row_bytes <= 1 << 20:
        raise ValueError("display KV row must be from 1 B through 1 MiB")
    chunk_rows = max(1, (1 << 20) // row_bytes)
    backwards = src < dst < src + count
    remaining = count
    while remaining:
        size = min(chunk_rows, remaining)
        start = remaining - size if backwards else count - remaining
        source = t[src + start : src + start + size]
        scratch = torch.empty_like(source)
        torch.bitwise_or(source, 0, out=scratch)
        torch.bitwise_or(scratch, 0, out=t[dst + start : dst + start + size])
        remaining -= size


class CacheArena:
    """One display-only mapping, with 256-byte-aligned GPU-initialized views."""

    def __init__(self, cfg, cap: int, display_bytes: int) -> None:
        needed = layout_bytes(cfg, cap)
        if needed > display_bytes:
            raise ValueError(f"display arena needs {needed} B, has {display_bytes} B")
        self.cfg, self.cap, self.size, self.offset = cfg, cap, display_bytes, 0
        self.span, self.flat = map_display(display_bytes), None
        print(
            f"[tensorfold] display-only KV: {display_bytes // MIB} MiB mapped, "
            f"{cap} shared positions, {needed} packed-plane bytes; no ordinary KV prefix",
            flush=True,
        )

    def allocate(self, shape, dtype, fill=0):
        import torch

        count = 1
        for dimension in shape:
            count *= int(dimension)
        bytes_ = count * torch.empty((), dtype=dtype).element_size()
        offset = _up(self.offset)
        if offset + bytes_ > self.size:
            raise MemoryError(f"{ENV}: display arena exhausted")
        if self.flat is None:
            self.flat = as_tensor(self.span)
        value = self.flat.narrow(0, offset, bytes_).view(dtype).view(tuple(shape))
        self.offset = offset + bytes_
        return value.zero_() if fill == 0 else value.fill_(fill)

    def zeros(self, shape, dtype):
        return self.allocate(shape, dtype, 0)

    def full(self, shape, fill, dtype):
        return self.allocate(shape, dtype, fill)
