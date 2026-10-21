"""Read-only per-rank cache metrics for DeepSeek-V4.1's parallel decoder."""


def _tensor_bytes(value):
    """Logical tensor bytes, using metadata only (never touches tensor storage)."""
    try:
        return int(value.numel()) * int(value.element_size())
    except (AttributeError, TypeError, ValueError):
        return None


def _snapshot_bytes(snaps):
    """Count unique tensors reachable from a kept prompt's GPU snapshots."""
    seen = set()

    def visit(value):
        if isinstance(value, dict):
            return sum(visit(item) for item in value.values())
        if isinstance(value, (tuple, list)):
            return sum(visit(item) for item in value)
        key = id(value)
        if key in seen:
            return 0
        size = _tensor_bytes(value)
        if size is None:
            return 0
        seen.add(key)
        return size

    return visit(snaps)


def _packed_bytes(pool):
    """Bytes of unique compressed KV and indexer tensors in this rank's pool."""
    seen = set()
    total = 0
    for planes in (pool.comp, pool.index_k):
        for pair in planes.values():
            for tensor in pair if isinstance(pair, tuple) else (pair,):
                key = id(tensor)
                if key in seen:
                    continue
                size = _tensor_bytes(tensor)
                if size is not None:
                    seen.add(key)
                    total += size
    return total


def _occupied(stream, prefilling):
    if prefilling:
        value = getattr(stream, "filled", None)
    else:
        state = getattr(stream, "st", None)
        cache = getattr(state, "sc", None)
        value = getattr(cache, "length", None)
        if value is None:
            context = getattr(stream, "context", None)
            value = len(context) if context is not None else None
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return None


def snapshot(engine) -> dict:
    """Return local parallel-decoder cache state, or ``{}`` when unsupported."""
    decoder = getattr(engine, "multi", None)
    if decoder is None:
        return {}
    extents = getattr(decoder, "extents", None)
    if extents is None or not hasattr(extents, "total") or not hasattr(extents, "gaps"):
        return {}

    # Copy the mutable scheduler collections before deriving related values.
    gaps = tuple(extents.gaps)
    kept = tuple(decoder.kept.values())
    decoding = tuple(decoder.streams.values())
    filling = tuple(decoder.filling)
    total = int(extents.total)
    free = sum(max(0, int(end) - int(start)) for start, end in gaps)
    free = min(max(free, 0), max(total, 0))
    largest = max((max(0, int(end) - int(start)) for start, end in gaps), default=0)
    largest = min(largest, free)

    pool = decoder.pool
    payload = _packed_bytes(pool)
    arena = getattr(pool, "display_arena", None)
    arena_size = getattr(arena, "size", None)
    display_bytes = int(arena_size) if arena_size is not None else 0
    backing = "display" if arena_size is not None else "cuda"
    capacity = display_bytes if backing == "display" else payload

    occupied = [value for stream, prefilling in ((stream, False) for stream in decoding)
                for value in (_occupied(stream, prefilling),) if value is not None]
    occupied.extend(value for stream in filling for value in (_occupied(stream, True),) if value is not None)
    limit = int(engine.limit)
    usage = min(1.0, max(occupied, default=0) / limit) if limit > 0 else 0.0

    result = {
        "pool_tokens": total,
        "pool_free_tokens": free,
        "pool_used_tokens": max(0, total - free),
        "pool_largest_free_tokens": largest,
        "pool_free_extents": len(gaps),
        "kept_prompts": len(kept),
        "kept_bytes": _snapshot_bytes(tuple(item.snaps for item in kept)),
        "pool_kept_tokens": sum(int(item.size) for item in kept),
        "context_length": limit,
        "stream_kv_usage_ratio": usage,
        "streams": {"decoding": len(decoding), "prefilling": len(filling), "max": len(decoder.slots)},
        "kv_cache_payload_bytes": payload,
        "kv_cache_display_bytes": display_bytes,
        "kv_cache_ordinary_bytes": 0 if backing == "display" else payload,
        "kv_cache_capacity_bytes": capacity,
        "kv_cache_backing": backing,
        "kv_cache_bytes_scope": "compressed_kv_and_indexer_per_rank",
    }
    spill = getattr(decoder, "spill", None)
    if callable(getattr(spill, "info", None)):
        result["spill"] = spill.info()
    return result
