"""Bounded-scratch equivalent of DeepSeek V4.1's ``topk_indices``."""

KEY_SCRATCH_BYTES = 16 << 20


def topk_indices(score, k: int):
    """Return each row's selected indices, ascending, with V4.1's total order.

    The key tile contains complete rows, so row selection is independent of the
    tile size.  ``KEY_SCRATCH_BYTES`` limits its int64 key allocation, except
    that one row is always processed even when it alone exceeds the limit.
    """
    import torch
    import operator

    if score.ndim < 1:
        raise ValueError("score must have at least one dimension")

    width = score.shape[-1]
    k = operator.index(k)
    if not 0 <= k <= width:
        raise ValueError("k must be between zero and the row width")
    rows = score.numel() // width if width else 0
    flat_score = score.reshape(rows, width)
    output = torch.empty((rows, k), dtype=torch.int64, device=score.device)
    if rows == 0 or k == 0:
        return output.reshape(*score.shape[:-1], k)

    rows_per_tile = max(1, KEY_SCRATCH_BYTES // (width * 8))
    index = torch.arange(width, dtype=torch.int64, device=score.device)
    index.neg_().add_(0xFFFFFFFF)

    for start in range(0, rows, rows_per_tile):
        stop = min(start + rows_per_tile, rows)
        # Keep the only full-sized int64 temporary in this tile: the packed key.
        keys = torch.empty((stop - start, width), dtype=torch.int64, device=score.device)
        bits = (flat_score[start:stop] + 0.0).view(torch.int32)
        torch.where(bits < 0, bits ^ 0x7FFFFFFF, bits, out=bits)
        keys.copy_(bits)
        keys.bitwise_left_shift_(32)
        keys.bitwise_or_(index)
        top = keys.topk(k, dim=-1, sorted=False).values
        top.bitwise_and_(0xFFFFFFFF)
        top.neg_().add_(0xFFFFFFFF)
        output[start:stop].copy_(top.sort(dim=-1).values)
        del keys, bits, top

    return output.reshape(*score.shape[:-1], k)
