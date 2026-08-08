"""3D scan serialization orders."""

import torch


def morton_order(nd, nh, nw, device="cpu"):
    """Morton (Z-order) curve permutation for a 3D grid.

    Returns a LongTensor of shape (nd*nh*nw,) that maps raster-order
    flat indices to Morton-order flat indices.
    """
    def code(i, j, k):
        c = 0
        for b in range(16):
            c |= ((i >> b) & 1) << (3 * b)
            c |= ((j >> b) & 1) << (3 * b + 1)
            c |= ((k >> b) & 1) << (3 * b + 2)
        return c

    flat = [(code(i, j, k), (i * nh + j) * nw + k)
            for i in range(nd) for j in range(nh) for k in range(nw)]
    flat.sort()
    return torch.tensor([lin for _, lin in flat], dtype=torch.long, device=device)


def raster_order(nd, nh, nw, device="cpu"):
    """Identity (raster) permutation for a 3D grid."""
    return torch.arange(nd * nh * nw, dtype=torch.long, device=device)


def scan_order(nd, nh, nw, use_morton=True, device="cpu"):
    """Morton (Z-order) or raster permutation, gated by use_morton."""
    if use_morton:
        return morton_order(nd, nh, nw, device=device)
    return raster_order(nd, nh, nw, device=device)


def hilbert_order(nd, nh, nw, device="cpu"):
    raise NotImplementedError("Hilbert order not implemented in v1")
