"""3D masking strategies for V-JEPA."""

import sys as _sys, os as _os
_UTILS = _os.path.dirname(_os.path.abspath(__file__))
_SRC = _os.path.dirname(_UTILS)
for _p in (_SRC, _UTILS):
    if _p not in _sys.path:
        _sys.path.insert(0, _p)


import torch


class RandomMasker:
    """Randomly select patches to mask."""

    def __init__(self, n_patches, mask_ratio):
        self.n_patches = n_patches
        self.n_visible = int(n_patches * (1 - mask_ratio))
        self.n_masked = n_patches - self.n_visible

    def __call__(self, batch_size, device="cpu"):
        visible_list = []
        masked_list = []
        for _ in range(batch_size):
            perm = torch.randperm(self.n_patches, device=device)
            visible_list.append(perm[: self.n_visible].sort().values)
            masked_list.append(perm[self.n_visible :].sort().values)
        visible = torch.stack(visible_list)
        masked = torch.stack(masked_list)
        return visible, masked


class BlockMasker:
    """V-JEPA-style multi-block masking in 3D grid."""

    def __init__(self, grid_size, mask_ratio, num_blocks=4,
                 block_scale=(0.15, 0.3), block_aspect=(0.75, 1.5)):
        self.gd, self.gh, self.gw = grid_size
        self.n_patches = self.gd * self.gh * self.gw
        self.target_masked = int(self.n_patches * mask_ratio)
        self.num_blocks = num_blocks
        self.block_scale = block_scale
        self.block_aspect = block_aspect

    def _sample_block(self):
        scale = torch.empty(1).uniform_(*self.block_scale).item()
        n_block = int(self.n_patches * scale)

        aspect = torch.empty(1).uniform_(*self.block_aspect).item()
        bw = max(1, min(self.gw, int(round((n_block * aspect) ** (1 / 3)))))
        bh = max(1, min(self.gh, int(round((n_block / bw) ** 0.5))))
        bd = max(1, min(self.gd, int(round(n_block / (bw * bh)))))

        d0 = torch.randint(0, max(1, self.gd - bd + 1), (1,)).item()
        h0 = torch.randint(0, max(1, self.gh - bh + 1), (1,)).item()
        w0 = torch.randint(0, max(1, self.gw - bw + 1), (1,)).item()

        indices = []
        for d in range(d0, d0 + bd):
            for h in range(h0, h0 + bh):
                for w in range(w0, w0 + bw):
                    indices.append(d * self.gh * self.gw + h * self.gw + w)
        return set(indices)

    def __call__(self, batch_size, device="cpu"):
        visible_list = []
        masked_list = []

        for _ in range(batch_size):
            masked_set = set()
            for _ in range(self.num_blocks * 3):
                if len(masked_set) >= self.target_masked:
                    break
                masked_set |= self._sample_block()

            masked_indices = sorted(masked_set)[:self.target_masked]
            if len(masked_indices) < self.target_masked:
                remaining = set(range(self.n_patches)) - set(masked_indices)
                extra = torch.tensor(list(remaining))
                perm = extra[torch.randperm(len(extra))]
                masked_indices += perm[: self.target_masked - len(masked_indices)].tolist()
                masked_indices.sort()

            visible_indices = sorted(set(range(self.n_patches)) - set(masked_indices))

            visible_list.append(torch.tensor(visible_indices, device=device))
            masked_list.append(torch.tensor(masked_indices, device=device))

        n_vis = min(v.shape[0] for v in visible_list)
        n_mask = min(m.shape[0] for m in masked_list)
        visible = torch.stack([v[:n_vis] for v in visible_list])
        masked = torch.stack([m[:n_mask] for m in masked_list])
        return visible, masked


def build_masker(config):
    if config.mask_strategy == "block":
        return BlockMasker(
            grid_size=config.grid_size,
            mask_ratio=config.mask_ratio,
        )
    return RandomMasker(config.n_patches, config.mask_ratio)
