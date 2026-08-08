"""3D Patch Embedding stem."""

import torch.nn as nn
from einops import rearrange


class PatchEmbed3D(nn.Module):

    def __init__(self, in_chans, dim, patch):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv3d(in_chans, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        B, C, D, H, W = x.shape
        pd, ph, pw = self.patch
        assert D % pd == 0 and H % ph == 0 and W % pw == 0, \
            f"Volume ({D},{H},{W}) not divisible by patch ({pd},{ph},{pw})"

        x = self.proj(x)
        gd, gh, gw = x.shape[2], x.shape[3], x.shape[4]
        tokens = rearrange(x, "b e d h w -> b (d h w) e")
        return tokens, (gd, gh, gw)
