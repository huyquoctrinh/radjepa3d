"""3D Segmentation: Mamba encoder + 3D decoder.

Architecture:
    CT Volume -> Mamba Encoder -> (B, 1024, 384) patch features
    Patch features -> 3D Decoder -> (B, 1, D, H, W) binary mask
"""

import sys
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

_SRC = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_SRC, os.path.join(_SRC, "utils")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from vjepa_model import MambaEncoder
from load_checkpoint import build_vjepa_config_from


class DecoderBlock(nn.Module):

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_ch, out_ch, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
            nn.Conv3d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.InstanceNorm3d(out_ch),
            nn.GELU(),
        )

    def forward(self, x):
        return self.conv(self.up(x))


class SegDecoder(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.grid_size = (
            config.volume_size[0] // config.patch_size[0],
            config.volume_size[1] // config.patch_size[1],
            config.volume_size[2] // config.patch_size[2],
        )

        dims = config.decoder_dims
        self.input_proj = nn.Linear(config.vision_dim, dims[0])

        # (4,16,16) -> (8,32,32) -> (16,64,64) -> (32,128,128)
        self.blocks = nn.ModuleList()
        in_dim = dims[0]
        for out_dim in dims:
            self.blocks.append(DecoderBlock(in_dim, out_dim))
            in_dim = out_dim

        # (32,128,128) -> (32,256,256); output a shared feature map (feat_dim ch)
        self.feat_dim = 32
        self.final_up = nn.Sequential(
            nn.ConvTranspose3d(dims[-1], dims[-1], kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Conv3d(dims[-1], self.feat_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv3d(self.feat_dim, self.feat_dim, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, patch_features):
        B = patch_features.shape[0]
        gd, gh, gw = self.grid_size

        x = self.input_proj(patch_features)
        x = rearrange(x, "b (gd gh gw) c -> b c gd gh gw", gd=gd, gh=gh, gw=gw)

        for block in self.blocks:
            x = block(x)

        return self.final_up(x)  # (B, feat_dim, D, H, W)


def multiclass_dice_loss(logits, target, num_classes, ignore_background=True):
    """Soft multi-class Dice over foreground classes. logits (B,C,D,H,W)."""
    probs = F.softmax(logits, dim=1)
    onehot = F.one_hot(target.clamp(0, num_classes - 1), num_classes)
    onehot = onehot.permute(0, 4, 1, 2, 3).float()  # (B,C,D,H,W)
    start = 1 if (ignore_background and num_classes > 1) else 0
    smooth = 1.0
    dims = (0, 2, 3, 4)
    inter = (probs[:, start:] * onehot[:, start:]).sum(dims)
    union = probs[:, start:].sum(dims) + onehot[:, start:].sum(dims)
    dice = (2 * inter + smooth) / (union + smooth)
    return 1.0 - dice.mean()


class SegModel(nn.Module):

    def __init__(self, config, num_classes_by_code):
        """num_classes_by_code: {dataset_code: num_classes (incl background)}."""
        super().__init__()
        self.config = config
        self.num_classes_by_code = dict(num_classes_by_code)

        vjepa_cfg = build_vjepa_config_from(config)
        self.encoder = MambaEncoder(vjepa_cfg)

        self.decoder = SegDecoder(config)

        # One 1x1-conv head per sub-dataset (keeps channel meaning consistent)
        feat = self.decoder.feat_dim
        self.heads = nn.ModuleDict({
            code: nn.Conv3d(feat, C, kernel_size=1)
            for code, C in self.num_classes_by_code.items()
        })

    def load_pretrained_encoder(self, path):
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        key = "target_encoder" if "target_encoder" in ckpt else "context_encoder"
        missing, unexpected = self.encoder.load_state_dict(ckpt[key], strict=False)
        print(f"Loaded encoder from {key} in {path} "
              f"(missing={list(missing)}, unexpected={list(unexpected)})")

    def forward(self, volumes, code, labels=None):
        feat = self.decoder(self.encoder(volumes))  # (B, feat_dim, D, H, W)
        logits = self.heads[code](feat)             # (B, C, D, H, W)
        C = self.num_classes_by_code[code]

        result = {"logits": logits, "num_classes": C}

        if labels is not None:
            if logits.shape[2:] != labels.shape[1:]:
                labels = F.interpolate(
                    labels.unsqueeze(1).float(), size=logits.shape[2:], mode="nearest",
                ).squeeze(1).long()
            ce = F.cross_entropy(logits, labels)
            dice = multiclass_dice_loss(logits, labels, C)
            result["loss"] = ce + dice
            result["ce_loss"] = ce
            result["dice_loss"] = dice

        return result

    @torch.no_grad()
    def predict(self, volumes, code):
        logits = self.forward(volumes, code)["logits"]
        return logits.argmax(dim=1)  # (B, D, H, W) class indices
