"""EMA target encoder wrapper — one per branch."""

import copy
import torch
import torch.nn as nn


class EMATarget(nn.Module):

    def __init__(self, encoder, decay=0.996):
        super().__init__()
        self.encoder = copy.deepcopy(encoder)
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.decay = decay

    @torch.no_grad()
    def update(self, online_encoder):
        for p_t, p_o in zip(self.encoder.parameters(), online_encoder.parameters()):
            p_t.data.mul_(self.decay).add_(p_o.data, alpha=1.0 - self.decay)

    def forward(self, *args, **kwargs):
        return self.encoder(*args, **kwargs)
