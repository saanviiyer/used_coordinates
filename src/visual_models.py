"""Convolutional recurrent world model for low-resolution RGB panoramas."""
from __future__ import annotations

import torch
import torch.nn as nn

from visual_env import IMAGE_H, IMAGE_W


class VisualGRU(nn.Module):
    kind = "visual_gru"

    def __init__(self, hidden=128):
        super().__init__(); self.hidden = hidden
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * (IMAGE_H//8) * (IMAGE_W//8), 96), nn.ReLU())
        self.core = nn.GRU(96 + 1 + 2, hidden, batch_first=True)
        self.decoder = nn.Sequential(nn.Linear(hidden, 256), nn.ReLU(),
                                     nn.Linear(256, 3 * IMAGE_H * IMAGE_W),
                                     nn.Sigmoid())

    def forward(self, frames, mask, action, h0=None, return_h=False):
        B, T = frames.shape[:2]
        z = self.frame_encoder(frames.reshape(B*T, 3, IMAGE_H, IMAGE_W))
        z = z.reshape(B, T, -1)
        x = torch.cat([z, mask[..., None], action], -1)
        h0 = z.new_zeros(1, B, self.hidden) if h0 is None else h0[None]
        hs, _ = self.core(x, h0.contiguous())
        pred = self.decoder(hs).reshape(B, T, 3, IMAGE_H, IMAGE_W)
        return (pred, hs) if return_h else pred

    def step(self, frame, mask, action, h):
        z = self.frame_encoder(frame)
        x = torch.cat([z, mask[:, None], action], 1)[:, None]
        _, hn = self.core(x, h[None].contiguous())
        return hn[0]

    def readout(self, h):
        """Decode a hidden state without advancing the recurrent dynamics."""
        return self.decoder(h).reshape(-1, 3, IMAGE_H, IMAGE_W)
