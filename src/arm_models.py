"""Action-routed modular visual world model for articulated robots."""
from __future__ import annotations

import torch
import torch.nn as nn

from visual_env import IMAGE_H, IMAGE_W
from visual_models import VisualGRU


class ModularArmGRU(nn.Module):
    """Two recurrent modules, one routed actuator per module.

    Both modules see the same image embedding and blackout flag, but module j
    receives only action j.  The decoder combines both states.  No joint angle
    or pose supervision is used; the inductive bias is only the robot's known
    actuator factorization.
    """
    kind = "arm_modular"

    def __init__(self, module_hidden=64):
        super().__init__()
        self.module_hidden = module_hidden
        self.hidden = 2 * module_hidden
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, 2, 1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1), nn.ReLU(), nn.Flatten(),
            nn.Linear(32 * (IMAGE_H // 8) * (IMAGE_W // 8), 96), nn.ReLU())
        self.cores = nn.ModuleList([
            nn.GRU(96 + 1 + 1, module_hidden, batch_first=True)
            for _ in range(2)])
        self.decoder = nn.Sequential(nn.Linear(self.hidden, 256), nn.ReLU(),
                                     nn.Linear(256, 3 * IMAGE_H * IMAGE_W),
                                     nn.Sigmoid())

    def _split_h0(self, h0, batch, ref):
        if h0 is None:
            return [ref.new_zeros(1, batch, self.module_hidden)
                    for _ in range(2)]
        return [h0[:, j * self.module_hidden:(j + 1) * self.module_hidden][None]
                for j in range(2)]

    def forward(self, frames, mask, action, h0=None, return_h=False):
        B, T = frames.shape[:2]
        z = self.frame_encoder(frames.reshape(B * T, 3, IMAGE_H, IMAGE_W))
        z = z.reshape(B, T, -1)
        starts = self._split_h0(h0, B, z)
        states = []
        for joint, core in enumerate(self.cores):
            x = torch.cat([z, mask[..., None], action[..., joint:joint + 1]], -1)
            hs, _ = core(x, starts[joint].contiguous())
            states.append(hs)
        hidden = torch.cat(states, dim=-1)
        pred = self.decoder(hidden).reshape(B, T, 3, IMAGE_H, IMAGE_W)
        return (pred, hidden) if return_h else pred

    def step(self, frame, mask, action, h):
        z = self.frame_encoder(frame)
        states = []
        for joint, core in enumerate(self.cores):
            x = torch.cat([z, mask[:, None], action[:, joint:joint + 1]], 1)[:, None]
            h0 = h[:, joint * self.module_hidden:(joint + 1) * self.module_hidden]
            _, hn = core(x, h0[None].contiguous())
            states.append(hn[0])
        return torch.cat(states, dim=-1)

    def readout(self, h):
        return self.decoder(h).reshape(-1, 3, IMAGE_H, IMAGE_W)


def build_arm_model(kind):
    if kind == "gru":
        return VisualGRU()
    if kind == "modular":
        return ModularArmGRU()
    if kind == "rssm":
        # Imported lazily so the recurrent baselines do not depend on it.
        from rssm import RSSM
        return RSSM()
    raise ValueError(f"unknown arm model kind: {kind}")
