"""Train the RSSM on a control-suite replay buffer.

Same model, same objective and the same blackout regime as the arm sweeps; only
the data is a real simulated robot. Windows are sampled from a pre-rendered
buffer because rendering is slower than training.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train as T
from rssm import RSSM

# See train_rssm.py. The 0.01-per-dimension floor was still above this model's
# operating KL (measured 0.044 to 0.155 on this buffer against a floor of
# 0.32), so the clamp bound on every step and the KL term contributed no
# gradient. The floor now sits an order of magnitude below the observed range,
# and the trainer records whether the term was actually live.
FREE_NATS_PER_DIM = 0.0005


class Buffer:
    def __init__(self, path):
        d = np.load(path, allow_pickle=True)
        self.frames = d["frames"]                     # E, L, 3, H, W  uint8
        self.actions = d["actions"].astype(np.float32)
        self.angles = d["angles"].astype(np.float32)
        self.domain = str(d["domain"]); self.task = str(d["task"])
        self.E, self.L = self.frames.shape[:2]
        self.H, self.W = self.frames.shape[3:]
        self.action_dim = self.actions.shape[-1]

    def sample(self, B, T, cond, rng):
        e = rng.integers(0, self.E, B)
        t0 = rng.integers(0, self.L - T, B)
        idx = t0[:, None] + np.arange(T)[None, :]
        frames = self.frames[e[:, None], idx].astype(np.float32) / 255.0
        action = self.actions[e[:, None], idx].copy()
        angles = self.angles[e[:, None], idx].copy()
        occupancy = 0.0 if cond == "lit" else 0.5
        mask = T_blackout(B, T, occupancy, rng)
        if cond == "shuffle":
            order = np.argsort(rng.random((B, T)), axis=1)
            action = np.take_along_axis(action, order[..., None], axis=1)
        masked = frames * (1 - mask[..., None, None, None])
        out = (masked, frames, mask, action, angles)
        return tuple(torch.from_numpy(x.astype(np.float32)) for x in out)


def T_blackout(B, length, occupancy, rng):
    return T.blackout_mask(B, length, occupancy, rng)


def prediction_loss(pred, target, mask=None, blackout_weight=4.0,
                    foreground_weight=6.0):
    sq = (pred - target) ** 2
    # Weight pixels that differ from the scene's modal colour: the robot is a
    # small part of the frame and an unweighted loss is dominated by backdrop.
    modal = target.mean(dim=(0, 1, 3, 4), keepdim=True)
    fg = ((target - modal).abs().max(2, keepdim=True).values > 0.08).float()
    weight = 1.0 + foreground_weight * fg
    per_time = (sq * weight).sum((-3, -2, -1)) / weight.sum((-3, -2, -1))
    if mask is None:
        return per_time.mean()
    temporal = 1.0 + blackout_weight * mask
    return (per_time * temporal).sum() / temporal.sum()


def fit(buf, cond, seed, steps, B, Tlen, outdir: Path, kl_scale):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RSSM(image_h=buf.H, image_w=buf.W, action_dim=buf.action_dim)
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed + 1_777)
    history, t0 = [], time.time()
    n_steps = n_binding = 0
    kl_raw_max = 0.0
    for step in range(1, steps + 1):
        masked, target, mask, action, _ = buf.sample(B, Tlen, cond, rng)
        pred, _, kl = model.observe(masked, mask, action, sample=True)
        recon = prediction_loss(pred[:, :-1], target[:, 1:], mask[:, :-1])
        floor = FREE_NATS_PER_DIM * model.stoch
        raw = kl.mean()
        n_steps += 1
        n_binding += float(raw) < floor
        kl_raw_max = max(kl_raw_max, float(raw))
        loss = recon + kl_scale * torch.clamp(raw, min=floor)
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step(); sched.step()
        if step % max(1, steps // 10) == 0:
            history.append({"step": step, "recon": float(recon),
                            "kl_raw": float(raw), "kl_floor": float(floor),
                            "kl_clamped": bool(float(raw) < floor)})
            print(buf.domain, cond, seed, step, round(float(recon), 5),
                  "kl_raw", round(float(raw), 4), "floor", round(floor, 4),
                  "CLAMPED" if float(raw) < floor else "live", flush=True)
    # No statistic of the KL term itself separates "the term never engaged"
    # from "the term engaged and converged": free nats are meant to stop
    # pushing once the KL is under the allowance, so a high binding fraction
    # late in training is healthy, and a transient spike at initialisation
    # lets a dead term pass an "ever reached the floor" test.  Both of those
    # were tried and both failed.  These numbers are therefore recorded as
    # diagnostics, not as a verdict.  The verdict is src/blackout_validity.py,
    # which measures the thing that actually matters -- whether the prior path
    # can predict during a blackout -- against a frame-persistence baseline.
    binding_frac = n_binding / max(1, n_steps)
    floor = FREE_NATS_PER_DIM * model.stoch
    result = {"kind": "rssm", "domain": buf.domain, "task": buf.task,
              "kl_binding_fraction": round(binding_frac, 4),
              "kl_raw_max": round(kl_raw_max, 5),
              "kl_floor": round(floor, 5),
              "cond": cond, "seed": seed, "steps": steps,
              "params": sum(p.numel() for p in model.parameters()),
              "image": [buf.H, buf.W], "action_dim": buf.action_dim,
              "history": history, "wall_s": time.time() - t0}
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"dmc_rssm_{cond}_s{seed}"
    torch.save(model.state_dict(), outdir / (tag + ".pt"))
    (outdir / (tag + ".json")).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--buffer", default="runs/dmc/reacher_easy.npz")
    ap.add_argument("--conds", default="dark")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--outdir", type=Path, default=Path("runs/dmc_reacher"))
    ap.add_argument("--kl-scale", type=float, default=1.0)
    a = ap.parse_args()
    buf = Buffer(a.buffer)
    print(f"buffer {buf.domain}-{buf.task}: {buf.E}x{buf.L} frames "
          f"{buf.H}x{buf.W}, action dim {buf.action_dim}")
    for cond in a.conds.split(","):
        for seed in map(int, a.seeds.split(",")):
            tag = f"dmc_rssm_{cond}_s{seed}"
            if (a.outdir / (tag + ".json")).exists():
                print("skip", tag); continue
            fit(buf, cond, seed, a.steps, a.batch, a.T, a.outdir, a.kl_scale)
