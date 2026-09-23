"""Train pixel-predictive recurrent world models for the two-joint arm."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import arm_env
import arm_env_v2
import arm_env_v4
import arm_env_v5
import arm_env_v7
import train
from visual_models import VisualGRU
from arm_models import build_arm_model


def prediction_loss(pred, target, foreground_weight=12.0,
                    distal_weight=0.0, mask=None, blackout_weight=0.0):
    """Pixel loss that cannot be minimized by predicting only background."""
    sq = ((pred - target) ** 2).mean(dim=2)
    foreground = (target.amax(dim=2) > 0.30).float()
    # In v2 the distal link is deliberately blue-dominant, so this term gives
    # the second joint explicit loss coverage without using joint labels.
    distal = ((target[:, :, 2] > target[:, :, 0] + 0.18) &
              (target[:, :, 2] > target[:, :, 1] + 0.18)).float()
    weight = 1.0 + foreground_weight * foreground + distal_weight * distal
    per_time = (sq * weight).sum(dim=(-2, -1)) / weight.sum(dim=(-2, -1))
    if mask is None or blackout_weight == 0:
        return per_time.mean()
    temporal = 1.0 + blackout_weight * mask
    return (per_time * temporal).sum() / temporal.sum()


# v3 shares v2's renderer and differs only in loss weighting.  v4 and v5 are
# new geometries: a parity serial arm and a visually decoupled mechanism.
ENVIRONMENTS = {"v1": arm_env, "v2": arm_env_v2, "v3": arm_env_v2,
                "v4": arm_env_v4, "v5": arm_env_v5, "v6": arm_env_v5,
                "v7": arm_env_v7}
BLACKOUT_WEIGHTED = ("v3", "v4", "v5", "v6", "v7")
# The colour-keyed distal term up-weights blue pixels 4.11x.  That is intended
# for the serial arm, where blue is the under-represented distal link, and is
# an unintended asymmetry in the decoupled mechanism, where the two pointers
# differ only in colour.  v6 removes it.
SYMMETRIC_LOSS = ("v6", "v7")


def make_batch(B, T, cond, seed, rng, palette=0, lighting=1.0,
               variant="v1"):
    environment = ENVIRONMENTS[variant]
    frames, action, q = environment.rollout(B, T, seed, palette, lighting)
    occupancy = 0 if cond == "lit" else 0.5
    mask = train.blackout_mask(B, T, occupancy, rng)
    used_action = action.copy()
    if cond == "shuffle":
        index = np.argsort(rng.random((B, T)), axis=1)
        used_action = np.take_along_axis(used_action, index[..., None], axis=1)
    masked = frames * (1 - mask[..., None, None, None])
    return tuple(torch.from_numpy(x.astype(np.float32)) for x in
                 (masked, frames, mask, used_action, q))


@torch.no_grad()
def evaluate(model, cond, seed=870_000, variant="v1"):
    rng = np.random.default_rng(seed + 19)
    masked, target, mask, action, _ = make_batch(
        64, 64, cond, seed, rng, variant=variant)
    pred = model(masked, mask, action)
    se = ((pred[:, :-1] - target[:, 1:]) ** 2).mean((2, 3, 4))
    weighted = prediction_loss(pred[:, :-1], target[:, 1:],
                               foreground_weight=8 if variant != "v1" else 12,
                               distal_weight=0 if variant in SYMMETRIC_LOSS
                          else (28 if variant != "v1" else 0),
                               mask=mask[:, :-1],
                               blackout_weight=4 if variant in BLACKOUT_WEIGHTED else 0)
    m = mask[:, :-1]
    return {"lit_mse": float(se[m == 0].mean()),
            "dark_mse": float(se[m == 1].mean()) if (m == 1).any() else None,
            "weighted_mse": float(weighted)}


def fit(cond, seed, steps, B, T, outdir: Path, variant="v1", model_kind="gru"):
    torch.manual_seed(seed); np.random.seed(seed)
    model = build_arm_model(model_kind)
    optimizer = torch.optim.Adam(model.parameters(), 2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    rng = np.random.default_rng(seed + 1_777)
    history, t0 = [], time.time()
    for step in range(1, steps + 1):
        masked, target, mask, action, _ = make_batch(
            B, T, cond, seed * 100_003 + step, rng, variant=variant)
        pred = model(masked, mask, action)
        loss = prediction_loss(
            pred[:, :-1], target[:, 1:],
            foreground_weight=8 if variant != "v1" else 12,
            distal_weight=0 if variant in SYMMETRIC_LOSS
                          else (28 if variant != "v1" else 0),
            mask=mask[:, :-1],
            blackout_weight=4 if variant in BLACKOUT_WEIGHTED else 0)
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step(); scheduler.step()
        if step % max(1, steps // 10) == 0:
            history.append({"step": step, "mse": float(loss)})
            print(cond, seed, step, round(float(loss), 5), flush=True)
    result = {"kind": f"arm_visual_{model_kind}", "cond": cond, "seed": seed,
              "steps": steps,
              "params": sum(p.numel() for p in model.parameters()),
              "variant": variant, "history": history,
              "eval_lit": evaluate(model, "lit", variant=variant),
              "eval_dark": evaluate(model, "dark", variant=variant),
              "wall_s": time.time() - t0}
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"arm_{model_kind}_{cond}_s{seed}"
    torch.save(model.state_dict(), outdir / (tag + ".pt"))
    (outdir / (tag + ".json")).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="dark,lit,shuffle")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--steps", type=int, default=800)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--outdir", type=Path, default=Path("runs/arm_visual"))
    ap.add_argument("--variant", choices=tuple(ENVIRONMENTS), default="v1")
    ap.add_argument("--model", choices=("gru", "modular"), default="gru")
    args = ap.parse_args()
    for cond in args.conds.split(","):
        for seed in map(int, args.seeds.split(",")):
            tag = f"arm_{args.model}_{cond}_s{seed}"
            if (args.outdir / (tag + ".json")).exists():
                print("skip", tag); continue
            fit(cond, seed, args.steps, args.batch, args.T, args.outdir,
                args.variant, args.model)
