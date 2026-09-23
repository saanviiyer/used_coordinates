"""Train convolutional recurrent world models on RGB panorama prediction."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train
import visual_env
from visual_models import VisualGRU


def make_batch(B, T, cond, seed, rng, domain="rectangle", palette=0,
               lighting=1.0):
    frames, action, pos, head = visual_env.rollout(
        B, T, seed, domain, palette, lighting)
    occupancy = 0 if cond == "lit" else .5
    mask = train.blackout_mask(B, T, occupancy, rng)
    used_action = action.copy()
    if cond == "shuffle":
        index = np.argsort(rng.random((B, T)), axis=1)
        used_action = np.take_along_axis(used_action, index[..., None], axis=1)
    masked = frames * (1 - mask[..., None, None, None])
    return tuple(torch.from_numpy(x.astype(np.float32)) for x in
                 (masked, frames, mask, used_action, pos, head))


@torch.no_grad()
def evaluate(model, cond, seed=810_000, domain="rectangle", palette=0):
    rng = np.random.default_rng(seed + 9)
    masked, target, mask, action, _, _ = make_batch(
        64, 64, cond, seed, rng, domain, palette)
    pred = model(masked, mask, action)
    se = ((pred[:, :-1] - target[:, 1:]) ** 2).mean((2,3,4))
    m = mask[:, :-1]
    return {"lit_mse": float(se[m == 0].mean()),
            "dark_mse": float(se[m == 1].mean()) if (m == 1).any() else None}


def fit(cond, seed, steps, B, T, outdir: Path, domain="rectangle"):
    torch.manual_seed(seed); np.random.seed(seed)
    model = VisualGRU(); optimizer = torch.optim.Adam(model.parameters(), 2e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, steps)
    rng = np.random.default_rng(seed + 777); t0 = time.time(); history = []
    for step in range(1, steps + 1):
        masked, target, mask, action, _, _ = make_batch(
            B, T, cond, seed * 100_003 + step, rng, domain)
        pred = model(masked, mask, action)
        loss = ((pred[:, :-1] - target[:, 1:]) ** 2).mean()
        optimizer.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1); optimizer.step(); scheduler.step()
        if step % max(1, steps//10) == 0:
            history.append({"step": step, "mse": float(loss)})
            print(cond, seed, step, round(float(loss),5), flush=True)
    result = {"kind": "visual_gru", "cond": cond, "seed": seed,
              "steps": steps, "params": sum(p.numel() for p in model.parameters()),
              "history": history,
              "eval_lit": evaluate(model, "lit", domain=domain),
              "eval_dark": evaluate(model, "dark", domain=domain),
              "wall_s": time.time()-t0, "domain": domain}
    outdir.mkdir(parents=True, exist_ok=True); tag=f"visual_gru_{cond}_s{seed}"
    torch.save(model.state_dict(), outdir/(tag+".pt"))
    (outdir/(tag+".json")).write_text(json.dumps(result,indent=2))


if __name__ == "__main__":
    ap=argparse.ArgumentParser()
    ap.add_argument("--conds",default="lit,dark,shuffle")
    ap.add_argument("--seeds",default="0,1,2,3")
    ap.add_argument("--steps",type=int,default=2500)
    ap.add_argument("--batch",type=int,default=32)
    ap.add_argument("--T",type=int,default=64)
    ap.add_argument("--domain",choices=("rectangle","ellipse"),default="rectangle")
    ap.add_argument("--outdir",type=Path,default=Path("runs/visual"))
    a=ap.parse_args()
    for cond in a.conds.split(","):
        for seed in map(int,a.seeds.split(",")):
            tag=f"visual_gru_{cond}_s{seed}"
            if (a.outdir/(tag+".json")).exists(): print("skip",tag); continue
            fit(cond,seed,a.steps,a.batch,a.T,a.outdir,a.domain)
