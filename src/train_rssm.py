"""Train the small RSSM on the arm environments.

Same data, same conditions, same budget as the convolutional GRU sweeps, so the
only thing that differs is the architecture.  That is what makes the comparison
worth anything: if the criterion returns the same family verdicts on a
stochastic latent-variable world model as on a deterministic recurrent one, the
result is about world models rather than about one network.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

import train_arm_visual as TAV
from rssm import RSSM

# Free nats are per-dimension, not per-state.  Two floors have now been wrong
# in the same direction, so the number is no longer trusted on its own and the
# trainer measures whether the term is actually live (see ``kl_term_active`` in
# the result JSON).
#
#   1.0 flat   (to 30 Aug)  sat above the model's total KL of ~0.16.
#   0.01/dim   (30 Aug)     gives 0.32 for stoch=32, still above it: the clamp
#                           bound on 38 of 40 steps and the KL term contributed
#                           no gradient, so the prior was never trained to match
#                           the posterior.  The prior is what the model rolls on
#                           during a blackout, which is the regime the criterion
#                           probes, so every family verdict read off those
#                           checkpoints was meaningless.
#
# The operating KL measured across both trainers is 0.04 to 0.30 nats, so the
# floor is set an order of magnitude below the bottom of that range.
FREE_NATS_PER_DIM = 0.0005


def loss_terms(model, masked, target, mask, action, variant):
    pred, _, kl = model.observe(masked, mask, action, sample=True)
    recon = TAV.prediction_loss(
        pred[:, :-1], target[:, 1:],
        foreground_weight=8 if variant != "v1" else 12,
        distal_weight=0 if variant in TAV.SYMMETRIC_LOSS
        else (28 if variant != "v1" else 0),
        mask=mask[:, :-1],
        blackout_weight=4 if variant in TAV.BLACKOUT_WEIGHTED else 0)
    # Free nats, scaled by latent width so the floor sits below the KL the
    # model actually uses and the term keeps its gradient.
    floor = FREE_NATS_PER_DIM * model.stoch
    raw = kl.mean()
    kl_term = torch.clamp(raw, min=floor)
    return recon, kl_term, float(raw)


@torch.no_grad()
def evaluate(model, cond, variant, seed=870_000):
    rng = np.random.default_rng(seed + 19)
    masked, target, mask, action, _ = TAV.make_batch(
        64, 64, cond, seed, rng, variant=variant)
    pred, _, kl = model.observe(masked, mask, action, sample=False)
    se = ((pred[:, :-1] - target[:, 1:]) ** 2).mean((2, 3, 4))
    m = mask[:, :-1]
    return {"lit_mse": float(se[m == 0].mean()),
            "dark_mse": float(se[m == 1].mean()) if (m == 1).any() else None,
            "kl": float(kl.mean())}


def fit(cond, seed, steps, B, T, outdir: Path, variant, kl_scale):
    torch.manual_seed(seed); np.random.seed(seed)
    model = RSSM()
    opt = torch.optim.Adam(model.parameters(), 2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    rng = np.random.default_rng(seed + 1_777)
    history, t0 = [], time.time()
    n_steps = n_binding = 0
    kl_raw_max = 0.0
    for step in range(1, steps + 1):
        masked, target, mask, action, _ = TAV.make_batch(
            B, T, cond, seed * 100_003 + step, rng, variant=variant)
        recon, kl, raw_kl = loss_terms(model, masked, target, mask, action,
                                       variant)
        floor = FREE_NATS_PER_DIM * model.stoch
        n_steps += 1
        n_binding += raw_kl < floor
        kl_raw_max = max(kl_raw_max, raw_kl)
        loss = recon + kl_scale * kl
        opt.zero_grad(); loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        opt.step(); sched.step()
        if step % max(1, steps // 10) == 0:
            history.append({"step": step, "recon": float(recon),
                            "kl_raw": raw_kl, "kl_term": float(kl),
                            "kl_floor": floor,
                            "kl_clamped": raw_kl < floor})
            print(variant, cond, seed, step, round(float(recon), 5),
                  "kl_raw", round(raw_kl, 4), "floor", round(floor, 4),
                  "CLAMPED" if raw_kl < floor else "live", flush=True)
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
    result = {"kind": "rssm", "cond": cond, "seed": seed, "steps": steps,
              "kl_binding_fraction": round(binding_frac, 4),
              "kl_raw_max": round(kl_raw_max, 5),
              "kl_floor": round(floor, 5),
              "params": sum(p.numel() for p in model.parameters()),
              "variant": variant, "kl_scale": kl_scale, "history": history,
              "eval_dark": evaluate(model, "dark", variant),
              "wall_s": time.time() - t0}
    outdir.mkdir(parents=True, exist_ok=True)
    tag = f"arm_rssm_{cond}_s{seed}"
    torch.save(model.state_dict(), outdir / (tag + ".pt"))
    (outdir / (tag + ".json")).write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default="dark")
    ap.add_argument("--seeds", default="0,1,2,3,4,5,6,7")
    ap.add_argument("--steps", type=int, default=1600)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--outdir", type=Path, default=Path("runs/rssm_v6"))
    ap.add_argument("--variant", default="v6")
    ap.add_argument("--kl-scale", type=float, default=1.0)
    a = ap.parse_args()
    for cond in a.conds.split(","):
        for seed in map(int, a.seeds.split(",")):
            tag = f"arm_rssm_{cond}_s{seed}"
            if (a.outdir / (tag + ".json")).exists():
                print("skip", tag); continue
            fit(cond, seed, a.steps, a.batch, a.T, a.outdir, a.variant,
                a.kl_scale)
