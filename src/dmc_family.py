"""Point the criterion at a world model trained on the control suite.

Differs from the arm analysis in three ways that matter, and in nothing else.

Actions are torques rather than velocity commands, so a perturbation moves the
joint over the following steps instead of instantaneously.  The probe window is
therefore longer and the fitted gain absorbs the unit conversion; ``dt`` is 1.

The simulator reports hinge angles unwrapped, so a free joint accumulates past
2*pi.  Scoring wraps them; discovery never sees them at all.

The number of joints and actuators is read from the buffer rather than assumed.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

import arm_fast
from arm_torus import COARSE, FINE_HALFWIDTH, FINE_POINTS, plane_from_diff
from rssm import RSSM
from train_rssm_dmc import Buffer
import zeroshot as zs

DELTAS = (-0.6, -0.4, -0.2, 0.2, 0.4, 0.6)
DT = 1.0


def wrap(a):
    return np.arctan2(np.sin(a), np.cos(a))


@torch.no_grad()
def collect(model, buf, seed, batch=96, T=48, t_hit=28, horizon=8, deep=6):
    """A forced-blackout state cloud, plus the joint angles for scoring."""
    rng = np.random.default_rng(seed)
    masked, _, mask, action, angles = buf.sample(batch, T, "dark", rng)
    lo, hi = t_hit - deep, t_hit + horizon + 1
    masked[:, lo:hi] = 0.0
    mask[:, lo:hi] = 1.0
    state = model.initial(batch, masked)
    for t in range(t_hit):
        state = model.step(masked[:, t], mask[:, t], action[:, t], state)
    n_pixel = 3 * buf.H * buf.W
    rows = torch.cat([masked[:, t_hit:t_hit + horizon].flatten(2),
                      mask[:, t_hit:t_hit + horizon, None],
                      action[:, t_hit:t_hit + horizon]], dim=-1)
    return state, rows, angles[:, t_hit].numpy()


def _refine(losses, grid, refit):
    still = float(losses[int(np.argmin(np.abs(grid)))])
    best = int(np.argmin(losses))
    fine = np.linspace(grid[best] - FINE_HALFWIDTH,
                       grid[best] + FINE_HALFWIDTH, FINE_POINTS)
    fl = refit(fine)
    b2 = int(np.argmin(fl))
    return {"gain": float(fine[b2]),
            "residual": float(fl[b2] / (still + 1e-12)),
            "gain_at_grid_edge": bool(best in (0, len(grid) - 1))}


def analyse(path: Path, buf, batch=96, probe=0.6):
    base = RSSM(image_h=buf.H, image_w=buf.W, action_dim=buf.action_dim)
    base.load_state_dict(torch.load(path, map_location="cpu"))
    base.eval()
    t0 = time.time()
    state, rows, angles = collect(base, buf, 61_001, batch=batch)
    _, rows_t, angles_t = collect(base, buf, 91_001, batch=batch)
    mu = state.mean(0, keepdim=True)
    cache = arm_fast.CachedRollout(base, rows)
    cache.n_pixel = 3 * buf.H * buf.W

    joints = []
    for j in range(buf.action_dim):
        b = cache.roll(state, j, 0.0, want_pred=False)
        plus = cache.roll(state, j, probe, want_pred=False) - b
        minus = cache.roll(state, j, -probe, want_pred=False) - b
        fam = arm_fast.response_family(plus, minus)
        direction = np.asarray(fam["direction"], dtype=np.float64)
        plane, _ = plane_from_diff(torch.cat([minus, plus]).numpy())
        rot = _refine(
            arm_fast.conjugacy_losses(cache, state, mu, plane, j, DELTAS,
                                      COARSE, DT), COARSE,
            lambda g: arm_fast.conjugacy_losses(cache, state, mu, plane, j,
                                                DELTAS, g, DT))
        tra = _refine(
            arm_fast.translation_losses(cache, state, direction, j, DELTAS,
                                        COARSE, DT), COARSE,
            lambda g: arm_fast.translation_losses(cache, state, direction, j,
                                                  DELTAS, g, DT))
        win = "rotation" if rot["residual"] < tra["residual"] else "translation"
        scored = zs.score_phase(state.numpy().astype(np.float64),
                                mu.numpy().astype(np.float64), plane,
                                wrap(angles[:, min(j, angles.shape[1] - 1)]))
        joints.append({"joint": j + 1, "family": win,
                       "rotation": rot, "translation": tra,
                       "translation_score": round(fam["translation_score"], 4),
                       "response_magnitude": round(
                           float(np.median(np.linalg.norm(plus.numpy(), axis=1))), 5),
                       "kappa_vs_own_angle": scored["kappa"],
                       "winding": scored["winding"]})
    return {"tag": path.stem, "domain": buf.domain, "task": buf.task,
            "seed": int(path.stem.split("_s")[-1]),
            "discovery_uses_joint_labels": False,
            "joints": joints, "seconds": round(time.time() - t0, 1)}


def main(indir, buffer_path, output, batch):
    buf = Buffer(buffer_path)
    rows = []
    for f in sorted(glob.glob(str(Path(indir) / "dmc_rssm_dark_s*.pt"))):
        r = analyse(Path(f), buf, batch)
        rows.append(r)
        print(r["tag"], [(j["family"], round(j["rotation"]["residual"], 3),
                          round(j["translation"]["residual"], 3),
                          j["kappa_vs_own_angle"]) for j in r["joints"]],
              f"{r['seconds']}s", flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(
        {"domain": buf.domain, "task": buf.task, "rows": rows}, indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="runs/dmc_reacher")
    ap.add_argument("--buffer", default="runs/dmc/reacher_easy.npz")
    ap.add_argument("--output", default="runs/dmc_family_reacher.json")
    ap.add_argument("--batch", type=int, default=96)
    a = ap.parse_args()
    main(a.indir, a.buffer, a.output, a.batch)
