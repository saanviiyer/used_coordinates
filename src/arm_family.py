"""Which one-parameter group does each actuator induce on the latent state?

The estimator inherited from the arm work asks a narrower question than it
appears to: it tests whether an action acts as a *rotation* in a plane.  That
is one group action among many, and it cannot read a bounded joint, a position,
or any other monotone coordinate.  Most actuated joints in standard continuous
control benchmarks are bounded hinges, so the restriction is not academic; it
decides whether the criterion can be pointed at a larger system at all.

This module fits both families on the same targets and the same scale, and
reports which one the action actually induces.  The answer is itself a result:
a model that carries a bounded joint as a shift along a direction and an
unbounded one as a turn in a plane is telling us something about what it built.

Discovery uses actions and the frozen model's own predictions.  Joint angles
enter afterwards, for scoring only.
"""
from __future__ import annotations

import argparse
import glob
import json
import time
from pathlib import Path

import numpy as np
import torch

import arm_env
import arm_fast
from analyze_arm_zeroshot import DELTAS, collect_states
from arm_models import build_arm_model
from arm_torus import COARSE, FINE_HALFWIDTH, FINE_POINTS, plane_from_diff

TRANSLATION_GRID = np.unique(np.concatenate(
    [np.linspace(-8.0, 8.0, 129), [0.0]]))


def signed_responses(cache, h, joint, probe=0.4):
    """Response to +probe and to -probe, kept separate.

    ``arm_torus.response_diff`` concatenates the two, which is right for
    finding a plane and destroys exactly the signal the family detector needs:
    a translation's mean response is zero once the two signs are pooled.
    """
    base = cache.roll(h, joint, 0.0, want_pred=False)
    plus = cache.roll(h, joint, probe, want_pred=False) - base
    minus = cache.roll(h, joint, -probe, want_pred=False) - base
    return plus, minus


def _refine(losses, grid, refit):
    still = float(losses[int(np.argmin(np.abs(grid)))])
    best = int(np.argmin(losses))
    at_edge = best in (0, len(grid) - 1)
    fine = np.linspace(grid[best] - FINE_HALFWIDTH,
                       grid[best] + FINE_HALFWIDTH, FINE_POINTS)
    fine_losses = refit(fine)
    b2 = int(np.argmin(fine_losses))
    return {"gain": float(fine[b2]),
            "residual": float(fine_losses[b2] / (still + 1e-12)),
            "gain_at_grid_edge": bool(at_edge)}


def fit_families(cache, h, mu, joint, probe=0.4):
    """Fit rotation and translation on the same targets, and compare."""
    plus, minus = signed_responses(cache, h, joint, probe)
    family = arm_fast.response_family(plus, minus)
    direction = np.asarray(family["direction"], dtype=np.float64)

    plane, variance = plane_from_diff(torch.cat([minus, plus]).numpy())
    rot = _refine(
        arm_fast.conjugacy_losses(cache, h, mu, plane, joint, DELTAS, COARSE,
                                  arm_env.DT_ANGLE),
        COARSE,
        lambda g: arm_fast.conjugacy_losses(cache, h, mu, plane, joint,
                                            DELTAS, g, arm_env.DT_ANGLE))
    tra = _refine(
        arm_fast.translation_losses(cache, h, direction, joint, DELTAS,
                                    TRANSLATION_GRID, arm_env.DT_ANGLE),
        TRANSLATION_GRID,
        lambda g: arm_fast.translation_losses(cache, h, direction, joint,
                                              DELTAS, g, arm_env.DT_ANGLE))
    winner = "rotation" if rot["residual"] < tra["residual"] else "translation"
    return {"joint": joint + 1,
            "translation_score": round(family["translation_score"], 4),
            "response_mean_norm": round(family["mean_norm"], 5),
            "response_spread": round(family["residual_spread"], 5),
            "rotation": {k: (round(v, 4) if isinstance(v, float) else v)
                         for k, v in rot.items()},
            "translation": {k: (round(v, 4) if isinstance(v, float) else v)
                            for k, v in tra.items()},
            "family": winner,
            "margin": round(abs(rot["residual"] - tra["residual"]), 4),
            "response_var_frac": variance}


def analyse(path: Path, batch=128, variant="v4", probe=0.4):
    tag = path.stem
    _, model_kind, cond, seed_text = tag.split("_")
    base = build_arm_model(model_kind)
    base.load_state_dict(torch.load(path, map_location="cpu"))
    base.eval()
    t0 = time.time()
    h, rows, _ = collect_states(base, 61_001, batch=batch, variant=variant)
    mu = h.mean(0, keepdim=True)
    cache = arm_fast.CachedRollout(base, rows)
    joints = [fit_families(cache, h, mu, j, probe) for j in (0, 1)]
    return {"tag": tag, "model_kind": model_kind, "cond": cond,
            "seed": int(seed_text[1:]), "variant": variant,
            "discovery_uses_joint_labels": False,
            "joints": joints, "seconds": round(time.time() - t0, 1)}


def main(indir, output, batch, variant, conds, models):
    rows = []
    for filename in sorted(glob.glob(str(indir / "arm_*_*.pt"))):
        stem = Path(filename).stem.split("_")
        if stem[1] not in models or stem[2] not in conds:
            continue
        row = analyse(Path(filename), batch, variant)
        rows.append(row)
        print(row["tag"], [(j["family"], j["rotation"]["residual"],
                            j["translation"]["residual"],
                            j["translation_score"]) for j in row["joints"]],
              f"{row['seconds']}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"variant": variant, "rows": rows}, indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("runs/arm_v4"))
    ap.add_argument("--output", type=Path,
                    default=Path("runs/arm_family_v4.json"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--variant", default="v4")
    ap.add_argument("--conds", default="dark")
    ap.add_argument("--models", default="gru,modular")
    a = ap.parse_args()
    main(a.indir, a.output, a.batch, a.variant, set(a.conds.split(",")),
         set(a.models.split(",")))
