"""D1: pose-free discovery of a two-dimensional torus in an arm world model.

The per-joint estimator in ``analyze_arm_zeroshot`` searches for each joint's
plane independently.  When the two responses share a subspace it recovers the
same circle twice, which is what the v1/v2/v3 diagnostics show: the plane
discovered for joint 2 scores 0.50-0.78 concentration against joint *1*.

A torus is a pair of commuting circle actions, so this module discovers the
two circles jointly.  In ``deflate`` mode the first circle is found, its span
is removed, and the second is sought only in the orthogonal complement.
Discovery uses actions and the frozen model's own predictions.  Joint angles
enter after discovery, for scoring only.
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
import interventions as iv
import zeroshot as zs
from analyze_arm_zeroshot import DELTAS, collect_states
from arm_models import build_arm_model

# Integer combinations of the two joint angles.  (1,0) is the proximal
# actuator, (0,1) the distal actuator, (1,1) the absolute distal-link angle
# that the rendered image actually depends on.  Scoring only.
BASES = ((1, 0), (0, 1), (1, 1), (1, -1), (2, 0), (0, 2), (2, 1), (1, 2))

# Widened twice.  The repository's +-2 grid railed in every old diagnostic;
# +-4 then railed for the parity arm's second plane, which carries winding 2.
# The fine pass must span at least half the coarse spacing or it cannot reach
# the true minimum between coarse samples.  0.0 is present so the no-rotation
# reference is exact.
COARSE = np.unique(np.concatenate([np.linspace(-8.0, 8.0, 129), [0.0]]))
FINE_HALFWIDTH = 0.07
FINE_POINTS = 29


def orthonormal(M):
    q, _ = np.linalg.qr(np.asarray(M, dtype=np.float64))
    return q


def deflate_matrix(dim, taken):
    P = orthonormal(taken)
    return np.eye(dim) - P @ P.T


SEARCH_GRID = np.linspace(-6.0, 6.0, 33)


def subspace_from_diff(diff, projector=None, n_comp=6):
    """Top-``n_comp`` response directions, used as a search space."""
    d = np.asarray(diff, dtype=np.float64)
    d = d - d.mean(0, keepdims=True)
    if projector is not None:
        d = d @ projector.T
    _, _, vectors = np.linalg.svd(d, full_matrices=False)
    return vectors[:n_comp].T


def search_plane(cache, h, mu, subspace, joint, n_candidates=24, seed=0):
    """Best conjugate plane inside a response subspace.

    The top-two principal directions of a response are the directions it moves
    *most*, which need not be the plane the action *rotates*.  This searches
    the subspace directly on the conjugacy objective, with the principal plane
    included so the search cannot do worse than the default.
    """
    rng = np.random.default_rng(seed)
    k = subspace.shape[1]
    candidates = [subspace[:, :2]]
    for _ in range(n_candidates):
        q, _ = np.linalg.qr(rng.normal(size=(k, 2)))
        candidates.append(subspace @ q)
    best, best_cost = candidates[0], np.inf
    for plane in candidates:
        losses = arm_fast.conjugacy_losses(cache, h, mu, plane, joint, DELTAS,
                                           SEARCH_GRID, arm_env.DT_ANGLE)
        still = float(losses[int(np.argmin(np.abs(SEARCH_GRID)))])
        cost = float(np.min(losses)) / (still + 1e-12)
        if cost < best_cost:
            best, best_cost = plane, cost
    return best, best_cost


def plane_from_diff(diff, projector=None, n_comp=2):
    d = np.asarray(diff, dtype=np.float64)
    d = d - d.mean(0, keepdims=True)
    if projector is not None:
        d = d @ projector.T
    _, singular, vectors = np.linalg.svd(d, full_matrices=False)
    variance = singular ** 2 / (np.sum(singular ** 2) + 1e-12)
    return vectors[:n_comp].T, [round(float(v), 4) for v in variance[:4]]


def conjugacy(cache, h, mu, plane, joint):
    """Best rotation-imitates-action fit on ``plane``.  No pose labels."""
    losses = arm_fast.conjugacy_losses(cache, h, mu, plane, joint, DELTAS,
                                       COARSE, arm_env.DT_ANGLE)
    still = float(losses[int(np.argmin(np.abs(COARSE)))])
    best = int(np.argmin(losses))
    at_edge = best in (0, len(COARSE) - 1)
    fine = np.linspace(COARSE[best] - FINE_HALFWIDTH,
                       COARSE[best] + FINE_HALFWIDTH, FINE_POINTS)
    fine_losses = arm_fast.conjugacy_losses(cache, h, mu, plane, joint,
                                            DELTAS, fine, arm_env.DT_ANGLE)
    b2 = int(np.argmin(fine_losses))
    return {"gain": float(fine[b2]),
            "residual": float(fine_losses[b2] / (still + 1e-12)),
            "gain_at_grid_edge": bool(at_edge)}


def response_diff(cache, h, joint, probe=0.4):
    """State displacement caused by perturbing one actuator.  No labels."""
    base = cache.roll(h, joint, 0.0, want_pred=False)
    return torch.cat([cache.roll(h, joint, -probe, want_pred=False) - base,
                      cache.roll(h, joint, probe, want_pred=False) - base])


def phase_of(h, mu, plane):
    c = (np.asarray(h, dtype=np.float64) - np.asarray(mu, dtype=np.float64)) \
        @ np.asarray(plane)
    return np.arctan2(c[:, 1], c[:, 0])


def phase_shift(cache, h, mu, plane, moving_joint, probe=0.4):
    """Median absolute phase movement on ``plane`` when one actuator acts.

    A torus needs two commuting circle actions: each action should turn its
    own circle and leave the other alone.  Reported in radians.
    """
    base = cache.roll(h, moving_joint, 0.0, want_pred=False).numpy()
    moved = cache.roll(h, moving_joint, probe, want_pred=False).numpy()
    mu_np = mu.numpy()
    d = phase_of(moved, mu_np, plane) - phase_of(base, mu_np, plane)
    return float(np.median(np.abs(np.arctan2(np.sin(d), np.cos(d)))))


def uniformity(h_test, mu, plane, angle, n_bins=12):
    """Is the discovered circle a uniform parameterisation of ``angle``?

    A continuous circle map has an integer degree, so a fitted gain of 1.8 with
    winding 1 is not a contradiction: it says the embedding wraps once but
    advances at an uneven rate.  This bins by the true angle and measures the
    local slope of latent phase against it, which separates a distorted
    degree-1 coordinate from a genuine double cover.
    """
    phase = phase_of(h_test.numpy().astype(np.float64),
                     mu.numpy().astype(np.float64), plane)
    edges = np.linspace(-np.pi, np.pi, n_bins + 1)
    slopes = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (angle >= lo) & (angle < hi)
        if sel.sum() < 8:
            continue
        a = angle[sel]
        z = np.exp(1j * phase[sel])
        # Local slope from the circular-circular regression of phase on angle.
        centred_a = a - a.mean()
        z = z * np.exp(-1j * np.angle(z.mean()))
        num = float(np.sum(centred_a * np.angle(z)))
        den = float(np.sum(centred_a ** 2)) + 1e-12
        slopes.append(num / den)
    if not slopes:
        return None
    slopes = np.asarray(slopes)
    return {"n_bins_used": int(len(slopes)),
            "median_local_slope": float(np.median(slopes)),
            "iqr_local_slope": float(np.subtract(*np.percentile(slopes,
                                                                [75, 25]))),
            "min_local_slope": float(slopes.min()),
            "max_local_slope": float(slopes.max())}


def score_bases(h_test, mu, plane, q_test):
    """Best integer combination a*q1 + b*q2 for this plane.  Evaluation only."""
    out = []
    for a, b in BASES:
        angle = arm_env.wrap(a * q_test[:, 0] + b * q_test[:, 1])
        s = zs.score_phase(h_test.numpy().astype(np.float64),
                           mu.numpy().astype(np.float64),
                           np.asarray(plane), angle)
        out.append({"basis": [a, b], "kappa": s["kappa"],
                    "winding": s["winding"]})
    out.sort(key=lambda r: -r["kappa"])
    return out


def discover(cache, h_fit, mu, order, mode, n_null, rng, probe=0.4):
    """Find one circle per actuator.  ``order`` fixes which is found first."""
    dim = h_fit.shape[1]
    diffs = {j: response_diff(cache, h_fit, j, probe).numpy() for j in (0, 1)}
    magnitude = {j: float(np.median(np.linalg.norm(diffs[j], axis=1)))
                 for j in (0, 1)}
    planes, variances, taken = {}, {}, None
    search_cost = {}
    for step, joint in enumerate(order):
        projector = None
        if mode in ("deflate", "search") and taken is not None:
            projector = deflate_matrix(dim, taken)
        plane, variance = plane_from_diff(diffs[joint], projector)
        if mode == "search":
            subspace = subspace_from_diff(diffs[joint], projector)
            plane, cost = search_plane(cache, h_fit, mu, subspace, joint,
                                       seed=17 + joint)
            search_cost[joint] = round(float(cost), 4)
        planes[joint], variances[joint] = plane, variance
        taken = plane if taken is None else np.concatenate([taken, plane], 1)

    rows = []
    for joint in (0, 1):
        plane = planes[joint]
        own = conjugacy(cache, h_fit, mu, plane, joint)
        cross = conjugacy(cache, h_fit, mu, plane, 1 - joint)
        nulls = iv.matched_random_planes(h_fit.numpy(),
                                         plane.astype(np.float32), rng,
                                         n_planes=n_null)
        nulls = nulls[0] if isinstance(nulls, tuple) else nulls
        null_res = [conjugacy(cache, h_fit, mu, p, joint)["residual"]
                    for p in nulls]
        rows.append({
            "joint": joint + 1,
            "response_var_frac": variances[joint],
            "response_magnitude": round(magnitude[joint], 5),
            "gain": round(own["gain"], 4),
            "residual": round(own["residual"], 4),
            "gain_at_grid_edge": own["gain_at_grid_edge"],
            "null_residual_median": round(float(np.median(null_res)), 4),
            "null_residual_min": round(float(np.min(null_res)), 4),
            "beats_null": bool(own["residual"] < float(np.median(null_res))),
            "cross_action_residual": round(cross["residual"], 4),
            "cross_action_gain": round(cross["gain"], 4),
            "self_phase_shift_rad": round(
                phase_shift(cache, h_fit, mu, plane, joint), 4),
            "crosstalk_phase_shift_rad": round(
                phase_shift(cache, h_fit, mu, plane, 1 - joint), 4),
            "search_cost": search_cost.get(joint),
        })
    return planes, rows


def analyse(path: Path, batch=128, n_null=4, variant="v3", mode="deflate",
            order=(0, 1), horizon=4, probe=0.4, deep=6, untrained=False):
    tag = path.stem
    _, model_kind, cond, seed_text = tag.split("_")
    seed = int(seed_text[1:])
    base = build_arm_model(model_kind)
    if untrained:
        torch.manual_seed(90_000 + seed)
        base = build_arm_model(model_kind)
    else:
        base.load_state_dict(torch.load(path, map_location="cpu"))
    base.eval()
    t0 = time.time()
    h_fit, rows_fit, _ = collect_states(base, 61_001, batch=batch,
                                        variant=variant, horizon=horizon,
                                        deep=deep)
    h_test, _, q_test = collect_states(base, 91_001, batch=batch,
                                       variant=variant, horizon=horizon,
                                       deep=deep)
    mu = h_fit.mean(0, keepdim=True)
    cache = arm_fast.CachedRollout(base, rows_fit)
    rng = np.random.default_rng(31_000 + seed)
    planes, rows = discover(cache, h_fit, mu, order, mode, n_null, rng,
                            probe=probe)
    for r in rows:
        plane = planes[r["joint"] - 1]
        r["bases"] = score_bases(h_test, mu, plane, q_test)
        r["best_basis"] = r["bases"][0]["basis"]
        r["best_kappa"] = r["bases"][0]["kappa"]
        a, b = r["best_basis"]
        r["best_winding"] = r["bases"][0]["winding"]
        r["gain_over_winding"] = (round(r["gain"] / r["best_winding"], 3)
                                  if r["best_winding"] else None)
        r["uniformity"] = uniformity(
            h_test, mu, plane,
            arm_env.wrap(a * q_test[:, 0] + b * q_test[:, 1]))
    angles = iv.principal_angles(planes[0].astype(np.float32),
                                 planes[1].astype(np.float32))
    return {"tag": tag, "model_kind": model_kind, "cond": cond, "seed": seed,
            "variant": variant, "mode": mode, "order": list(order),
            "horizon": horizon, "probe": probe, "deep": deep,
            "untrained": bool(untrained),
            "discovery_uses_joint_labels": False,
            "scoring_uses_joint_labels": True,
            "planes": rows,
            "plane_angles_deg": [round(float(x), 2) for x in angles],
            "seconds": round(time.time() - t0, 1)}


def main(indir, output, batch, n_null, variant, mode, conds, models, order,
         horizon=4, probe=0.4, deep=6, untrained=False):
    rows = []
    for filename in sorted(glob.glob(str(indir / "arm_*_*.pt"))):
        stem = Path(filename).stem.split("_")
        if stem[1] not in models or stem[2] not in conds:
            continue
        row = analyse(Path(filename), batch, n_null, variant, mode, order,
                      horizon, probe, deep, untrained)
        rows.append(row)
        print(row["tag"], mode,
              [(p["residual"], p["null_residual_median"], p["best_basis"],
                p["best_kappa"]) for p in row["planes"]],
              f"{row['seconds']}s", flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"mode": mode, "variant": variant,
                                  "horizon": horizon, "probe": probe,
                                  "deep": deep, "untrained": bool(untrained),
                                  "rows": rows}, indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("runs/arm_v3"))
    ap.add_argument("--output", type=Path, default=Path("runs/arm_torus.json"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--variant", choices=("v1", "v2", "v3", "v4", "v5", "v6", "v7"), default="v3")
    ap.add_argument("--mode", choices=("sequential", "deflate", "search"),
                    default="deflate")
    ap.add_argument("--conds", default="dark,lit,shuffle")
    ap.add_argument("--models", default="gru,modular")
    ap.add_argument("--order", default="0,1")
    ap.add_argument("--horizon", type=int, default=4)
    ap.add_argument("--probe", type=float, default=0.4)
    ap.add_argument("--deep", type=int, default=6)
    ap.add_argument("--untrained", action="store_true")
    a = ap.parse_args()
    main(a.indir, a.output, a.batch, a.n_null, a.variant, a.mode,
         set(a.conds.split(",")), set(a.models.split(",")),
         tuple(int(x) for x in a.order.split(",")),
         a.horizon, a.probe, a.deep, a.untrained)
