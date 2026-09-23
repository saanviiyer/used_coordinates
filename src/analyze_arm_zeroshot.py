"""Pose-free discovery of two independent robot-joint coordinates."""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

import arm_env
import interventions as iv
import train_arm_visual
import zeroshot as zs
from analyze_visual_zeroshot import FlatVisualAdapter
from visual_models import VisualGRU
from arm_models import build_arm_model


DELTAS = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)
GAIN_GRID = np.linspace(-2.0, 2.0, 81)


def action_column(model, joint):
    return model.n_pixel + 1 + int(joint)


def with_action_delta(model, row, joint, delta):
    changed = row.clone()
    col = action_column(model, joint)
    changed[:, col] = torch.clamp(changed[:, col] + delta, -1.0, 1.0)
    return changed


def _rotate(h, mu, plane, phi):
    c = (h - mu) @ plane
    ca, sa = torch.cos(phi), torch.sin(phi)
    c2 = torch.stack([ca * c[:, 0] - sa * c[:, 1],
                      sa * c[:, 0] + ca * c[:, 1]], 1)
    return h + (c2 - c) @ plane.T


def roll(model, h, rows, joint, delta=0.0, mu=None, plane=None, phi=None,
         state_only=False):
    if plane is not None:
        h = _rotate(h, mu, plane, phi)
    preds = []
    for t in range(rows.shape[1]):
        row = rows[:, t]
        if t == 0 and delta != 0.0:
            row = with_action_delta(model, row, joint, delta)
        h = model.step(row, h)
        if not state_only:
            preds.append(model.readout(h))
    return h if state_only else torch.stack(preds)


@torch.no_grad()
def collect_states(base, seed, batch=128, T=100, t_hit=64, deep=6,
                   horizon=4, variant="v1"):
    rng = np.random.default_rng(seed + 8_001)
    masked, _, mask, action, q = train_arm_visual.make_batch(
        batch, T, "dark", seed, rng, variant=variant)
    lo, hi = t_hit - deep, t_hit + horizon + 1
    masked[:, lo:hi] = 0.0
    mask[:, lo:hi] = 1.0
    h = torch.zeros(batch, base.hidden)
    for t in range(t_hit):
        h = base.step(masked[:, t], mask[:, t], action[:, t], h)
    rows = torch.cat([
        masked[:, t_hit:t_hit + horizon].flatten(2),
        mask[:, t_hit:t_hit + horizon, None],
        action[:, t_hit:t_hit + horizon]], dim=-1)
    return h, rows, q[:, t_hit].numpy()


@torch.no_grad()
def response_plane(model, h, rows, joint, probe=0.4):
    base = roll(model, h, rows, joint, state_only=True)
    diff = torch.cat([
        roll(model, h, rows, joint, delta=d, state_only=True) - base
        for d in (-probe, probe)])
    diff = diff - diff.mean(0, keepdim=True)
    _, singular, vectors = torch.linalg.svd(diff, full_matrices=False)
    variance = singular.square() / (singular.square().sum() + 1e-12)
    return vectors[:2].T.contiguous(), [round(float(v), 4)
                                       for v in variance[:4]]


@torch.no_grad()
def gain_on_plane(model, h, rows, joint, plane, mu=None, grid=GAIN_GRID):
    mu = h.mean(0, keepdim=True) if mu is None else mu
    targets = torch.stack([roll(model, h, rows, joint, delta=d)
                           for d in DELTAS])
    losses = []
    for gain in grid:
        preds = torch.stack([
            roll(model, h, rows, joint, mu=mu, plane=plane,
                 phi=torch.tensor(float(gain * d * arm_env.DT_ANGLE)))
            for d in DELTAS])
        losses.append(float(((preds - targets) ** 2).mean()))
    still = losses[int(np.argmin(np.abs(np.asarray(grid))))]
    best = int(np.argmin(losses))
    return {"gain": float(grid[best]), "loss": losses[best],
            "loss_still": still, "residual": losses[best] / (still + 1e-12)}


def analyse(path: Path, batch=128, n_null=4, variant="v1"):
    tag = path.stem
    _, _, cond, seed_text = tag.split("_")
    seed = int(seed_text[1:])
    _, model_kind, cond, seed_text = tag.split("_")
    base = build_arm_model(model_kind)
    base.load_state_dict(torch.load(path, map_location="cpu")); base.eval()
    model = FlatVisualAdapter(base)
    h_fit, rows_fit, _ = collect_states(
        base, 61_001, batch=batch, variant=variant)
    h_test, rows_test, q_test = collect_states(
        base, 91_001, batch=batch, variant=variant)
    mu = h_fit.mean(0, keepdim=True)
    rng = np.random.default_rng(22_000 + seed)
    joints, planes = [], []
    for joint in (0, 1):
        plane, variance = response_plane(model, h_fit, rows_fit, joint)
        planes.append(plane.numpy())
        gain = gain_on_plane(model, h_fit, rows_fit, joint, plane, mu)
        cross_gain = gain_on_plane(model, h_fit, rows_fit, 1 - joint,
                                   plane, mu)
        nulls = iv.matched_random_planes(h_fit.numpy(), plane.numpy(), rng,
                                         n_planes=n_null)
        nulls = nulls[0] if isinstance(nulls, tuple) else nulls
        null_residuals = [gain_on_plane(
            model, h_fit, rows_fit, joint,
            torch.as_tensor(p, dtype=h_fit.dtype), mu)["residual"]
            for p in nulls]
        own = zs.score_phase(h_test.numpy(), mu.numpy(), plane.numpy(),
                             q_test[:, joint])
        other = zs.score_phase(h_test.numpy(), mu.numpy(), plane.numpy(),
                               q_test[:, 1 - joint])
        distal = zs.score_phase(
            h_test.numpy(), mu.numpy(), plane.numpy(),
            arm_env.wrap(q_test[:, 0] + q_test[:, 1]))
        joints.append({
            "joint": joint + 1, "response_var_frac": variance,
            "gain": round(gain["gain"], 4),
            "residual": round(gain["residual"], 4),
            "null_residual_median": round(float(np.median(null_residuals)), 4),
            "own_kappa": own["kappa"], "own_winding": own["winding"],
            "other_kappa": other["kappa"],
            "absolute_distal_kappa": distal["kappa"],
            "cross_action_residual": round(cross_gain["residual"], 4),
            "cross_action_gain": round(cross_gain["gain"], 4),
        })
    angles = iv.principal_angles(planes[0], planes[1])
    return {"tag": tag, "cond": cond, "seed": seed, "variant": variant,
            "discovery_uses_joint_labels": False,
            "scoring_uses_joint_labels": True,
            "fit_states": int(batch), "held_out_states": int(batch),
            "joints": joints,
            "joint_plane_angles_deg": [round(float(x), 2) for x in angles]}


def main(indir: Path, output: Path, batch: int, n_null: int, conds, variant):
    rows = []
    for filename in sorted(glob.glob(str(indir / "arm_*_*.pt"))):
        if Path(filename).stem.split("_")[2] not in conds:
            continue
        row = analyse(Path(filename), batch=batch, n_null=n_null,
                      variant=variant)
        rows.append(row)
        compact = [(j["residual"], j["own_kappa"], j["other_kappa"],
                    j["absolute_distal_kappa"])
                   for j in row["joints"]]
        print(row["tag"], compact, flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("runs/arm_visual"))
    ap.add_argument("--output", type=Path,
                    default=Path("runs/arm_visual_zeroshot.json"))
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--conds", default="dark,lit,shuffle")
    ap.add_argument("--variant", choices=("v1", "v2", "v3", "v4", "v5", "v6", "v7"), default="v1")
    args = ap.parse_args()
    main(args.indir, args.output, args.batch, args.n_null,
         set(args.conds.split(",")), args.variant)
