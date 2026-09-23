"""Label-free action--latent conjugacy for pixel world models.

The visual model has a structured step signature (frame, mask, action).  A
small adapter exposes the same frozen-dynamics interface used by ``zeroshot``;
the discovery objective therefore sees only hidden states, actions, and the
model's own pixel predictions.  Position and heading are retained separately
and enter only after the plane is fixed, for evaluation.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

import interventions as iv
import train_visual
import zeroshot as zs
from visual_env import IMAGE_H, IMAGE_W
from visual_models import VisualGRU


class FlatVisualAdapter(torch.nn.Module):
    """Expose ``VisualGRU`` through the vector-state interface in zeroshot."""

    def __init__(self, model: VisualGRU):
        super().__init__()
        self.model = model
        self.hidden = model.hidden
        self.n_pixel = 3 * IMAGE_H * IMAGE_W

    def step(self, row, h):
        frame = row[:, :self.n_pixel].reshape(-1, 3, IMAGE_H, IMAGE_W)
        mask = row[:, self.n_pixel]
        action = row[:, self.n_pixel + 1:self.n_pixel + 3]
        return self.model.step(frame, mask, action, h)

    def readout(self, h):
        return self.model.readout(h)


@torch.no_grad()
def collect_states(model, seed, batch=192, T=110, t_hit=70, deep=6,
                   horizon=4, domain="rectangle", palette=0, lighting=1.0):
    """Collect a forced-blackout state cloud; pose is held out for scoring."""
    rng = np.random.default_rng(seed + 9_001)
    masked, _, mask, action, pos, head = train_visual.make_batch(
        batch, T, "dark", seed, rng, domain=domain, palette=palette,
        lighting=lighting)
    lo, hi = t_hit - deep, t_hit + horizon + 1
    masked[:, lo:hi] = 0.0
    mask[:, lo:hi] = 1.0
    h = torch.zeros(batch, model.hidden)
    for t in range(t_hit):
        h = model.step(masked[:, t], mask[:, t], action[:, t], h)
    pixels = masked[:, t_hit:t_hit + horizon].flatten(2)
    rows = torch.cat([pixels, mask[:, t_hit:t_hit + horizon, None],
                      action[:, t_hit:t_hit + horizon]], dim=-1)
    return h, rows, head[:, t_hit].numpy(), pos[:, t_hit].numpy()


def analyse(path, batch=192, n_null=8, domain="rectangle", palette=0,
            lighting=1.0):
    tag = path.stem
    _, _, cond, seed_text = tag.split("_")
    seed = int(seed_text[1:])
    base = VisualGRU()
    base.load_state_dict(torch.load(path, map_location="cpu"))
    base.eval()
    model = FlatVisualAdapter(base)

    h_fit, rows_fit, _, _ = collect_states(
        base, 54321, batch=batch, domain=domain, palette=palette,
        lighting=lighting)
    h_test, rows_test, head_test, _ = collect_states(
        base, 99991, batch=batch, domain=domain, palette=palette,
        lighting=lighting)
    probe_delta = 0.4
    plane, variance = zs.response_plane(
        model, h_fit, rows_fit, deltas=(-probe_delta, probe_delta))
    deltas = (-0.3, -0.2, -0.1, 0.1, 0.2, 0.3)
    gain = zs.gain_on_plane(
        model, h_fit, rows_fit, plane, deltas=deltas,
        grid=np.linspace(-2.0, 2.0, 81))
    mu = h_fit.mean(0, keepdim=True).numpy()

    rng = np.random.default_rng(7_000 + seed)
    nulls = iv.matched_random_planes(h_fit.numpy(), plane, rng,
                                     n_planes=n_null)
    nulls = nulls[0] if isinstance(nulls, tuple) else nulls
    null_residuals = [zs.gain_on_plane(
        model, h_fit, rows_fit, q, deltas=deltas,
        grid=np.linspace(-2.0, 2.0, 81))["residual"] for q in nulls]
    return {
        "tag": tag, "cond": cond, "seed": seed,
        "discovery_uses_pose": False, "scoring_uses_pose": True,
        "fit_states": int(batch), "held_out_states": int(batch),
        "response_var_frac": variance,
        "gain": round(gain["gain"], 4),
        "residual": round(gain["residual"], 4),
        "null_residual_median": round(float(np.median(null_residuals)), 4),
        "held_out": zs.score_phase(h_test.numpy(), mu, plane, head_test),
        "domain": domain, "palette": palette, "lighting": lighting,
    }


def main(indir: Path, output: Path, batch: int, n_null: int, domain: str,
         palette: int, lighting: float):
    rows = []
    for filename in sorted(glob.glob(str(indir / "visual_gru_*.pt"))):
        row = analyse(Path(filename), batch=batch, n_null=n_null,
                      domain=domain, palette=palette,
                      lighting=lighting)
        rows.append(row)
        print(row["tag"], "residual", row["residual"], "gain", row["gain"],
              "kappa", row["held_out"]["kappa"], flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"rows": rows}, indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("runs/visual"))
    ap.add_argument("--output", type=Path,
                    default=Path("runs/visual_zeroshot.json"))
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--n-null", type=int, default=8)
    ap.add_argument("--domain", choices=("rectangle", "ellipse"),
                    default="rectangle")
    ap.add_argument("--palette", type=int, default=0)
    ap.add_argument("--lighting", type=float, default=1.0)
    args = ap.parse_args()
    main(args.indir, args.output, args.batch, args.n_null, args.domain,
         args.palette, args.lighting)
