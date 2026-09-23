"""The validity gate: can the prior path predict during a blackout?

`PREREG_DMC.md` lists, before any model was trained, what would invalidate the
measurement.  The first item is the model failing to predict the arm at all:
if the blackout rollout carries no information about where the arm went, there
is no latent worth probing and a family verdict read off it means nothing.

This is also the gate for the arm sweeps, not only the control-suite one.  Two
successive free-nats floors silently switched the KL term off, leaving the
prior untrained; the prior is exactly what the model rolls on during a
blackout, so a checkpoint can train to a respectable reconstruction loss and
still be worthless to the criterion.  No statistic of the loss caught that.
This does, because it measures the prior path directly.

The comparator is frame persistence.  During a blackout the model sees nothing,
so the cheapest non-trivial prediction is "the scene has not moved since the
last frame I saw".  A world model that cannot beat that on blacked-out steps
has not learned dynamics; it has learned to hold still.

This is a gate, not a result.  It runs before `dmc_family.py` is read, and it
is reported whether it passes or fails.
"""
from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import torch

from arm_models import build_arm_model
from rssm import RSSM
from train_rssm_dmc import Buffer


class ArmSource:
    """The arm environments, wrapped to look like the control-suite buffer."""

    def __init__(self, variant):
        import train_arm_visual as TAV
        self.TAV, self.variant = TAV, variant
        self.domain, self.task = "arm", variant
        self.H = self.W = None
        self.action_dim = None

    def sample(self, B, T, cond, rng):
        masked, frames, mask, action, angles = self.TAV.make_batch(
            B, T, cond, int(rng.integers(1 << 30)), rng, variant=self.variant)
        return masked, frames, mask, action, angles

    def model(self, kind="rssm"):
        return build_arm_model(kind)


class BufferSource:
    def __init__(self, path):
        self.buf = Buffer(path)
        self.domain, self.task = self.buf.domain, self.buf.task

    def sample(self, B, T, cond, rng):
        return self.buf.sample(B, T, cond, rng)

    def model(self, kind="rssm"):
        return RSSM(image_h=self.buf.H, image_w=self.buf.W,
                    action_dim=self.buf.action_dim)


def _last_visible(frames, mask):
    """For each step, the most recent frame the model was allowed to encode.

    Before any frame is visible there is nothing to persist, so those steps are
    excluded from scoring rather than charged to either arm.
    """
    B, T = mask.shape
    out = torch.zeros_like(frames)
    valid = torch.zeros(B, T, dtype=torch.bool)
    held = frames[:, 0].clone()
    seen = mask[:, 0] < 0.5
    held[~seen] = 0.0
    for t in range(T):
        visible = mask[:, t] < 0.5
        held = torch.where(visible[:, None, None, None], frames[:, t], held)
        seen = seen | visible
        out[:, t] = held
        valid[:, t] = seen
    return out, valid


@torch.no_grad()
def check(path: Path, src, batch: int, T: int, seed: int) -> dict:
    sd = torch.load(path, map_location="cpu")
    # Tags are arm_<kind>_<cond>_s<seed> or dmc_rssm_<cond>_s<seed>.
    parts = path.stem.split("_")
    kind = parts[1] if len(parts) > 1 else "rssm"
    model = src.model(kind)
    model.load_state_dict(sd)
    model.eval()

    rng = np.random.default_rng(seed)
    masked, frames, mask, action, _ = src.sample(batch, T, "dark", rng)
    if hasattr(model, "observe"):
        pred, _, _ = model.observe(masked, mask, action, sample=False)
    else:
        pred = model(masked, mask, action)

    # Score one step ahead, on steps the model had to predict without seeing:
    # the target at t+1 is blacked out.  The model's prediction for it is
    # pred[:, t]; persistence's is the last frame it was shown.
    held, valid = _last_visible(frames, mask)
    tgt = frames[:, 1:]
    dark = (mask[:, 1:] > 0.5) & valid[:, :-1]
    if not dark.any():
        return {"tag": path.stem, "scored_steps": 0}

    model_se = ((pred[:, :-1] - tgt) ** 2).mean((2, 3, 4))
    persist_se = ((held[:, :-1] - tgt) ** 2).mean((2, 3, 4))
    m = float(model_se[dark].mean())
    p = float(persist_se[dark].mean())
    return {"tag": path.stem, "model_kind": kind,
            "seed": int(path.stem.split("_s")[-1]),
            "scored_steps": int(dark.sum()),
            "blackout_mse_model": round(m, 6),
            "blackout_mse_persistence": round(p, 6),
            "ratio_model_over_persistence": round(m / p, 4) if p else None,
            "beats_persistence": bool(m < p)}


def main(indir, buffer_path, output, batch, T, seed, variant, pattern):
    src = ArmSource(variant) if variant else BufferSource(buffer_path)
    rows = []
    for f in sorted(glob.glob(str(Path(indir) / pattern))):
        r = check(Path(f), src, batch, T, seed)
        rows.append(r)
        print(r["tag"], "model", r.get("blackout_mse_model"),
              "persistence", r.get("blackout_mse_persistence"),
              "ratio", r.get("ratio_model_over_persistence"),
              "PASS" if r.get("beats_persistence") else "FAIL", flush=True)
    n_pass = sum(1 for r in rows if r.get("beats_persistence"))
    verdict = "PASS" if n_pass == len(rows) and rows else (
        "PARTIAL" if n_pass else "FAIL")
    out = {"gate": "prereg_dmc_validity_1_model_predicts_the_arm",
           "comparator": "frame persistence on blacked-out steps",
           "domain": src.domain, "task": src.task,
           "n_models": len(rows), "n_beating_persistence": n_pass,
           "verdict": verdict,
           "median_ratio": round(float(np.median(
               [r["ratio_model_over_persistence"] for r in rows])), 4)
           if rows else None,
           "rows": rows}
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(out, indent=2))
    print(f"{verdict}: {n_pass}/{len(rows)} beat persistence -> {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", default="runs/dmc_reacher")
    ap.add_argument("--buffer", default="runs/dmc/reacher_easy.npz")
    ap.add_argument("--output", default="runs/dmc_validity_reacher.json")
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--T", type=int, default=48)
    ap.add_argument("--seed", type=int, default=91_000)
    ap.add_argument("--variant", default=None,
                    help="arm variant (v6, v7). Omit to use --buffer.")
    ap.add_argument("--pattern", default="dmc_rssm_dark_s*.pt")
    a = ap.parse_args()
    main(a.indir, a.buffer, a.output, a.batch, a.T, a.seed, a.variant,
         a.pattern)
