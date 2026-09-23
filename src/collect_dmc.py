"""Pre-render a replay buffer from a control-suite domain.

Rendering runs at roughly a hundred frames a second, so generating batches on
the fly would cost more wall clock than training. Episodes are rendered once,
stored as uint8, and sampled as windows during training.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import dmc_env


def main(domain, task, episodes, length, seed, out, height, width):
    frames = np.empty((episodes, length, 3, height, width), dtype=np.uint8)
    actions, angles = None, None
    t0 = time.time()
    for e in range(episodes):
        f, a, q = dmc_env.rollout(1, length, domain, task, seed=seed + e,
                                  height=height, width=width)
        if actions is None:
            actions = np.empty((episodes, length, a.shape[-1]),
                               dtype=np.float32)
            angles = np.empty((episodes, length, q.shape[-1]),
                              dtype=np.float32)
        frames[e] = (f[0] * 255).astype(np.uint8)
        actions[e], angles[e] = a[0], q[0]
        if (e + 1) % 16 == 0:
            done = (e + 1) * length
            print(f"{e+1}/{episodes} episodes, {done} frames, "
                  f"{done / (time.time() - t0):.0f} fps", flush=True)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out, frames=frames, actions=actions, angles=angles,
                        domain=domain, task=task)
    print(f"wrote {out}: frames {frames.shape} uint8, actions "
          f"{actions.shape}, angles {angles.shape}, "
          f"{Path(out).stat().st_size / 1e6:.0f} MB, "
          f"{time.time() - t0:.0f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", default="reacher")
    ap.add_argument("--task", default="easy")
    ap.add_argument("--episodes", type=int, default=128)
    ap.add_argument("--length", type=int, default=160)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=64)
    ap.add_argument("--out", default="runs/dmc/reacher_easy.npz")
    a = ap.parse_args()
    main(a.domain, a.task, a.episodes, a.length, a.seed, a.out, a.height,
         a.width)
