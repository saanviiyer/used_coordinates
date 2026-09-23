"""Does the second joint become more readable without becoming more used?

The dissociation section rested on `arm_torus_calib.py`, which swept the
comparison horizon on a single checkpoint of the *v2* environment.  The
manuscript reports the *v3* serial arm, and the v2 checkpoints no longer
exist, so those numbers could not be quoted next to Table 1 without comparing
two different environments.

This sweeps the same horizons over the checkpoints the manuscript actually
reports: variant v3, dark condition, both architectures, every seed, under the
same `mode="search"`, `batch` and `n_null` as the Serial arm row.  Writes after
every analysis, and skips work already on disk, so it can be interrupted and
restarted.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import arm_torus

HORIZONS = (4, 8, 12)


def key(row):
    return (row["tag"], row["horizon"], bool(row["untrained"]))


def main(indir, output, variant, mode, batch, n_null, probe, untrained,
         horizons):
    indir, output = Path(indir), Path(output)
    rows = []
    if output.exists():
        rows = json.loads(output.read_text())["rows"]
        print(f"resuming from {len(rows)} rows already on disk")
    done = {key(r) for r in rows}

    todo = []
    for model_kind in ("gru", "modular"):
        for seed in range(8):
            ckpt = indir / f"arm_{model_kind}_dark_s{seed}.pt"
            if not ckpt.exists():
                continue
            for horizon in horizons:
                for unt in ((False, True) if untrained else (False,)):
                    if (ckpt.stem, horizon, unt) not in done:
                        todo.append((ckpt, horizon, unt))

    print(f"{len(todo)} analyses to run")
    for i, (ckpt, horizon, unt) in enumerate(todo, 1):
        t0 = time.time()
        r = arm_torus.analyse(ckpt, batch=batch, n_null=n_null,
                              variant=variant, mode=mode, horizon=horizon,
                              probe=probe, untrained=unt)
        rows.append(r)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(
            {"mode": mode, "variant": variant, "probe": probe,
             "batch": batch, "n_null": n_null, "rows": rows}, indent=2))
        p0, p1 = r["planes"][0], r["planes"][1]
        print(f"[{i}/{len(todo)}] {ckpt.stem} H={horizon} "
              f"{'untrained' if unt else 'trained'} "
              f"| j0 resid={p0['residual']:.3f} null={p0['null_residual_median']:.3f} "
              f"k={p0['best_kappa']:.3f} "
              f"| j1 resid={p1['residual']:.3f} null={p1['null_residual_median']:.3f} "
              f"k={p1['best_kappa']:.3f} | {time.time() - t0:.0f}s", flush=True)
    print(f"done: {len(rows)} rows -> {output}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", type=Path, default=Path("runs/arm_v3"))
    ap.add_argument("--output", type=Path,
                    default=Path("runs/arm_torus_horizon_v3.json"))
    ap.add_argument("--variant", default="v3")
    ap.add_argument("--mode", default="search")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--n-null", type=int, default=4)
    ap.add_argument("--probe", type=float, default=0.4)
    ap.add_argument("--untrained", action="store_true")
    ap.add_argument("--horizons", default="4,8,12")
    a = ap.parse_args()
    main(a.indir, a.output, a.variant, a.mode, a.batch, a.n_null, a.probe,
         a.untrained, tuple(int(x) for x in a.horizons.split(",")))
