"""D1 instrument check: is the second circle absent, or merely unresolved?

The parent project already found this estimator to be horizon-sensitive: at a
one-step comparison horizon it fails outright and two steps recover it.  Joint
2's action response is roughly three times weaker than joint 1's, so before any
claim that the second circle does not exist, the probe window, probe size and
blackout depth are swept, and an untrained model supplies the floor.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import arm_torus


def main(checkpoint, output, variant, mode, batch, n_null):
    rows = []
    for untrained in (True, False):
        for horizon in (4, 8, 12):
            for probe in (0.4, 0.8):
                r = arm_torus.analyse(checkpoint, batch=batch, n_null=n_null,
                                      variant=variant, mode=mode,
                                      horizon=horizon, probe=probe,
                                      untrained=untrained)
                rows.append(r)
                print(("untrained" if untrained else "trained"),
                      "H", horizon, "probe", probe,
                      [(p["residual"], p["null_residual_median"],
                        p["response_magnitude"], p["best_basis"],
                        p["best_kappa"]) for p in r["planes"]], flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps({"checkpoint": str(checkpoint),
                                        "rows": rows}, indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--variant", default="v3")
    ap.add_argument("--mode", default="deflate")
    ap.add_argument("--batch", type=int, default=96)
    ap.add_argument("--n-null", type=int, default=3)
    a = ap.parse_args()
    main(a.checkpoint, a.output, a.variant, a.mode, a.batch, a.n_null)
