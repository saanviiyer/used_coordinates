"""Is joint 2 visible at all?

The pre-specified floor in PLAN_FIETE_D145.md says a negative result for the
second circle must be checked against the environment before it is read as a
property of the model.  If moving joint 2 barely changes the image, and if
joint 2 is barely decodable from the image, then a predictive world model has
no incentive to carry a coordinate for it and the finding is about the arm, not
about world models.

Everything here is a property of the renderer.  No trained model is involved.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import arm_env
import arm_env_v2
import arm_env_v4
import arm_env_v5


def sensitivity(env, n=512, delta=0.12, seed=0):
    """Mean absolute image change caused by moving each joint alone."""
    rng = np.random.default_rng(seed)
    q = rng.uniform(-np.pi, np.pi, (n, 2))
    base = env.render(q, noise=0.0)
    out = {}
    for joint in (0, 1):
        step = np.zeros_like(q)
        step[:, joint] = delta
        moved = env.render(arm_env.wrap(q + step), noise=0.0)
        diff = np.abs(moved - base)
        out[f"joint{joint + 1}_mean_abs_change"] = float(diff.mean())
        out[f"joint{joint + 1}_changed_pixel_frac"] = float(
            (diff.max(axis=1) > 0.02).mean())
    step = np.full_like(q, delta)
    both = env.render(arm_env.wrap(q + step), noise=0.0)
    out["both_joints_mean_abs_change"] = float(np.abs(both - base).mean())
    out["ratio_joint1_over_joint2"] = (
        out["joint1_mean_abs_change"]
        / max(out["joint2_mean_abs_change"], 1e-12))
    return out


def decodability(env, n=4000, seed=1, ridge=1.0):
    """Cross-validated circular decode of each angle from raw pixels.

    A linear readout is a weak decoder, so this is a lower bound on visibility.
    It is reported for the actuator angles and for the absolute distal angle
    q1+q2, which is the variable the rendered distal link actually depends on.
    """
    rng = np.random.default_rng(seed)
    q = rng.uniform(-np.pi, np.pi, (n, 2))
    frames = env.render(q, noise=0.01, rng=rng).reshape(n, -1)
    frames = frames - frames.mean(0, keepdims=True)
    half = n // 2
    out = {}
    targets = {"q1": q[:, 0], "q2": q[:, 1],
               "q1_plus_q2": arm_env.wrap(q[:, 0] + q[:, 1])}
    for name, angle in targets.items():
        y = np.stack([np.cos(angle), np.sin(angle)], 1)
        X, Y = frames[:half], y[:half]
        gram = X.T @ X + ridge * np.eye(X.shape[1])
        weights = np.linalg.solve(gram, X.T @ Y)
        pred = frames[half:] @ weights
        got = np.arctan2(pred[:, 1], pred[:, 0])
        residual = np.arctan2(np.sin(got - angle[half:]),
                              np.cos(got - angle[half:]))
        out[name] = {
            "held_out_concentration": float(np.abs(np.mean(
                np.exp(1j * residual)))),
            "median_abs_error_deg": float(np.degrees(np.median(
                np.abs(residual))))}
    return out


def main(output):
    report = {}
    for name, env in (("v1", arm_env), ("v2", arm_env_v2),
                      ("v4", arm_env_v4), ("v5", arm_env_v5)):
        report[name] = {"sensitivity": sensitivity(env),
                        "linear_decodability": decodability(env)}
        s = report[name]["sensitivity"]
        d = report[name]["linear_decodability"]
        print(name,
              "img change j1 %.5f j2 %.5f ratio %.2f" % (
                  s["joint1_mean_abs_change"], s["joint2_mean_abs_change"],
                  s["ratio_joint1_over_joint2"]),
              "| decode kappa q1 %.3f q2 %.3f q1+q2 %.3f" % (
                  d["q1"]["held_out_concentration"],
                  d["q2"]["held_out_concentration"],
                  d["q1_plus_q2"]["held_out_concentration"]), flush=True)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text(json.dumps(report, indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path,
                    default=Path("runs/arm_env_diagnostic.json"))
    main(ap.parse_args().output)
