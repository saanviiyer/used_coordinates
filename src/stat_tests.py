"""Paired uncertainty and exact tests for the main representation claims.

Units are matched by architecture and seed. The script deliberately reports
effect sizes and confidence intervals rather than treating individual hidden
states as independent samples.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def nested(row, path):
    for key in path.split("."):
        row = row[int(key)] if isinstance(row, list) else row[key]
    return float(row)


def paired_values(rows, path, cond_a="dark", cond_b="lit", recurrent_only=False):
    allowed = {"gru", "rnn"} if recurrent_only else {"gru", "rnn", "tf"}
    index = {(r["kind"], r["seed"], r["cond"]): r for r in rows if r["kind"] in allowed}
    pairs = []
    for kind in sorted(allowed):
        seeds = sorted({seed for k, seed, cond in index if k == kind and cond == cond_a})
        for seed in seeds:
            a = index.get((kind, seed, cond_a)); b = index.get((kind, seed, cond_b))
            if a is not None and b is not None:
                pairs.append((kind, seed, nested(a, path), nested(b, path)))
    return pairs


def exact_sign_flip_p(differences):
    """Two-sided exact paired randomization test on the mean difference."""
    d = np.asarray(differences, dtype=float)
    observed = abs(d.mean())
    null = []
    for signs in itertools.product((-1.0, 1.0), repeat=len(d)):
        null.append(abs((d * np.asarray(signs)).mean()))
    return float(np.mean(np.asarray(null) >= observed - 1e-12))


def paired_summary(pairs, bootstrap=20000, seed=41):
    a = np.asarray([p[2] for p in pairs]); b = np.asarray([p[3] for p in pairs])
    d = a - b
    rng = np.random.default_rng(seed)
    draws = d[rng.integers(0, len(d), size=(bootstrap, len(d)))].mean(axis=1)
    return {
        "n_pairs": len(d),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "mean_difference": float(d.mean()),
        "ci95_difference": [float(x) for x in np.quantile(draws, [0.025, 0.975])],
        "exact_p": exact_sign_flip_p(d),
        "pairs": [{"kind": k, "seed": s, "a": va, "b": vb} for k, s, va, vb in pairs],
    }


def run(input_path: Path, output_path: Path):
    rows = json.loads(input_path.read_text())
    specs = {
        "fidelity_dark_vs_lit": ("dark_regime.align.concentration", "dark", "lit", False),
        "fidelity_dark_vs_shuffle": ("dark_regime.align.concentration", "dark", "shuffle", False),
        "heading_error_dark_vs_lit": ("dark_regime.head_err_deg", "dark", "lit", False),
        "heading_error_dark_vs_shuffle": ("dark_regime.head_err_deg", "dark", "shuffle", False),
        "h1_gap_dark_vs_lit": ("dark_regime.h1_gap", "dark", "lit", False),
        "phase_persistence_dark_vs_lit": ("steer_persist.-1", "dark", "lit", True),
        "phase_persistence_dark_vs_shuffle": ("steer_persist.-1", "dark", "shuffle", True),
        "steering_gain_dark_vs_shuffle": ("steer.slope", "dark", "shuffle", True),
    }
    result = {}
    for name, (path, a, b, recurrent_only) in specs.items():
        pairs = paired_values(rows, path, a, b, recurrent_only)
        result[name] = paired_summary(pairs)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=Path("runs/analysis.json"))
    parser.add_argument("--output", type=Path, default=Path("runs/statistics.json"))
    args = parser.parse_args()
    result = run(args.input, args.output)
    for name, row in result.items():
        lo, hi = row["ci95_difference"]
        print(f"{name:38s} delta={row['mean_difference']:+.4f} "
              f"95% CI [{lo:+.4f}, {hi:+.4f}] p={row['exact_p']:.5f} n={row['n_pairs']}")
