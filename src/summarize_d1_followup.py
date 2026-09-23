"""Gate verdicts for the parity arm and the decoupled mechanism.

Thresholds are those fixed in PLAN_FIETE_D145.md before either environment was
trained.  E1 asks whether the single circle is the absolute distal angle. E2
asks whether two visually separable degrees of freedom produce two used
coordinates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from stat_tests import exact_sign_flip_p


def load(path):
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else None


def basis_votes(rows, plane):
    votes = {}
    for r in rows:
        votes[tuple(r["planes"][plane]["best_basis"])] = votes.get(
            tuple(r["planes"][plane]["best_basis"]), 0) + 1
    return {str(list(k)): v for k, v in sorted(votes.items(),
                                               key=lambda x: -x[1])}


def mean_basis_kappa(rows, plane):
    agg = {}
    for r in rows:
        for b in r["planes"][plane]["bases"]:
            agg.setdefault(tuple(b["basis"]), []).append(b["kappa"])
    return {str(list(k)): round(float(np.mean(v)), 4)
            for k, v in sorted(agg.items(), key=lambda x: -np.mean(x[1]))}


def paired(a_rows, b_rows, plane, key, lower_is_better=True):
    a = {r["seed"]: r["planes"][plane] for r in a_rows}
    b = {r["seed"]: r["planes"][plane] for r in b_rows}
    seeds = sorted(set(a) & set(b))
    if not seeds:
        return None
    diff = np.asarray([a[s][key] - b[s][key] for s in seeds], dtype=float)
    favours = int(np.sum(diff < 0)) if lower_is_better else int(
        np.sum(diff > 0))
    return {"mean_a": float(np.mean([a[s][key] for s in seeds])),
            "mean_b": float(np.mean([b[s][key] for s in seeds])),
            "pairs_favouring_dark": favours, "n_pairs": len(seeds),
            "exact_p": exact_sign_flip_p(diff)}


def analyse(path, floor_path):
    data = load(path)
    if data is None:
        return {"verdict": "INCOMPLETE", "reason": f"{path} missing"}
    rows = data["rows"]
    floor = load(floor_path)
    out = {"n_checkpoints": len(rows), "by_model": {}}
    for kind in sorted({r["model_kind"] for r in rows}):
        sub = [r for r in rows if r["model_kind"] == kind]
        conds = {c: [r for r in sub if r["cond"] == c]
                 for c in sorted({r["cond"] for r in sub})}
        block = {"per_condition": {}, "controls": {}}
        for cond, cr in conds.items():
            block["per_condition"][cond] = [{
                "plane": j + 1, "n": len(cr),
                "mean_residual": float(np.mean(
                    [r["planes"][j]["residual"] for r in cr])),
                "mean_null": float(np.mean(
                    [r["planes"][j]["null_residual_median"] for r in cr])),
                "beats_null_n": sum(1 for r in cr
                                    if r["planes"][j]["residual"]
                                    < r["planes"][j]["null_residual_median"]),
                "mean_best_kappa": float(np.mean(
                    [r["planes"][j]["best_kappa"] for r in cr])),
                "mean_response_magnitude": float(np.mean(
                    [r["planes"][j]["response_magnitude"] for r in cr])),
                "basis_votes": basis_votes(cr, j),
                "mean_kappa_by_basis": mean_basis_kappa(cr, j),
            } for j in (0, 1)]
        for other in ("lit", "shuffle"):
            if "dark" in conds and other in conds:
                for j in (0, 1):
                    for metric, low in (("residual", True),
                                        ("best_kappa", False)):
                        r = paired(conds["dark"], conds[other], j, metric, low)
                        if r:
                            block["controls"][
                                f"plane{j+1}_{metric}_dark_vs_{other}"] = r
        if floor:
            # Match the floor to the architecture: an untrained modular model
            # and an untrained GRU do not have the same response scale.
            fr = [r for r in floor["rows"] if r["model_kind"] == kind]
            if fr:
                block["untrained_floor"] = {
                    f"plane{j+1}_response_magnitude": float(np.mean(
                        [r["planes"][j]["response_magnitude"] for r in fr]))
                    for j in (0, 1)}
                for cond in ("dark",):
                    for j, pl in enumerate(block["per_condition"].get(cond, [])):
                        base = block["untrained_floor"][
                            f"plane{j+1}_response_magnitude"]
                        pl["response_over_floor"] = round(
                            pl["mean_response_magnitude"] / max(base, 1e-9), 2)
        out["by_model"][kind] = block
    return out


def verdicts(v4, v5):
    result = {}
    for kind in ("gru", "modular"):
        block = v4.get("by_model", {}).get(kind)
        if not block:
            continue
        dark = block["per_condition"].get("dark")
        if not dark:
            continue
        p1 = dark[0]
        votes = p1["basis_votes"]
        n = p1["n"]
        sum_votes = votes.get("[1, 1]", 0)
        k = p1["mean_kappa_by_basis"]
        wins = (k.get("[1, 1]", 0) > k.get("[1, 0]", 0)
                and k.get("[1, 1]", 0) > k.get("[0, 1]", 0))
        result[f"E1_{kind}"] = {
            "verdict": "PASS" if (sum_votes >= 6 and wins) else "FAIL",
            "plane1_basis_votes": votes,
            "plane1_kappa_1_1": k.get("[1, 1]"),
            "plane1_kappa_1_0": k.get("[1, 0]"),
            "plane1_kappa_0_1": k.get("[0, 1]"),
            "plane1_beats_null": f"{p1['beats_null_n']}/{n}",
            "plane2_beats_null": f"{dark[1]['beats_null_n']}/{n}",
            "estimator_check": (
                "second plane also passed on a one-degree-of-freedom image; "
                "treat v5 with suspicion"
                if dark[1]["beats_null_n"] == n else
                "second plane fails here as expected, so the estimator is not "
                "manufacturing circles"),
        }
    for kind in ("gru", "modular"):
        block = v5.get("by_model", {}).get(kind)
        if not block:
            continue
        dark = block["per_condition"].get("dark")
        if not dark:
            continue
        n = dark[0]["n"]
        v1 = dark[0]["basis_votes"].get("[1, 0]", 0)
        v2 = dark[1]["basis_votes"].get("[0, 1]", 0)
        both_null = (dark[0]["beats_null_n"] == n
                     and dark[1]["beats_null_n"] == n)
        ctrl = block["controls"]
        floors = [c for k, c in ctrl.items() if c is not None]
        controls_ok = all(c["pairs_favouring_dark"] == c["n_pairs"]
                          for c in floors) if floors else False
        result[f"E2_{kind}"] = {
            "verdict": "PASS" if (both_null and v1 >= 6 and v2 >= 6
                                  and controls_ok) else "FAIL",
            "plane1_beats_null": f"{dark[0]['beats_null_n']}/{n}",
            "plane2_beats_null": f"{dark[1]['beats_null_n']}/{n}",
            "plane1_basis_votes": dark[0]["basis_votes"],
            "plane2_basis_votes": dark[1]["basis_votes"],
            "plane1_kappa_by_basis": dark[0]["mean_kappa_by_basis"],
            "plane2_kappa_by_basis": dark[1]["mean_kappa_by_basis"],
            "all_controls_complete": controls_ok,
        }
    return result


def main(runs, output):
    runs = Path(runs)
    v4 = analyse(runs / "arm_torus_search_v4.json",
                 runs / "arm_torus_untrained_v4.json")
    v5 = analyse(runs / "arm_torus_search_v5.json",
                 runs / "arm_torus_untrained_v5.json")
    v6 = analyse(runs / "arm_torus_search_v6.json",
                 runs / "arm_torus_untrained_v6.json")
    report = {"v4_parity_arm": v4, "v5_decoupled": v5,
              "v6_decoupled_symmetric_loss": v6,
              "verdicts": verdicts(v4, v5)}
    if "by_model" in v6:
        extra = verdicts(v4, v6)
        report["verdicts"].update(
            {k.replace("E2_", "E2_v6_"): v for k, v in extra.items()
             if k.startswith("E2_")})
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps(report["verdicts"], indent=2))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--output", default="runs/D1_followup_report.json")
    a = ap.parse_args()
    main(a.runs, a.output)
