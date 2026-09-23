"""Figure 1: what the criterion asks, and what it answers everywhere we ran it.

Panel A is a schematic and carries no data.  Panel B is drawn from the run
JSON, one point per seed, so the figure cannot disagree with the text.  A
setting whose JSON is absent is skipped and named in the caption data written
alongside, rather than drawn from stale numbers.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROT = "#1b6ca8"      # rotation fit
TRA = "#c8541a"      # translation fit
INK = "#1a1a1a"
MUTE = "#8a8a8a"

# (label, file, joint index, expected family). The expectation is written from
# the environment's kinematics, not from the result: joints that wrap should be
# read as rotations, joints with stops as translations.
SETTINGS = [
    ("Arm, wraps\nConv GRU",   "arm_family_v4.json",      0, "rotation"),
    ("Arm, stops\nConv GRU",   "arm_family_v7.json",      0, "translation"),
    ("Arm, wraps\nRSSM",       "rssm_family_v6.json",     0, "rotation"),
    ("Arm, stops\nRSSM",       "rssm_family_v7.json",     0, "translation"),
    ("Reacher shoulder\nRSSM", "dmc_family_reacher.json", 0, "rotation"),
    ("Reacher wrist\nRSSM",    "dmc_family_reacher.json", 1, None),
]


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


def panel_a(ax):
    ax.set_xlim(0, 10); ax.set_ylim(0.15, 4.3); ax.axis("off")

    def box(x, y, w, h, text, fc="white", ec=INK, fs=7.2):
        ax.add_patch(FancyBboxPatch((x, y), w, h,
                                    boxstyle="round,pad=0.06,rounding_size=0.12",
                                    fc=fc, ec=ec, lw=0.9, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color=INK, zorder=3, linespacing=1.35)

    def arrow(x0, y0, x1, y1, color=INK, style="-|>"):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                                     mutation_scale=8, lw=0.9, color=color,
                                     shrinkA=0, shrinkB=0, zorder=1))

    box(0.15, 1.75, 1.15, 0.9, "state\n$h$", fc="#f2f2f2")

    # Upper path: perturb the action, roll normally.
    box(2.2, 2.85, 2.05, 0.95, "act\n$a_0 + \\delta$", fc="#eef4f9")
    box(5.05, 2.85, 2.0, 0.95, "roll\n(no obs.)", fc="#eef4f9")
    box(7.85, 2.85, 1.95, 0.95, "$\\hat{o}_{1:T}$", fc="#eef4f9")
    arrow(1.3, 2.4, 2.2, 3.3); arrow(4.25, 3.32, 5.05, 3.32)
    arrow(7.05, 3.32, 7.85, 3.32)

    # Lower path: leave the action alone, transform the state.
    box(2.2, 0.6, 2.05, 0.95, "move state\n$R_{g\\delta}(h)$", fc="#fbf0e8")
    box(5.05, 0.6, 2.0, 0.95, "roll, act $a_0$", fc="#fbf0e8")
    box(7.85, 0.6, 1.95, 0.95, "$\\hat{o}'_{1:T}$", fc="#fbf0e8")
    arrow(1.3, 2.0, 2.2, 1.1); arrow(4.25, 1.07, 5.05, 1.07)
    arrow(7.05, 1.07, 7.85, 1.07)

    ax.annotate("", xy=(8.82, 2.78), xytext=(8.82, 1.62),
                arrowprops=dict(arrowstyle="<|-|>", lw=0.9, color=MUTE,
                                mutation_scale=8))
    ax.text(9.0, 2.2, "match?", fontsize=7.2, color=MUTE, va="center")
    ax.text(0.15, 3.95,
            "A   Does moving the state imitate taking the action?",
            fontsize=8.2, color=INK, fontweight="bold")
    ax.text(0.15, 0.06,
            "Discovery uses actions and the model's own predictions. "
            "No pose labels.",
            fontsize=6.8, color=MUTE)


def panel_b(ax, runs: Path):
    xs, drawn, missing, edge_flags = [], [], [], []
    for i, (label, fname, j, expect) in enumerate(SETTINGS):
        data = load(runs / fname)
        if not data or not data.get("rows"):
            missing.append(label.replace("\n", " "))
            continue
        rows = [r for r in data["rows"] if len(r["joints"]) > j]
        rot = np.array([r["joints"][j]["rotation"]["residual"] for r in rows])
        tra = np.array([r["joints"][j]["translation"]["residual"]
                        for r in rows])
        fams = [r["joints"][j]["family"] for r in rows]
        win = max(set(fams), key=fams.count)
        edge_flags.append(sum(
            1 for r in rows if r["joints"][j][win]["gain_at_grid_edge"]))
        x = len(xs)
        xs.append(label)
        rng = np.random.default_rng(11 + i)
        jit = rng.uniform(-0.09, 0.09, len(rows))
        ax.scatter(x - 0.16 + jit, rot, s=13, color=ROT, alpha=0.85,
                   linewidths=0, zorder=3)
        ax.scatter(x + 0.16 + jit, tra, s=13, color=TRA, alpha=0.85,
                   linewidths=0, zorder=3)
        ax.plot([x - 0.3, x - 0.02], [rot.mean()] * 2, color=ROT, lw=1.6,
                zorder=4)
        ax.plot([x + 0.02, x + 0.3], [tra.mean()] * 2, color=TRA, lw=1.6,
                zorder=4)
        drawn.append({"setting": label.replace("\n", " "), "n": len(rows),
                      "winner": win, "n_winner": fams.count(win),
                      "expected": expect,
                      "matches_expectation": None if expect is None
                      else win == expect,
                      "rotation_mean": round(float(rot.mean()), 3),
                      "translation_mean": round(float(tra.mean()), 3),
                      "winner_gain_at_grid_edge": edge_flags[-1]})
        ax.text(x, 1.10, "rot." if win == "rotation" else "trans.",
                ha="center", fontsize=6.6,
                color=ROT if win == "rotation" else TRA)

    ax.axhline(1.0, color=MUTE, lw=0.8, ls=(0, (3, 3)), zorder=1)
    ax.text(len(xs) - 0.45, 0.985, "no transformation helps", fontsize=6.4,
            color=MUTE, ha="right", va="top")
    ax.set_xticks(range(len(xs)))
    ax.set_xticklabels(xs, fontsize=6.9)
    ax.set_ylabel("residual (lower = imitates the action)", fontsize=7.4)
    ax.set_ylim(0, 1.20)
    ax.tick_params(axis="y", labelsize=7)
    ax.set_xlim(-0.6, len(xs) - 0.4)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.scatter([], [], s=13, color=ROT, label="rotation fit")
    ax.scatter([], [], s=13, color=TRA, label="shift fit")
    ax.legend(fontsize=6.9, frameon=False, loc="upper left", ncol=2,
              handletextpad=0.3, columnspacing=1.0,
              bbox_to_anchor=(-0.012, 1.035))
    # The title claims only the axes that are actually plotted. With the
    # architecture and domain runs absent this panel is two GRU settings, and
    # saying "across architecture and domain" over it would be a caption
    # asserting a result that is not in the figure.
    kinds = {d["setting"].split()[-1] for d in drawn}
    spans = []
    if "RSSM" in kinds and "GRU" in kinds:
        spans.append("architecture")
    if any(d["setting"].startswith("Reacher") for d in drawn):
        spans.append("domain")
    tail = (", across " + " and ".join(spans)) if spans else ""
    ax.set_title(f"B   The verdict follows the joint{tail}",
                 fontsize=8.2, fontweight="bold", loc="left", color=INK,
                 pad=6)
    return drawn, missing


def main(runs, output, meta_out):
    runs = Path(runs)
    fig = plt.figure(figsize=(6.6, 4.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[0.88, 1.35], hspace=0.42,
                          left=0.085, right=0.985, top=0.97, bottom=0.16)
    panel_a(fig.add_subplot(gs[0]))
    drawn, missing = panel_b(fig.add_subplot(gs[1]), runs)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=400)
    fig.savefig(str(Path(output).with_suffix(".pdf")))
    Path(meta_out).write_text(json.dumps(
        {"panels": drawn, "settings_without_json": missing}, indent=2))
    for d in drawn:
        print(d["setting"], "->", d["winner"],
              f"{d['n_winner']}/{d['n']}",
              "expected", d["expected"],
              "EDGE" if d["winner_gain_at_grid_edge"] else "")
    if missing:
        print("no JSON yet:", "; ".join(missing))
    print(output)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs")
    ap.add_argument("--output", default="figs/criterion.png")
    ap.add_argument("--meta", default="runs/figure1_meta.json")
    a = ap.parse_args()
    main(a.runs, a.output, a.meta)
