"""Diagram of the physical system-identification pipeline.

One box per pipeline stage, grouped into four lanes (data collection,
fit orchestration, uncertainty accounting, consumers), each annotated
with the module that implements it. Green badges mark the stage-1
honesty fixes that landed on master in PRs #99-#103 (2026-07-30), each
labelled with the run that motivated it; the legend states what each
one replaced. Regenerate with:

    PYTHONPATH=. python docs/sysid/make_sysid_pipeline_fig.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).parent / "sysid_pipeline.png"
# Legend-free variant for slide decks that carry the caption themselves.
OUT_COMPACT = Path(__file__).parent / "sysid_pipeline_compact.png"

LANE_FILL = "#f4f4f4"
BOX_FILL = "#ffffff"
BOX_EDGE = "#555555"
HEADER_COLORS = ["#3d6b99", "#3d8a5f", "#8a6d3d", "#7a4f8a"]
FLAG_COLOR = "#1a7a1a"

LANES = [
    (12.0, "DATA (real env, true params)"),
    (37.0, "FIT  (code_sim_learning)"),
    (62.0, "UNCERTAINTY"),
    (87.0, "CONSUMERS"),
]
LANE_W = 23.0
# Vertical band the stacked stage boxes may occupy, and the largest gap
# to leave between two boxes (a sparse lane centers instead of spreading).
LANE_BOTTOM = 8.0
LANE_TOP = 52.0
MAX_GAP = 3.2

# (id, lane, height, title, body, flag), listed top to bottom per lane;
# y centers are computed in _layout so boxes cannot drift into each other.
BOXES = [
    ("A1", 0, 7.5, "Explore episodes", "agent_bilevel explorer, 2/cycle\n"
     "-> LowLevelTrajectory\n"
     "12 Hz pose setpoints, no velocities", None),
    ("A2", 0, 10.0, "Declaration evidence", "sim.residuals(rollout=True):\n"
     "open-loop replay + per-param\n"
     "box sweep; a flat sweep is the\n"
     "honest reason to omit a param", "5  declaring is a decision"),
    ("A3", 0, 8.5, "Learn session artifact", "sandbox/simulator.py:\n"
     "PHYSICAL_PARAMS (ParamSpec,\n"
     "log scale) + RESIDUAL_FEATURES\n"
     "+ rules and rule params", None),
    ("B1", 1, 7.0, "Trajectory prep", "settled-tail truncation\n"
     "-> rest-point segmentation\n"
     "-> residual scaling  [trajectory_prep]", None),
    ("B2", 1, 8.5, "Rollout objective", "free-run replay per theta,\n"
     "fresh env per rollout  [rollout_env]\n"
     "Huber-capped SSE + summary\n"
     "terms  [rollout_objective]", "1  robust to chaos spikes"),
    ("B3", 1, 8.5, "Search", "explainability trim -> per-param\n"
     "grid + flat-edge refine  [grid_seed]\n"
     "-> joint LM MAP, log space,\n"
     "Gaussian prior  [lm]", None),
    ("B4", 1, 4.2, "Anchor ablation", "revert compensatory moves", None),
    ("B5", 1, 7.5, "Consistency loop", "per-segment refits; on disagreement\n"
     "DROP least trustworthy + refit;\n"
     "dropped fits -> hull candidates", "2  the doubt is kept"),
    ("C1", 2, 8.0, "Laplace posterior", "posterior_std per param,\n"
     "floored at 0.1 (log space)", "3  floor is a lower bound"),
    ("C2", 2, 10.0, "Identifiability report", "posterior/prior contraction:\n"
     "identified / weakly / NOT /\n"
     "INCONSISTENT across cycles,\n"
     "arbitrated on pooled SSE", "4  INCONSISTENT is swept"),
    ("C3", 2, 5.5, "Trustworthy selection", "select_trustworthy_params\n"
     "-> applied dict (physical only)", None),
    ("C4", 2, 8.0, "Margin grid", "physics_sigma_points: 32 points\n"
     "over the DISAGREEMENT HULL\n"
     "(sigma band + all candidates)\n"
     "-> ctx.physics_margin_provider", None),
    ("D1", 3, 7.0, "Belief env", "applied to base env for planning;\n"
     "fresh validation envs re-apply;\n"
     "dropped params revert to registry", None),
    ("D2", 3, 10.0, "Capture gate", "evaluate_option_plan: parse ->\n"
     "legitimacy -> 3x/6x decorrelated\n"
     "validation -> 32-pt hull sweep\n"
     "-> PARAM-SENSITIVE on failure", None),
    ("D3", 3, 6.5, "Certificate probe", "legitimacy replays run on base sim\n"
     "+ learned rules  [probe factory]", "6  the agent's own substrate"),
    ("D4", 3, 5.5, "Agent pre-check", "sim.run(plan, physics_sweep=True)\n"
     "same grid, one rollout per point", None),
    ("D5", 3, 5.5, "Exploration ensemble", "6 Laplace members -> explorer\n"
     "info-seeking disagreement", None),
]

ARROWS = [
    ("A1", "B1"),
    ("A1", "A2"),
    ("A2", "A3"),
    ("A3", "B2"),
    ("B1", "B2"),
    ("B2", "B3"),
    ("B3", "B4"),
    ("B4", "B5"),
    ("B5", "C1"),
    ("C1", "C2"),
    ("C2", "C3"),
    ("C3", "C4"),
    ("C3", "D1"),
    ("C4", "D2"),
    ("D2", "D3"),
    ("C4", "D4"),
]

LEGEND = ("Green badges: honesty fixes landed on master 2026-07-30 (PRs "
          "#99-#103), each replacing a measured failure mode.\n"
          "[1] Huber cap + summary-statistic residuals (settled poses, motion "
          "onset), replacing bare per-step SSE that one chaos spike could "
          "steer;  [2] a dropped segment's own-best fits become hull "
          "candidates - what leaves the mean reappears in the variance;\n"
          "[3] the floored posterior_std is a lower bound only: the swept "
          "interval is the disagreement HULL (sigma band widened to every "
          "candidate fit);  [4] INCONSISTENT is swept and held on the last "
          "TRUSTED value (it used to disarm the gate), with flagged jumps "
          "arbitrated on pooled SSE;\n"
          "[5] declaring PHYSICAL_PARAMS is an explicit decision backed by an "
          "open-loop sweep, not a silent omission;  [6] certificate replays "
          "judge plans on the combined substrate the agent plans on.\n"
          "Drivers: run_20260724_232411 (fits 1.0358 / 0.3236 -> 0.6267 vs "
          "true 0.5, both 0/1), run_20260724_140531 (failure hole AT truth), "
          "run_20260727_210827 (sticky biased fit), al_margin seeds 1-2 "
          "(sysID skipped, gate stuck at the prior).")


def _lane_x(lane):
    return LANES[lane][0]


def _layout():
    """Y center per box: equal gaps, each lane's stack vertically centered."""
    ys = {}
    for lane in range(len(LANES)):
        boxes = [b for b in BOXES if b[1] == lane]
        total = sum(b[2] for b in boxes)
        gaps = max(len(boxes) - 1, 1)
        gap = min(MAX_GAP, (LANE_TOP - LANE_BOTTOM - total) / gaps)
        span = total + gap * (len(boxes) - 1)
        y = LANE_TOP - (LANE_TOP - LANE_BOTTOM - span) / 2.0
        for bid, _, h, *_rest in boxes:
            ys[bid] = y - h / 2.0
            y -= h + gap
    return ys


def draw(out=OUT, include_legend=True):
    fig, ax = plt.subplots(figsize=(21, 11.5 if include_legend else 10.0))
    ax.set_xlim(0, 100)
    ax.set_ylim(0 if include_legend else 5.5, 60)
    ax.axis("off")

    for li, (cx, title) in enumerate(LANES):
        ax.add_patch(
            FancyBboxPatch((cx - LANE_W / 2, 6.0),
                           LANE_W,
                           49.5,
                           boxstyle="round,pad=0.4",
                           facecolor=LANE_FILL,
                           edgecolor="none",
                           zorder=0))
        ax.add_patch(
            FancyBboxPatch((cx - LANE_W / 2, 53.5),
                           LANE_W,
                           3.4,
                           boxstyle="round,pad=0.3",
                           facecolor=HEADER_COLORS[li],
                           edgecolor="none",
                           zorder=2))
        ax.text(cx,
                55.2,
                title,
                ha="center",
                va="center",
                fontsize=13,
                color="white",
                fontweight="bold",
                zorder=3)

    centers = {}
    y_centers = _layout()
    for bid, lane, h, title, body, flag in BOXES:
        cx = _lane_x(lane)
        yc = y_centers[bid]
        w = LANE_W - 2.0
        ax.add_patch(
            FancyBboxPatch((cx - w / 2, yc - h / 2),
                           w,
                           h,
                           boxstyle="round,pad=0.25",
                           facecolor=BOX_FILL,
                           edgecolor=BOX_EDGE,
                           linewidth=1.1,
                           zorder=2))
        ax.text(cx,
                yc + h / 2 - 1.1,
                title,
                ha="center",
                va="center",
                fontsize=10.5,
                fontweight="bold",
                zorder=3)
        ax.text(cx,
                yc + h / 2 - 2.2,
                body,
                ha="center",
                va="top",
                fontsize=8.6,
                zorder=3,
                linespacing=1.35)
        if flag:
            ax.text(cx - w / 2 + 0.6,
                    yc - h / 2 + 0.5,
                    flag,
                    ha="left",
                    va="bottom",
                    fontsize=8.6,
                    color=FLAG_COLOR,
                    fontweight="bold",
                    zorder=4)
        centers[bid] = (cx, yc, w, h)

    for src, dst in ARROWS:
        sx, sy, sw, sh = centers[src]
        dx, dy, dw, dh = centers[dst]
        if abs(sx - dx) < 1.0:
            start, end = (sx, sy - sh / 2 - 0.3), (dx, dy + dh / 2 + 0.3)
            style = "arc3,rad=0.0"
        else:
            start, end = (sx + sw / 2 + 0.3, sy), (dx - dw / 2 - 0.3, dy)
            style = "arc3,rad=-0.08"
        ax.add_patch(
            FancyArrowPatch(start,
                            end,
                            arrowstyle="-|>",
                            mutation_scale=13,
                            linewidth=1.3,
                            color="#666666",
                            connectionstyle=style,
                            zorder=1))

    if include_legend:
        ax.text(2.0,
                4.3,
                LEGEND,
                ha="left",
                va="top",
                fontsize=9.0,
                linespacing=1.5)
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    draw()
    draw(OUT_COMPACT, include_legend=False)
