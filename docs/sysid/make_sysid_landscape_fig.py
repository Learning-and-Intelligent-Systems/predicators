"""Measured replay-SSE landscapes: which recordings identify friction.

Data measured 2026-07-25 on seed1/run_20260724_232411's exact config
(true lateral_friction 0.5): four episodes were executed on the real
env and re-scored through the ACTUAL fit objective path
(``rollout_states`` with a fresh env per evaluation + scaled per-step
SSE) across a lateral_friction grid. The numbers are hardcoded because
regenerating them costs ~10 minutes of PyBullet rollouts; the harness
lives in the session notes (exp_sysid_landscape.py).

Left panel - recordings that identify friction: a slide-rich push
(contact z 0.05: the domino topples then slides 0.18 m) and a plain
green-domino push. Both have their SSE minimum at the true 0.5.
Right panel - recordings that cannot: a pure topple (contact z 0.08)
is flat everywhere EXCEPT deterministic chaos spikes (a replay that
diverges qualitatively at one grid candidate and not its neighbors),
and a pick-place-carry episode is flat everywhere. Regenerate with:

    PYTHONPATH=. python docs/sysid/make_sysid_landscape_fig.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path(__file__).parent / "sysid_landscapes.png"

THETAS = [
    0.15, 0.1817, 0.2202, 0.2667, 0.3063, 0.3232, 0.3236, 0.3915, 0.4743,
    0.4746, 0.5, 0.5747, 0.6267, 0.6962, 0.8435, 1.0219, 1.0358, 1.2381, 1.5
]
SLIDE_PUSH = [
    0.8579, 0.8705, 4.2988, 0.8385, 0.7923, 5.7535, 5.6406, 1.3416, 1.1877,
    0.5777, 0.3554, 28.8911, 267.1505, 288.1091, 121.5445, 19.8388, 11.9516,
    30.0900, 37.7553
]
GREEN_PUSH = [
    1.1984, 0.7908, 0.3673, 0.1764, 0.1090, 0.0763, 0.0681, 0.0642, 0.0883,
    0.0314, 0.0223, 0.0422, 0.0281, 0.0453, 0.0527, 0.0951, 0.0426, 0.0703,
    0.1013
]
PURE_TOPPLE = [
    0.0150, 0.0148, 0.0128, 0.0104, 0.0237, 0.0138, 0.0083, 250.9570, 0.0025,
    248.7303, 0.0000, 342.9730, 0.0040, 237.8221, 0.0636, 0.0078, 0.0080,
    0.0479, 0.0396
]
CARRY_ONLY = [
    0.0004, 0.0004, 0.0004, 0.0002, 0.0001, 0.0010, 0.0011, 0.0002, 0.0009,
    0.0008, 0.0001, 0.0003, 0.0001, 0.0011, 0.0006, 0.0010, 0.0010, 0.0043,
    0.0040
]
FLOOR = 1e-4  # log-plot floor (one SSE evaluated exactly 0.0)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"


def _plot(ax, series, title, legend_loc):
    for vals, color, label, _ in series:
        vals = [max(v, FLOOR) for v in vals]
        ax.plot(THETAS,
                vals,
                color=color,
                linewidth=2,
                marker="o",
                markersize=4,
                zorder=3,
                label=label)
    ax.axvline(0.5, color="#888888", linestyle="--", linewidth=1.2, zorder=1)
    ax.annotate("true 0.5", (0.5, 0.02),
                xycoords=("data", "axes fraction"),
                xytext=(5, 2),
                textcoords="offset points",
                color="#666666",
                fontsize=9)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title(title, fontsize=11)
    ax.set_xlabel("lateral_friction (log)", fontsize=9)
    ax.grid(True, which="major", color="#eeeeee", linewidth=0.8, zorder=0)
    ax.tick_params(labelsize=8)
    ax.legend(loc=legend_loc,
              fontsize=9,
              frameon=True,
              framealpha=0.9,
              edgecolor="#dddddd")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.4), sharey=True)
    _plot(ax1, [
        (SLIDE_PUSH, BLUE, "slide-rich push (z 0.05)", 30.0),
        (GREEN_PUSH, ORANGE, "green push", 0.1),
    ], "Identifying recordings: SSE minimum at the true friction", "upper left")
    _plot(ax2, [
        (PURE_TOPPLE, AQUA, "pure topple (z 0.08)", 0.03),
        (CARRY_ONLY, YELLOW, "pick-place-carry", 0.003),
    ], "Non-identifying: flat, plus deterministic chaos spikes",
          "center right")
    ax1.set_ylabel("replay SSE (fit objective, log)", fontsize=9)
    fig.suptitle(
        "Replay-SSE vs lateral_friction, four recordings at true 0.5 "
        "(fresh-env fit path, measured 2026-07-25)",
        fontsize=11,
        y=1.0)
    fig.tight_layout()
    fig.savefig(OUT, dpi=180, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
