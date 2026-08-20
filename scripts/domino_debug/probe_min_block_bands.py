"""Anchor probes for calibrating min-block task bands (spans / turn legs).

The min-block differentiation bands are friction-pair specific: straight
spans need a window where the true friction's chain count is below the
planning friction's, and turn legs need cells where a natural corner
tops at the true friction while the believed side needs an extra blue.
This script measures both at the canonical probe anchor with the SAME
machinery task generation uses (memoized straight probes, the labeled
turn-layout family search, real Push rollouts), so its numbers transfer
to the generator's certificates.

Used for the 2026-07-12 domino_high_friction short-leg retune (see the
env block comments in scripts/configs/predicatorv3/envs/all.yaml). Run
it whenever domino_true_friction / domino_planning_friction change:

    python scripts/domino_debug/probe_min_block_bands.py reach \
        --frictions 0.1 0.5 --span-lo 0.12 --span-hi 0.64
    python scripts/domino_debug/probe_min_block_bands.py turn \
        --frictions 0.1 0.5 --cells 0.22,0.18 0.23,0.19 --reps 3

Reading the output:
  * reach: pick a span window where k(true) < k(planning) is stable
    across --reps (repeats clear the probe memo, sampling solver-history
    variance; a span whose count flips between rounds is knife-edge).
  * turn: per (entry, exit) cell and friction, the first k with a
    toppling layout plus per-family topple counts. "corner" (single
    natural-yaw corner blue) is the agent-buildable style - a cell whose
    only topplers are "pair" (the legacy 45-degree pair) is
    agent-intractable and must NOT ship (that was the pre-retune
    high_friction failure). On the planning-friction side, prefer cells
    whose k ties are impossible: believed k should exceed the true k in
    EVERY rep (single-rep believed reads flicker ~1/3 on knife-edge
    cells, which is why the generator re-runs its believed certificates
    twice post-staging).
"""
import argparse
import json
import time
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from predicators import utils


def _build_env() -> Any:
    """Env with the min-block flags that affect probe physics."""
    utils.reset_config({
        "env": "pybullet_domino",
        "approach": "oracle",
        "seed": 0,
        "use_gui": False,
        "domino_initialize_at_finished_state": False,
        "domino_use_domino_blocks_as_target": True,
        "domino_use_continuous_place": True,
        "domino_has_glued_dominos": False,
        "domino_use_skill_factories": True,
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": False,
        "pybullet_birrt_extend_num_interp": 20,
        "pybullet_birrt_path_subsample_ratio": 2,
        "domino_min_block_tasks": True,
        "horizon": 500,
    })
    # pylint: disable-next=import-outside-toplevel
    from predicators.envs.pybullet_domino.env import PyBulletDominoEnv
    return PyBulletDominoEnv(use_gui=False)


def _turn_poses(mbu: Any, entry: float, exit_: float) -> Tuple[Any, Any]:
    """Left-turn L at the canonical anchor, entry along +x (like the straight
    probes)."""
    sx, sy = mbu._PROBE_ANCHOR  # pylint: disable=protected-access
    syaw = np.pi / 2
    u = np.array([np.sin(syaw), np.cos(syaw)])
    p = np.array([-u[1], u[0]])
    t = np.array([sx, sy]) + entry * u + exit_ * p
    tyaw = float(np.arctan2(-p[0], p[1]))
    return (sx, sy, syaw), (float(t[0]), float(t[1]), tyaw)


def probe_reach(env: Any, mbu: Any, frictions: Sequence[float],
                spans: Sequence[float], budget: int,
                reps: int) -> Dict[str, List[Any]]:
    """k = straight_span_k_star per (friction, span), ``reps`` rounds with
    the memo cleared between rounds."""
    results: Dict[str, List[Any]] = {}
    for rep in range(reps):
        mbu._span_probe_memo.clear()  # pylint: disable=protected-access
        for f in frictions:
            env.set_domino_physical_params(lateral_friction=f)
            for span in spans:
                k = mbu.straight_span_k_star(env, span, budget=budget)
                results.setdefault(f"{f}|{span:.2f}", []).append(k)
                print(f"reach rep{rep} f={f} span={span:.2f} -> k={k}",
                      flush=True)
    return results


def probe_turn(env: Any, mbu: Any, frictions: Sequence[float],
               cells: Sequence[Tuple[float, ...]], budget: int,
               reps: int) -> Dict[str, List[Any]]:
    """First k with a toppler per (friction, cell), with per-family topple
    counts from full k-layer scans, ``reps`` times each."""
    comp = env._domino_component  # pylint: disable=protected-access
    results: Dict[str, List[Any]] = {}
    for rep in range(reps):
        for f in frictions:
            env.set_domino_physical_params(lateral_friction=f)
            for entry, exit_ in cells:
                sp, tp = _turn_poses(mbu, entry, exit_)
                push_opt = mbu._get_push_option(env)  # pylint: disable=protected-access
                t0 = time.time()
                first_k, layers = None, []
                for k in range(budget + 1):
                    fams: Dict[str, int] = {}
                    # pylint: disable-next=protected-access
                    for fam, od, s, t in mbu._candidate_turn_layouts_labeled(
                            comp, k, sp, tp):
                        if mbu._layout_topples(env, od, s, t, push_opt):  # pylint: disable=protected-access
                            fams[fam] = fams.get(fam, 0) + 1
                    layers.append({"k": k, "topples": fams})
                    if fams:
                        first_k = k
                        break  # the K* layer is fully scanned; stop
                results.setdefault(f"{f}|{entry}|{exit_}", []).append({
                    "k":
                    first_k,
                    "layers":
                    layers,
                })
                print(
                    f"turn rep{rep} f={f} legs=({entry},{exit_}) -> "
                    f"k={first_k} ({time.time() - t0:.1f}s) "
                    f"families={layers[-1]['topples']}",
                    flush=True)
    return results


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--frictions", type=float, nargs="+", required=True)
    common.add_argument("--budget", type=int, default=5)
    common.add_argument("--reps", type=int, default=1)
    common.add_argument("--out", type=str, default="")
    reach = sub.add_parser("reach", parents=[common])
    reach.add_argument("--span-lo", type=float, default=0.12)
    reach.add_argument("--span-hi", type=float, default=0.64)
    reach.add_argument("--span-step", type=float, default=0.04)
    turn = sub.add_parser("turn", parents=[common])
    turn.add_argument("--cells",
                      type=str,
                      nargs="+",
                      required=True,
                      help="entry,exit leg pairs, e.g. 0.22,0.18")
    args = parser.parse_args()

    env = _build_env()
    # pylint: disable-next=import-outside-toplevel
    from predicators.envs.pybullet_domino.task_generators import \
        min_block_utils as mbu
    if args.mode == "reach":
        n = int(round((args.span_hi - args.span_lo) / args.span_step)) + 1
        spans = [round(args.span_lo + i * args.span_step, 2) for i in range(n)]
        results = probe_reach(env, mbu, args.frictions, spans, args.budget,
                              args.reps)
    else:
        cells = [tuple(float(v) for v in c.split(",")) for c in args.cells]
        results = probe_turn(env, mbu, args.frictions, cells, args.budget,
                             args.reps)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=1)
        print(f"saved {args.out}")


if __name__ == "__main__":
    _main()
