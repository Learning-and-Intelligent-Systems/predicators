"""Measure InFront settle-drift: place a domino at the oracle sampler's
generator-faithful nominal pose through the REAL option model (PyBullet forward
sim), then compare the settled pose to the nominal one and to the InFront
tolerance window.

Usage: PYTHONPATH=. python \
    scripts/domino_debug/probe_infront_drift.py <seed> <env_task_idx>
"""
import logging
import sys

import numpy as np

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options
from predicators.ground_truth_models.domino import processes as P

logging.disable(logging.CRITICAL)

# Same flags the replay uses so the option model + oracle samplers match.
# pylint: disable=wrong-import-position
from scripts.domino_debug.replay_domino_sketches import _FLAGS  # noqa: E402


def main() -> None:
    """Probe InFront settle-drift for one seed/test-task index."""
    seed = int(sys.argv[1])
    ti = int(sys.argv[2])

    utils.reset_config(dict(_FLAGS, seed=seed))

    env = get_or_create_env("pybullet_domino")
    options = get_gt_options(env.get_name())
    preds, _ = utils.parse_config_excluded_predicates(env)
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = create_approach("agent_sim_learning", preds, options, env.types,
                               env.action_space, train_tasks)
    # pylint: disable=protected-access
    approach._maybe_install_oracle_samplers()  # type: ignore[attr-defined]
    om = approach._option_model  # type: ignore[attr-defined]
    assert om is not None, "no option model"

    task = env.get_test_tasks()[ti].task
    state = task.init
    dominoes = sorted([o for o in state if o.type.name == "domino"],
                      key=lambda o: o.name)
    robot = [o for o in state if o.type.name == "robot"][0]
    print(f"# seed{seed} env-task{ti}: {len(dominoes)} dominoes")
    for d in dominoes:
        print(f"  {d.name}: x={state.get(d,'x'):.4f} y={state.get(d,'y'):.4f} "
              f"yaw={state.get(d,'yaw'):+.4f} roll={state.get(d,'roll'):+.4f}")

    InFront = [p for p in preds if p.name == "InFront"][0]
    infront_holds = InFront.holds
    pos_gap = 0.098  # PyBulletDominoEnv.pos_gap
    pos_tol = pos_gap * 0.3
    print(f"\n# InFront window: pos_tol={pos_tol:.4f} m, ang_tol=15deg, "
          f"pos_gap={pos_gap:.4f}")

    opt_by_name = {o.name: o for o in options}
    Pick, Place = opt_by_name["Pick"], opt_by_name["Place"]

    # Place domino_1 to satisfy InFront(domino_1, domino_0).
    held, ref = dominoes[1], dominoes[0]
    sub = {utils.GroundAtom(InFront, [held, ref])}

    # 1) Pick the held domino.
    # pylint: disable=protected-access
    pick_params = P._pick_option_sampler(state, set(),
                                         np.random.default_rng(0),
                                         [robot, held])
    pick_opt = Pick.ground([robot, held], pick_params)
    assert pick_opt.initiable(state), "pick not initiable"
    s1, _ = om.get_next_state_and_num_actions(state, pick_opt)
    print(f"\n# after Pick({held.name}): is_held={s1.get(held,'is_held'):.2f}")

    # 2) Sample the generator-faithful placement and run Place.
    for trial in range(5):
        rng = np.random.default_rng(100 + trial)
        place_params = P._place_option_sampler(s1, sub, rng, [robot])
        nom_x, nom_y, _, nom_yaw = [float(v) for v in place_params]
        place_opt = Place.ground([robot], place_params)
        if not place_opt.initiable(s1):
            print(f"  trial{trial}: place not initiable")
            continue
        s2, _ = om.get_next_state_and_num_actions(s1, place_opt)
        gx, gy, gyaw = (s2.get(held, "x"), s2.get(held,
                                                  "y"), s2.get(held, "yaw"))
        groll = s2.get(held, "roll")
        rx, ry, ryaw = (s2.get(ref, "x"), s2.get(ref, "y"), s2.get(ref, "yaw"))
        infront = infront_holds(s2, [held, ref])
        # nominal InFront check (kinematic, roll=0, exactly at sampler pose)
        snom = s2.copy()
        snom.set(held, "x", nom_x)
        snom.set(held, "y", nom_y)
        snom.set(held, "yaw", nom_yaw)
        snom.set(held, "roll", 0.0)
        nom_infront = infront_holds(snom, [held, ref])
        print(f"\n  trial{trial}: nominal place=({nom_x:.4f},{nom_y:.4f},"
              f"yaw={nom_yaw:+.4f})  nominal_InFront={nom_infront}")
        print(f"    settled {held.name}=({gx:.4f},{gy:.4f},yaw={gyaw:+.4f},"
              f"roll={groll:+.4f})")
        print(f"    drift dx={gx-nom_x:+.4f} dy={gy-nom_y:+.4f} "
              f"dyaw={gyaw-nom_yaw:+.4f}  (pos_tol={pos_tol:.4f})")
        tol10 = np.sin(np.radians(10))
        is_cardinal = (abs(np.sin(ryaw)) < tol10 or abs(np.cos(ryaw)) < tol10)
        cardinal = "yes" if is_cardinal else "NO"
        print(f"    {ref.name}=({rx:.4f},{ry:.4f},yaw={ryaw:+.4f}) "
              f"cardinal={cardinal}")
        print(f"    => settled InFront({held.name},{ref.name}) = {infront}")


if __name__ == "__main__":
    main()
