"""Replay each task's recorded sketches with ik_validate=True and compare to
the recorded (ik_validate=False) run outcome — to separate pure IK-artifact
failures from genuine geometric ones, and to check for regressions on solved
tasks.

Mirrors the pipeline solve loop: for each task, try recorded sketches in
order, up to N refine attempts each, stop at first success; enforce a
per-task wall budget. Run ONE seed+arm per process. Usage: PYTHONPATH=.
python scripts/domino_debug/replay_ikval_sweep.py <seed> <demo|no_demo>
[budget_s]
"""
import logging
import re
import sys
import time
from glob import glob
from typing import Any, Callable, Dict, List, Tuple

logging.disable(logging.CRITICAL)

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(
    r"^\s*\d+:\s*([A-Za-z]\w*)\((.*?)\)(?:\s*->\s*\{(.*)\})?\s*$")
SKH = re.compile(r"Sketch \(attempt (\d+)\)")
TRES = re.compile(
    r"\[main\.py\] Task (\d+) / \d+: (.*)|Task (\d+) / \d+: (SOLVED)")


def extract(info_log: str) -> Dict[int, dict]:
    """Parse recorded sketches and outcomes per task from an info.log."""
    tasks: Dict[int, dict] = {}
    pending: List[List[Tuple[str, List[str], str]]] = []
    cur: List[Tuple[str, List[str], str]] | None = None
    for raw in open(info_log, encoding="utf-8"):
        line = ANSI.sub("", raw.rstrip("\n"))
        if SKH.search(line):
            cur = []
            pending.append(cur)
            continue
        m = STEP.match(line)
        if m and cur is not None:
            cur.append((m.group(1), [
                a.split(":")[0].strip() for a in m.group(2).split(",")
                if a.strip()
            ], m.group(3) or ""))
            continue
        cur = None
        tm = TRES.search(line)
        if tm:
            ti = int(tm.group(1) or tm.group(3)) - 1
            tasks[ti] = {
                "outcome": (tm.group(2) or tm.group(4) or "").strip(),
                "sketches": pending
            }
            pending = []
    return tasks


def main() -> None:
    """Replay recorded sketches with ik_validate and report flips."""
    seed = int(sys.argv[1])
    arm = sys.argv[2]
    budget = float(sys.argv[3]) if len(sys.argv) > 3 else 300.0
    ikv = (sys.argv[4].lower() == "true") if len(sys.argv) > 4 else True
    exp = f"domino-agent_oracle_hybrid_sim_oracle_samplers_{arm}"
    info_log = sorted(
        glob(f"logs/agent_sim_learning/{exp}/seed{seed}/run_*/info.log"))[-1]
    rec = extract(info_log)
    FLAGS = {
        "env": "pybullet_domino",
        "approach": "agent_sim_learning",
        "seed": seed,
        "num_train_tasks": 1,
        "num_test_tasks": 5,
        "skill_phase_use_motion_planning": True,
        "pybullet_ik_validate": ikv,  # <-- the change under test
        "demonstrator": "oracle_process_planning",
        "bilevel_plan_without_sim": True,
        "explorer": "agent_bilevel",
        "agent_sim_learn_oracle_sim_program": True,
        "agent_sim_learn_oracle_sim_params": True,
        "agent_sim_learn_parameterized_samplers": True,
        "agent_sim_learn_oracle_samplers": True,
        "execution_monitor": "subgoal_annotations",
        "agent_bilevel_max_execution_replans": 2,
        "horizon": 400,
        "excluded_objects_in_state_str": "loc,rot,angle,direction",
        "excluded_predicates": "InitialBlock,MovableBlock,Tilting,Upright",
        "domino_initialize_at_finished_state": False,
        "domino_use_domino_blocks_as_target": True,
        "domino_use_continuous_place": True,
        "domino_restricted_push": True,
        "domino_has_glued_dominos": False,
        "pybullet_birrt_extend_num_interp": 20,
        "pybullet_birrt_path_subsample_ratio": 2,
        "agent_sdk_use_local_sandbox": True,
        "option_model_terminate_on_repeat": False,
        "agent_planner_use_simulator": True
    }
    # pylint: disable=import-outside-toplevel
    # Imports are deferred until after reset_config so module-level CFG
    # reads in these modules observe the FLAGS set above.
    from predicators import utils
    utils.reset_config(FLAGS)
    from predicators.agent_sdk import bilevel_sketch
    from predicators.approaches import create_approach
    from predicators.envs import get_or_create_env
    from predicators.ground_truth_models import get_gt_options
    env = get_or_create_env("pybullet_domino")
    options = get_gt_options(env.get_name())
    preds, _ = utils.parse_config_excluded_predicates(env)
    # Cast to Any: this script probes approach-specific protected members
    # (sampler installers, option model) absent from the BaseApproach API.
    ap: Any = create_approach("agent_sim_learning", preds, options, env.types,
                              env.action_space,
                              [t.task for t in env.get_train_tasks()])
    ap._maybe_install_oracle_samplers()  # pylint: disable=protected-access
    n2t = {o.name: o.type.name for o in env.get_test_tasks()[0].task.init}

    def typed(steps: List[Tuple[str, List[str], str]]) -> str:
        """Render parsed sketch steps as typed operator lines."""
        lines = []
        for op, objs, sg in steps:
            args = ", ".join(o + ":" + n2t.get(o, "object") for o in objs)
            line = op + "(" + args + ")"
            if sg:
                line += " -> {" + sg + "}"
            lines.append(line)
        return "\n".join(lines)

    header = (f"# seed{seed} {arm} | ik_validate={ikv} | "
              f"NEW task-gen | budget={budget}s")
    print(header)
    for ti in sorted(rec):
        task = env.get_test_tasks()[ti].task
        recout = "SOLVED" if rec[ti]["outcome"].upper().startswith(
            "SOLVED") else "FAILED"
        t0 = time.perf_counter()
        solved_by = None
        deepest: Tuple[int, str] = (-1, "")
        for si, steps in enumerate(rec[ti]["sketches"]):
            if time.perf_counter() - t0 > budget:
                break
            sk = bilevel_sketch.parse_sketch_from_text(typed(steps),
                                                       task,
                                                       predicates=preds,
                                                       options=set(options),
                                                       types=env.types)
            if not sk:
                continue
            for r in range(2):
                if time.perf_counter() - t0 > budget:
                    break
                fail: Dict[str, object] = {"idx": -1, "reason": ""}

                def make_rc(
                    f: Dict[str,
                            object]) -> Callable[[int, object, str], None]:
                    """Build an on_step_fail recording the deepest fail."""

                    def rc(i: int, _p: object, reason: str) -> None:
                        if i > f["idx"]:  # type: ignore[operator]
                            f["idx"], f["reason"] = i, reason

                    return rc

                # attempt reproduces this script's historical RNG streams
                # (rng = CFG.seed + attempt inside _refine_sketch).
                # pylint: disable-next=protected-access
                _, ok = ap._refine_sketch(task,
                                          sk,
                                          timeout=budget,
                                          attempt=si * 5 + r,
                                          on_step_fail=make_rc(fail))
                if ok:
                    solved_by = (si, r)
                    break
                if fail["idx"] > deepest[0]:  # type: ignore[operator]
                    deepest = (fail["idx"], fail["reason"])  # type: ignore
            if solved_by:
                break
        dt = time.perf_counter() - t0
        verdict = "SOLVED" if solved_by else "FAILED"
        flip = "" if verdict == recout else (
            "  *** REGRESSION" if recout == "SOLVED" else "  *** FIXED")
        if solved_by:
            extra = f"by sketch{solved_by[0]}"
        else:
            extra = f"deepest step{deepest[0]}: {deepest[1][:32]}"
        line = (f"  task{ti+1}: recorded(F)={recout:6s} -> "
                f"new={verdict:6s} [{dt:5.0f}s] {extra}{flip}")
        print(line)


if __name__ == "__main__":
    main()
