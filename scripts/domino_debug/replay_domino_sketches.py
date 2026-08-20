"""Faithfully reproduce domino refinement failures by replaying the recorded
LLM sketches through the real bilevel refinement -- no LLM required.

The agent's plan sketches were logged verbatim in each run's ``info.log``
(``Sketch (attempt N):`` blocks). This script extracts them, regenerates the
deterministic test task, and runs the *exact same* ``refine_sketch`` the
pipeline uses (oracle option model + oracle samplers + subgoal checks, same
per-(sketch,refine) RNG seeding). The pass/fail outcome and the "stuck at step
K" reason therefore reproduce the run's solve-time failures deterministically.

Run ONE seed per process (task-gen RNG is shared; see
reproduce_domino_failures).

Usage:
    PYTHONPATH=. python scripts/domino_debug/replay_domino_sketches.py \
        <seed> <demo|no_demo> [--all]
        --all replays every task; default replays only tasks the run
        did not solve.
"""

import logging
import re
import sys
from glob import glob
from typing import Dict, List, Optional, Tuple

from predicators import utils
from predicators.agent_sdk import bilevel_sketch
from predicators.approaches import create_approach
from predicators.approaches.agent_sim_learning_approach import \
    AgentSimLearningApproach
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_options

logging.disable(logging.CRITICAL)

# Refine retries per sketch, matching the refine loop the audited runs
# ran with (their agent_bilevel_max_refine_retries setting); preserved
# here so old recordings keep replaying faithfully.
_REFINE_RETRIES = 5

ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(
    r"^\s*\d+:\s*([A-Za-z]\w*)\((.*?)\)(?:\s*->\s*\{(.*)\})?\s*$")
SKETCH_HDR = re.compile(r"Sketch \(attempt (\d+)\)")
TASK_RES = re.compile(
    r"\[main\.py\] Task (\d+) / \d+: (.*)|Task (\d+) / \d+: (SOLVED)")

_FLAGS = {
    "env": "pybullet_domino",
    "approach": "agent_sim_learning",
    "num_train_tasks": 1,
    "num_test_tasks": 5,
    "skill_phase_use_motion_planning": True,
    "pybullet_ik_validate": False,
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
    "process_planning_heuristic_weight": 2.0,
    "domino_has_glued_dominos": False,
    "pybullet_birrt_extend_num_interp": 20,
    "pybullet_birrt_path_subsample_ratio": 2,
    "agent_sdk_use_local_sandbox": True,
    "option_model_terminate_on_repeat": False,
    "agent_planner_use_simulator": True,
}


def find_info_log(seed: int, arm: str) -> str:
    """Return the newest matching run's info.log path for seed/arm."""
    exp = f"domino-agent_oracle_hybrid_sim_oracle_samplers_{arm}"
    pat = f"logs/agent_sim_learning/{exp}/seed{seed}/run_*/info.log"
    hits = sorted(glob(pat))
    if not hits:
        raise SystemExit(f"no info.log at {pat}")
    return hits[-1]


def extract_sketches(info_log: str) -> Dict[int, dict]:
    """Return {task_idx (0-based): {"outcome": str, "sketches": ...}}.

    Each step is (option_name, [obj_names], raw_subgoal_str).
    """
    tasks: Dict[int, dict] = {}
    pending: List[List[Tuple[str, List[str], str]]] = []
    cur: Optional[List[Tuple[str, List[str], str]]] = None
    with open(info_log, encoding="utf-8") as f:
        for raw in f:
            line = ANSI.sub("", raw.rstrip("\n"))
            if SKETCH_HDR.search(line):
                cur = []
                pending.append(cur)
                continue
            m = STEP.match(line)
            if m and cur is not None:
                opt, args, sg = m.group(1), m.group(2), m.group(3) or ""
                objs = [
                    a.split(":")[0].strip() for a in args.split(",")
                    if a.strip()
                ]
                cur.append((opt, objs, sg))
                continue
            cur = None  # any non-step line ends the current sketch block
            tm = TASK_RES.search(line)
            if tm:
                ti = int(tm.group(1) or tm.group(3)) - 1
                outcome = (tm.group(2) or tm.group(4) or "").strip()
                tasks[ti] = {"outcome": outcome, "sketches": pending}
                pending = []
    return tasks


def typed_text(steps: List[Tuple[str, List[str], str]],
               name_to_type: Dict[str, str]) -> str:
    """Rebuild typed sketch text the option-plan parser expects."""
    lines = []
    for opt, objs, sg in steps:
        typed = ", ".join(f"{o}:{name_to_type.get(o, 'object')}" for o in objs)
        line = f"{opt}({typed})"
        if sg:
            line += f" -> {{{sg}}}"
        lines.append(line)
    return "\n".join(lines)


class _DeepestFail:
    """Track the deepest (highest-index) step failure seen so far."""

    def __init__(self) -> None:
        self.idx: int = -1
        self.reason: str = ""

    def record(self, idx: int, _prefix: list, reason: str) -> None:
        """on_step_fail callback: keep the deepest failure."""
        if idx > self.idx:
            self.idx, self.reason = idx, reason


def main() -> None:
    """Replay recorded sketches through the real refinement."""
    seed = int(sys.argv[1])
    arm = sys.argv[2] if len(sys.argv) > 2 else "no_demo"
    replay_all = "--all" in sys.argv

    info_log = find_info_log(seed, arm)
    tasks = extract_sketches(info_log)

    utils.reset_config(dict(_FLAGS, seed=seed))

    env = get_or_create_env("pybullet_domino")
    options = get_gt_options(env.get_name())
    preds, _ = utils.parse_config_excluded_predicates(env)
    train_tasks = [t.task for t in env.get_train_tasks()]
    approach = create_approach("agent_sim_learning", preds, options, env.types,
                               env.action_space, train_tasks)
    assert isinstance(approach, AgentSimLearningApproach)
    # pylint: disable=protected-access
    approach._maybe_install_oracle_samplers()
    # pylint: enable=protected-access
    test_tasks = env.get_test_tasks()
    name_to_type = {o.name: o.type.name for o in test_tasks[0].task.init}

    print(f"# seed{seed} {arm}: replaying recorded sketches through real "
          f"refinement (oracle option-model + oracle samplers, no LLM)")
    for ti in sorted(tasks):
        rec = tasks[ti]
        solved = rec["outcome"].upper().startswith("SOLVED")
        if solved and not replay_all:
            continue
        task = test_tasks[ti].task
        print(f"\n== task{ti} (run Task{ti+1}) | run outcome: "
              f"{rec['outcome'][:60]}")
        if not rec["sketches"]:
            print("   (no sketches recorded)")
            continue
        for si, steps in enumerate(rec["sketches"]):
            sketch = bilevel_sketch.parse_sketch_from_text(
                typed_text(steps, name_to_type),
                task,
                predicates=preds,
                options=set(options),
                types=env.types)
            if not sketch:
                print(f"   sketch{si}: unparseable")
                continue
            any_success = False
            deepest_idx, deepest_reason = -1, ""
            for r in range(_REFINE_RETRIES):
                fail = _DeepestFail()
                # attempt reproduces the audited runs' per-(sketch,refine)
                # RNG seeding (rng = CFG.seed + attempt inside
                # _refine_sketch), so recorded failures replay exactly.
                # pylint: disable-next=protected-access
                _, success = approach._refine_sketch(
                    task,
                    sketch,
                    timeout=600.0,
                    attempt=si * _REFINE_RETRIES + r,
                    on_step_fail=fail.record)
                if success:
                    any_success = True
                    break
                if fail.idx > deepest_idx:
                    deepest_idx, deepest_reason = fail.idx, fail.reason
            verdict = "REFINED-OK" if any_success else \
                f"FAILED (stuck step {deepest_idx}: {deepest_reason[:60]})"
            head = " -> ".join(f"{o}({','.join(a)})" for o, a, _ in steps)
            print(f"   sketch{si} [{len(steps)} steps]: {verdict}")
            print(f"            {head}")


if __name__ == "__main__":
    main()
