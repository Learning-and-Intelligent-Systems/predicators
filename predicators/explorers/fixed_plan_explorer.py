"""An explorer that replays one fixed option plan from a file.

A stand-in for the planning explorers when what is being tested is the online
loop itself rather than what the agent chooses. The LLM-backed explorers cost
minutes per episode, which makes a full cycle expensive to exercise on
hardware; this one costs nothing and does the same thing every episode, so a
run that goes wrong is the loop's fault and not the planner's.

The plan file is ``replay_plan.py``'s format -- one grounded option per line,
``-> {...}`` subgoals optional and ignored::

    Pick(robot:robot, domino_1:domino)[0.0657]
    Place(robot:robot)[0.70, 1.16, 0.55, 1.75]
    Wait(robot:robot)[]

so a plan dumped by ``probe_real_scene --dump-plan`` and verified through
``replay_plan`` can be handed straight to the loop.
"""

import logging
import re
from typing import Dict, List, Tuple

import numpy as np

from predicators import utils
from predicators.explorers import BaseExplorer
from predicators.settings import CFG
from predicators.structs import ExplorationStrategy, Object, State, _Option

# Same grammar as scripts/domino_debug/replay_plan.py.
_LINE = re.compile(r"^\s*(\w+)\s*\(([^)]*)\)\s*\[([^\]]*)\]")


def _parse_plan(text: str) -> List[Tuple[str, List[str], List[float]]]:
    """[(option_name, [obj_names], [param_floats]), ...] from the plan text."""
    steps: List[Tuple[str, List[str], List[float]]] = []
    for raw in text.splitlines():
        line = raw.split("->", 1)[0].strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE.match(line)
        if not match:
            continue
        objs = [
            a.split(":", 1)[0].strip() for a in match.group(2).split(",")
            if a.strip()
        ]
        floats = [float(v) for v in match.group(3).split(",") if v.strip()]
        steps.append((match.group(1), objs, floats))
    return steps


class FixedPlanExplorer(BaseExplorer):
    """Replays the plan at ``CFG.fixed_plan_explorer_path`` every episode."""

    @classmethod
    def get_name(cls) -> str:
        return "fixed_plan"

    def _ground(self, state: State) -> List[_Option]:
        """Ground the plan's options against the objects in ``state``.

        Grounded per episode rather than once, because a human reset
        rebuilds the task and hands back fresh ``Object`` instances.
        """
        path = CFG.fixed_plan_explorer_path
        assert path, "fixed_plan explorer needs fixed_plan_explorer_path"
        with open(path, encoding="utf-8") as f:
            steps = _parse_plan(f.read())
        assert steps, f"no plan steps parsed from {path}"
        options: Dict[str, object] = {o.name: o for o in self._options}
        by_name: Dict[str, Object] = {o.name: o for o in state}
        plan = []
        for name, obj_names, params in steps:
            option = options[name]
            objs = [by_name[n] for n in obj_names]
            plan.append(
                option.ground(  # type: ignore[attr-defined]
                    objs, np.array(params, dtype=np.float32)))
        return plan

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:
        del timeout  # the plan is fixed; there is nothing to search
        state = self._train_tasks[train_task_idx].init
        plan = self._ground(state)
        logging.info("fixed_plan explorer: %s", [o.simple_str() for o in plan])
        policy = utils.option_plan_to_policy(
            plan,
            abstract_function=lambda s: utils.abstract(s, self._predicates))
        # Never terminate early: the plan running out raises
        # OptionExecutionFailure, which the interaction loop already handles,
        # and stopping sooner would cut the episode short of the plan.
        termination_function = lambda _: False
        return policy, termination_function
