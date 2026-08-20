"""Agent plan explorer: Claude agent generates grounded option plans.

Produces fully-grounded option plans (including continuous parameters)
and rolls them out in the real environment. Unlike
``AgentBilevelExplorer``, it runs no backtracking refinement against a
learned option model; the agent must supply complete parameters itself.
"""

import logging
from typing import Any, Dict, List, Set

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.session_base import AgentSessionFatalError
from predicators.agent_sdk.session_manager import SessionManagerProtocol, \
    run_query_sync
from predicators.agent_sdk.tools import ToolContext
from predicators.explorers.base_explorer import BaseExplorer
from predicators.settings import CFG
from predicators.structs import Action, ExplorationStrategy, \
    ParameterizedOption, Predicate, State, Task, Type


class AgentPlanExplorer(BaseExplorer):
    """Queries a Claude agent to produce grounded option plans."""

    def __init__(self, predicates: Set[Predicate],
                 options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task],
                 max_steps_before_termination: int, tool_context: ToolContext,
                 agent_session: SessionManagerProtocol) -> None:
        super().__init__(predicates, options, types, action_space, train_tasks,
                         max_steps_before_termination)
        self._tool_context = tool_context
        self._agent_session = agent_session

    @classmethod
    def get_name(cls) -> str:
        return "agent_plan"

    def _get_exploration_strategy(self, train_task_idx: int,
                                  timeout: int) -> ExplorationStrategy:
        task = self._train_tasks[train_task_idx]
        try:
            prompt = self._build_exploration_prompt(train_task_idx)
            responses = run_query_sync(self._agent_session,
                                       prompt,
                                       kind="explore")
            plan_text = self._extract_option_plan_text(responses)
            if plan_text:
                option_plan = self._parse_and_ground_plan(plan_text, task)
                if option_plan:
                    policy = utils.option_plan_to_policy(option_plan)
                    return policy, lambda _: False
            logging.info("Agent explorer: no valid plan, falling back to "
                         "random options.")
        except AgentSessionFatalError:
            # A random fallback would hide the broken session backend;
            # re-raise so the run terminates.
            raise
        except Exception as e:  # pylint: disable=broad-except
            logging.warning(f"Agent explorer failed: {e}. "
                            "Falling back to random options.")

        if not CFG.agent_explorer_fallback_to_random:
            raise utils.RequestActPolicyFailure(
                "Agent explorer failed and fallback disabled.")
        return self._random_options_fallback()

    def _random_options_fallback(self) -> ExplorationStrategy:
        """Fall back to random option sampling."""

        def fallback_policy(state: State) -> Action:
            del state
            raise utils.RequestActPolicyFailure(
                "Random option sampling failed!")

        policy = utils.create_random_option_policy(self._options, self._rng,
                                                   fallback_policy)
        return policy, lambda _: False

    def _build_exploration_prompt(self, train_task_idx: int) -> str:
        """Build a prompt for the agent to produce an option plan."""
        task = self._train_tasks[train_task_idx]
        init_state = task.init

        objects = list(init_state)
        obj_strs = []
        for obj in sorted(objects, key=lambda o: o.name):
            obj_strs.append(f"  {obj.name}: {obj.type.name}")

        # Goal atoms
        goal_strs = [str(a) for a in sorted(task.goal, key=str)]

        # Available options with signatures, including just-proposed ones.
        all_options = (self._options
                       |
                       self._tool_context.iteration_proposals.proposed_options)
        option_strs = []
        for opt in sorted(all_options, key=lambda o: o.name):
            type_sig = ", ".join(t.name for t in opt.types)
            params_dim = opt.params_space.shape[0]
            if params_dim > 0:
                low = opt.params_space.low.tolist()
                high = opt.params_space.high.tolist()
                param_info = (f", params_dim={params_dim}, "
                              f"low={low}, high={high}")
            else:
                param_info = ""
            option_strs.append(f"  {opt.name}({type_sig}{param_info})")

        # Current atoms
        atoms = utils.abstract(init_state, self._predicates)
        atom_strs = [str(a) for a in sorted(atoms, key=str)]

        # Trajectory summary
        traj_summary = self._build_trajectory_summary()

        # Planning results
        planning_info = ""
        if self._tool_context.planning_results:
            pr = self._tool_context.planning_results
            planning_info = (
                f"\n## Recent Planning Results\n"
                f"Success rate: {pr.get('success_str', 'N/A')}\n"
                f"Avg nodes expanded: {pr.get('avg_nodes_expanded', 'N/A')}\n"
                f"Failures: {pr.get('failure_summaries', 'None')}\n")

        # Available tools
        tools_str = ""
        if self._agent_session.tool_names:
            tool_list = "\n".join(f"  - {t}"
                                  for t in self._agent_session.tool_names)
            tools_str = f"\n## Available Tools\n{tool_list}\n"

        task_intro = ("You are exploring a task environment. "
                      f"Generate an option plan to explore task "
                      f"{train_task_idx}.")
        prompt = f"""{task_intro}

## Goal
{chr(10).join(goal_strs)}

## Initial State Atoms
{chr(10).join(atom_strs)}

## Objects
{chr(10).join(obj_strs)}

## Available Options
{chr(10).join(option_strs)}
{traj_summary}{planning_info}{tools_str}
## Instructions
Use your available tools to inspect the environment and test your plan before committing to it.

Output an option plan, one option per line, in this exact format:
OptionName(obj1:type1, obj2:type2)[param1, param2]

If an option has no continuous parameters, use empty brackets: OptionName(obj1:type1)[]

Output ONLY the option plan lines at the end, after any analysis."""

        return prompt

    def _build_trajectory_summary(self) -> str:
        """Summarize trajectory data for the agent."""
        all_trajs = (self._tool_context.offline_trajectories +
                     self._tool_context.online_trajectories)
        if not all_trajs:
            return ""

        max_trajs = CFG.agent_sdk_max_trajectories_in_context
        recent = all_trajs[-max_trajs:]
        lines = [
            f"\n## Trajectory Summary ({len(all_trajs)} total, "
            f"showing last {len(recent)})"
        ]

        for i, traj in enumerate(recent):
            n_steps = len(traj.actions)
            init_atoms = utils.abstract(traj.states[0], self._predicates)
            final_atoms = utils.abstract(traj.states[-1], self._predicates)
            new_atoms = final_atoms - init_atoms
            lost_atoms = init_atoms - final_atoms
            lines.append(f"\nTrajectory {i}: {n_steps} steps")
            if new_atoms:
                lines.append(
                    "  Gained: " +
                    f"{', '.join(str(a) for a in sorted(new_atoms, key=str))}")
            if lost_atoms:
                lines.append(
                    "  Lost: " +
                    f"{', '.join(str(a) for a in sorted(lost_atoms, key=str))}"
                )

        return "\n".join(lines)

    def _extract_option_plan_text(self, responses: List[Dict[str,
                                                             Any]]) -> str:
        """Extract plan text from the last assistant text response.

        Uses only the final assistant message so intermediate
        reasoning/tool-call text preceding the plan is excluded.
        """
        last_text_parts: List[str] = []
        for resp in responses:
            if resp.get("type") == "assistant":
                parts = [
                    block.get("text", "") for block in resp.get("content", [])
                    if isinstance(block, dict) and block.get("type") == "text"
                ]
                if parts:
                    last_text_parts = parts
        return "\n".join(last_text_parts)

    def _parse_and_ground_plan(self, plan_text: str, task: Task) -> list:
        """Parse option plan text and ground into executable options."""
        objects = list(task.init)
        all_options = (self._options
                       |
                       self._tool_context.iteration_proposals.proposed_options)
        parsed = utils.parse_model_output_into_option_plan(
            plan_text,
            objects,
            self._types,
            all_options,
            parse_continuous_params=True)
        if not parsed:
            logging.info("Agent explorer: parsed empty option plan.")
            return []

        grounded = []
        for option, objs, params in parsed:
            try:
                ground_opt = option.ground(objs,
                                           np.array(params, dtype=np.float32))
                grounded.append(ground_opt)
            except Exception as e:  # pylint: disable=broad-except
                logging.info(f"Agent explorer: failed to ground "
                             f"option {option.name}: {e}")
                break

        if not grounded:
            logging.info("Agent explorer: no options successfully grounded.")
        else:
            logging.info(f"Agent explorer: grounded {len(grounded)} options.")
        return grounded
