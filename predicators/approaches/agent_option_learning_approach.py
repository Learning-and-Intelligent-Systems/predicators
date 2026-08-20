"""Agent option learning approach: skill invention + planning via Claude Agent
SDK.

At solve time, the agent can invent new parameterized options (using skill
factory reference files) and then plan with them.  Requires
``agent_sdk_use_docker_sandbox=True`` so the agent can read skill factory
source files in ``/sandbox/reference/``.

Example command::

    python predicators/main.py --env pybullet_domino \\
        --approach agent_option_learning --seed 0 \\
        --num_train_tasks 1 --num_test_tasks 1 \\
        --agent_sdk_use_docker_sandbox True
"""
import logging
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Set

from gym.spaces import Box

from predicators import utils
from predicators.agent_sdk.proposal_exec import ProposalBundle
from predicators.approaches.agent_model_free_approach import \
    AgentModelFreeApproach
from predicators.settings import CFG
from predicators.structs import Action, ParameterizedOption, Predicate, \
    State, Task, Type


class AgentOptionLearningApproach(AgentModelFreeApproach):
    """Option-learning planning approach using Claude Agent SDK.

    Extends AgentModelFreeApproach with the ability to invent and
    retract parameterized options at solve time.  The agent reads skill
    factory reference files and writes Python code using skill factory
    functions (create_pick_skill, create_place_skill, etc.) to define
    new options, then plans with them in the same query.
    """

    _save_suffix = "AgentOptionLearning"

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task], *args: Any,
                 **kwargs: Any) -> None:
        # Agent-specific state (before super().__init__).
        # (_agent_session_id is initialized by the session mixin.)
        self._agent_proposed_options: Set[ParameterizedOption] = set()

        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks, *args, **kwargs)

    @classmethod
    def get_name(cls) -> str:
        return "agent_option_learning"

    # ------------------------------------------------------------------ #
    # AgentSessionMixin hooks
    # ------------------------------------------------------------------ #

    def _get_agent_system_prompt(self) -> str:
        ref_root = ("/sandbox" if CFG.agent_sdk_use_docker_sandbox else ".")
        return f"""\
You are a robot planning agent that can also invent new skills. Your
primary goal is to generate an option plan to achieve task goals. If the
existing options are insufficient, you can propose new parameterized
options before planning.

## Workflow
1. **Inspect** the task, available options, and trajectory data
2. **Invent** new options if needed — either by writing and executing
   Python code directly, or by using the `propose_options` tool
3. **Test** — either write and run Python experiments to verify your
   options, or use `evaluate_option_plan` to check that a plan achieves
   the goal. Use `retract_abstractions` to remove options that don't
   work.
4. **Plan** — output the final option plan

## Skill Factories
Read the reference files in {ref_root}/reference/skill_factories/ for the
full API. Key factory functions available in the exec context for
propose_options:
- `create_pick_skill(name, types, config, get_target_pose_fn)` — \
pick up an object (move above, descend, grasp, lift). \
Continuous params: `(grasp_z_offset,)`.
- `create_place_skill(name, types, config, use_move_above=False)` — \
place a held object (move to release position, open gripper, retreat). \
No get_target_pose_fn; target comes from continuous params: \
`(target_x, target_y, release_z, target_yaw)`. Set \
`use_move_above=True` to add a MoveAbove phase before descending.
- `create_push_skill(name, types, config, get_target_pose_fn)` — \
push with standard 4-waypoint trajectory. Requires \
`config.robot_home_pos` to be set. Facing direction is \
`(sin(yaw), cos(yaw))` from `get_target_pose_fn`. \
Continuous params: `(approach_distance, contact_z_offset)`.
- `create_pour_skill(name, types, config, get_target_pose_fn)` — pour \
from a held container. `get_target_pose_fn` returns cup position. \
The skill computes jug-to-robot displacement internally using fixed \
constants (y_off=-0.135, pour_z=0.65625, handle_h=0.1). \
No continuous params.
- `create_move_to_skill(name, types, params_space, config, \
get_target_pose_fn)` — move end-effector to a target pose
- `create_wait_option(name, config, robot_type)` — hold current pose; \
annotate with `-> {{atoms}}` in the plan to specify when it should \
terminate (e.g. `Wait(robot:Robot) -> {{Boiled(water:water_type)}}`). \
Use `NOT Pred(...)` for atoms that should become false

All factories (except `create_place_skill` and `create_wait_option`) \
take a `SkillConfig` (available as `skill_config` in the exec \
context) and a `get_target_pose_fn` callback with signature \
`(state, objects, params, config) -> (x, y, z, yaw)`. The callback \
receives empty params; geometry params are continuous params of the \
output ParameterizedOption (except pour, which has no continuous \
params). `config.transport_z` controls the transport height.

Also available: `Phase`, `PhaseSkill`, `PhaseAction`,
`make_move_to_phase` for building custom multi-phase skills, and
`chain_options(name, children)` for chaining options.

## Important
- No need to import — all standard imports (np, Box,
  ParameterizedOption, State, Type, etc.), current types (e.g.
  `robot_type`, `domino_type`), predicates, and options are already
  available in the exec context.
- Only propose new options if existing ones cannot achieve the goal
- You can invent and test options in two ways: (a) write and execute
  Python code directly in the sandbox, or (b) use the `propose_options`,
  `retract_abstractions`, and `evaluate_option_plan` tools
- Always test your plan before committing
- Output the final plan in the standard format at the end

## Debugging Tips
- Use `inspect_options` with `option_name` to save an option's source
  code to ./proposed_code/<name>.py, then Read it to study the implementation
- `evaluate_option_plan` automatically saves scene images to ./test_images/
  after each step — check them to debug spatial issues
- Your session logs are in ./session_logs/ — Glob and Read them to review
  past attempts when iterating
- All proposal and option source code is in ./proposed_code/ — Read
  files there to understand how existing options work
- When `evaluate_option_plan` fails, check the "Object poses at failure"
  and "Missing goal atoms" in the output"""

    def _get_solve_tool_names(self) -> Optional[List[str]]:
        return [
            "inspect_types",
            "inspect_options",
            "inspect_trajectories",
            "inspect_train_tasks",
            "inspect_past_proposals",
            "propose_options",
            "retract_abstractions",
            "evaluate_option_plan",
        ]

    def _get_sandbox_reference_files(  # pylint: disable=useless-super-delegation
            self) -> Dict[str, str]:
        # Inherit skill_factories + options.py from AgentModelFreeApproach
        return super()._get_sandbox_reference_files()

    # ------------------------------------------------------------------ #
    # Overridable helpers (from AgentModelFreeApproach)
    # ------------------------------------------------------------------ #

    def _get_all_options(self) -> Set[ParameterizedOption]:
        # Include tool_context.options so options proposed during the query
        # (via propose_options) reach the parser before
        # _agent_proposed_options is snapshotted. iteration_proposals is a
        # fallback for when the Docker sync to tool_context.options was
        # incomplete.
        proposal_opts = self._tool_context.iteration_proposals.proposed_options
        result = (self._initial_options | self._agent_proposed_options
                  | self._tool_context.options | proposal_opts)
        if not result:
            logging.warning(
                "_get_all_options() returning empty set. "
                "initial=%d, agent_proposed=%d, ctx.options=%d, "
                "proposal_opts=%d",
                len(self._initial_options),
                len(self._agent_proposed_options),
                len(self._tool_context.options),
                len(proposal_opts),
            )
        return result

    def _sync_tool_context(self) -> None:
        """Synchronize ToolContext with current state."""
        super()._sync_tool_context()

        # Override options to include agent-proposed ones
        self._tool_context.options = self._get_all_options()

        # Inject skill factory functions + config into exec context
        self._tool_context.skill_factory_context = \
            self._build_skill_factory_context()

    # ------------------------------------------------------------------ #
    # Skill factory context
    # ------------------------------------------------------------------ #

    def _build_skill_factory_context(self) -> Dict[str, Any]:
        """Build exec context with skill factory functions for
        propose_options."""
        # pylint: disable=import-outside-toplevel
        from predicators.ground_truth_models.skill_factories import Phase, \
            PhaseAction, PhaseSkill, SkillConfig, create_move_to_skill, \
            create_pick_skill, create_place_skill, create_pour_skill, \
            create_push_skill, create_wait_option, make_move_to_phase

        context: Dict[str, Any] = {
            # Skill factory functions
            "create_pick_skill": create_pick_skill,
            "create_place_skill": create_place_skill,
            "create_push_skill": create_push_skill,
            "create_pour_skill": create_pour_skill,
            "create_move_to_skill": create_move_to_skill,
            "create_wait_option": create_wait_option,
            "make_move_to_phase": make_move_to_phase,
            # Building blocks
            "Phase": Phase,
            "PhaseAction": PhaseAction,
            "PhaseSkill": PhaseSkill,
            "SkillConfig": SkillConfig,
            # Generic helpers
            "chain_options": utils.LinearChainParameterizedOption,
        }

        # For pybullet envs, provide a pre-built SkillConfig
        if CFG.env.startswith("pybullet"):
            try:
                context["skill_config"] = self._get_skill_config()
            except Exception as e:  # pylint: disable=broad-except
                logging.warning(
                    f"Failed to build SkillConfig for {CFG.env}: {e}")

        return context

    @staticmethod
    @lru_cache(maxsize=1)
    def _get_skill_config() -> Any:
        """Lazily build a SkillConfig for the current pybullet env."""
        from predicators.ground_truth_models.skill_factories import \
            SkillConfig  # pylint: disable=import-outside-toplevel

        env_cls = _get_pybullet_env_cls(CFG.env)
        _, robot, _ = env_cls.initialize_pybullet(using_gui=False)

        simulator = env_cls(use_gui=False) \
            if CFG.skill_phase_use_motion_planning else None

        return SkillConfig(
            robot=robot,
            open_fingers_joint=robot.open_fingers,
            closed_fingers_joint=robot.closed_fingers,
            fingers_state_to_joint=(  # pylint: disable=protected-access
                env_cls._fingers_state_to_joint),
            max_vel_norm=CFG.pybullet_max_vel_norm,
            ik_validate=CFG.pybullet_ik_validate,
            robot_init_tilt=getattr(env_cls, 'robot_init_tilt', 0.0),
            robot_init_wrist=getattr(env_cls, 'robot_init_wrist', 0.0),
            robot_home_pos=(env_cls.robot_init_x, env_cls.robot_init_y,
                            env_cls.robot_init_z),
            simulator=simulator,
        )

    # ------------------------------------------------------------------ #
    # Solving (with option invention)
    # ------------------------------------------------------------------ #

    def _build_solve_prompt(self, task: Task) -> str:
        """Build solve prompt that adds skill invention instructions."""
        base_prompt = super()._build_solve_prompt(task)

        ref_root = ("/sandbox" if CFG.agent_sdk_use_docker_sandbox else ".")
        skill_instructions = f"""

## Skill Invention
You can also invent new options before planning. Follow these steps:

1. **Analyse** — Determine whether the existing options are sufficient \
to achieve the goal.
2. **Invent** — If not, read the skill factory reference files in \
{ref_root}/reference/skill_factories/ to understand how to build new \
options. You can create options in two ways:
   - **Python code**: Write and execute Python scripts that import the \
skill factories and construct options directly.
   - **MCP tools**: Use `propose_options` to create options via the \
tool interface. Use `retract_abstractions` to remove options that \
don't work.
   A pre-built `skill_config` (SkillConfig) is available in the exec \
context for pybullet environments.
3. **Test** — Verify your options and plan work correctly:
   - **Python code**: Write and run Python experiments to unit-test \
individual options or full plans.
   - **MCP tools**: Use `evaluate_option_plan` to check that a plan \
(including any new options) achieves the goal.
   Iterate until the test passes.
4. **Commit** — Once the test passes, output the final plan. Your \
proposed options will be added to the option library for future tasks."""

        return base_prompt + skill_instructions

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        """Solve with option invention enabled.

        The propose_options and retract_abstractions tools directly
        update ctx.options during the agent query.  After solving we
        snapshot the agent-proposed options for persistence.
        """
        self._tool_context.iteration_proposals = ProposalBundle()

        policy = super()._solve(task, timeout)

        # Snapshot agent-proposed options (everything beyond initial)
        self._agent_proposed_options = (self._tool_context.options -
                                        self._initial_options)

        # Record iteration summary (options only)
        proposals = self._tool_context.iteration_proposals
        summary = {
            "cycle": self._online_learning_cycle,
            "proposed_options": [o.name for o in proposals.proposed_options],
            "retracted_options": sorted(proposals.retract_option_names),
        }
        self._tool_context.iteration_history.append(summary)

        return policy

    # ------------------------------------------------------------------ #
    # Save / Load
    # ------------------------------------------------------------------ #

    def _extra_save_state(self) -> Dict[str, Any]:
        return {"agent_proposed_options": self._agent_proposed_options}

    def _load_extra_save_state(self, save_dict: Dict[str, Any]) -> None:
        self._agent_proposed_options = save_dict.get("agent_proposed_options",
                                                     set())
        logging.info("[Run %s] Restored %d agent-proposed options.",
                     self._run_id, len(self._agent_proposed_options))


# --------------------------------------------------------------------------- #
# Lazy pybullet env lookup (module-level, cached)
# --------------------------------------------------------------------------- #


@lru_cache(maxsize=1)
def _get_pybullet_env_cls(env_name: str) -> Any:
    """Look up the concrete PyBulletEnv subclass by name."""
    # pylint: disable=import-outside-toplevel
    import predicators.envs as _envs_pkg  # noqa: F401
    from predicators.envs.base_env import BaseEnv
    from predicators.envs.pybullet_env import PyBulletEnv
    for cls in utils.get_all_subclasses(BaseEnv):
        if not cls.__abstractmethods__ and cls.get_name() == env_name:
            if issubclass(cls, PyBulletEnv):
                return cls
            break
    raise RuntimeError(f"No PyBulletEnv subclass found for env '{env_name}'")
