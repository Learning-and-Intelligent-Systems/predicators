"""Typed configuration surfaces for the agent_sdk package.

Each dataclass groups the ``CFG`` flags one agent-sdk concern reads,
under clean field names. ``from_cfg()`` is called at USE time (handler
entry / session construction), never cached at import time, so tests
that mutate settings via ``utils.reset_config`` keep working.

The experiment flags keep their historical names (``agent_bilevel_*``,
``agent_planner_*``, ...) for experiment-yaml compatibility; the
``from_cfg()`` classmethods below are the single place those old flag
names map to the clean field names.
"""
from dataclasses import dataclass

from predicators.settings import CFG


@dataclass(frozen=True)
class SessionConfig:
    """Agent session construction: model, budgets, and sandbox choice.

    Consumed by the three session managers and by
    ``AgentSessionMixin._ensure_agent_session``, which builds one
    instance per session and passes it to whichever manager it
    constructs.
    """
    model_name: str
    reasoning_effort: str
    max_turns: int
    max_buffer_size: int
    agent_timeout: int
    use_docker_sandbox: bool
    use_local_sandbox: bool
    docker_image: str
    use_scratchpad: bool

    @classmethod
    def from_cfg(cls) -> "SessionConfig":
        """Read the session flags from the live ``CFG``."""
        # Flags keep their names for experiment-yaml compatibility.
        return cls(
            model_name=CFG.agent_sdk_model_name,
            reasoning_effort=CFG.agent_sdk_reasoning_effort,
            max_turns=CFG.agent_sdk_max_agent_turns_per_iteration,
            max_buffer_size=CFG.agent_sdk_max_buffer_size,
            agent_timeout=CFG.agent_sdk_agent_timeout,
            use_docker_sandbox=CFG.agent_sdk_use_docker_sandbox,
            use_local_sandbox=CFG.agent_sdk_use_local_sandbox,
            docker_image=CFG.agent_sdk_docker_image,
            use_scratchpad=CFG.agent_planner_use_scratchpad,
        )


@dataclass(frozen=True)
class RefinementConfig:
    """Plan-sketch refinement: search budgets, gates, and ground samplers.

    Consumed at handler entry by ``refine_plan_sketch`` /
    ``evaluate_option_plan`` (tools.py) and by the probe's ``refine``
    (belief_probe.py).
    """
    ground_samplers: bool
    refinement_timeout_per_step: float
    refinement_timeout_min: float
    max_samples_per_step: int
    check_subgoals: bool
    log_state: bool
    refine_evaluator_attempts: int
    use_llm_initial_params: bool

    @classmethod
    def from_cfg(cls) -> "RefinementConfig":
        """Read the refinement flags from the live ``CFG``."""
        # Flags keep their names for experiment-yaml compatibility.
        return cls(
            ground_samplers=CFG.agent_bilevel_ground_samplers,
            refinement_timeout_per_step=(
                CFG.agent_bilevel_refinement_timeout_per_step),
            refinement_timeout_min=CFG.agent_bilevel_refinement_timeout_min,
            max_samples_per_step=CFG.agent_bilevel_max_samples_per_step,
            check_subgoals=CFG.agent_bilevel_check_subgoals,
            log_state=CFG.agent_bilevel_log_state,
            refine_evaluator_attempts=(
                CFG.agent_bilevel_refine_evaluator_attempts),
            use_llm_initial_params=CFG.agent_bilevel_use_llm_initial_params,
        )


@dataclass(frozen=True)
class ValidationConfig:
    """Capture-validation rollouts and the cross-attempt journal.

    Consumed at handler entry by ``evaluate_option_plan`` (tools.py) and
    the probe's ``run(trials=N)`` (belief_probe.py); ``use_journal``
    gates the ``record_journal`` tool.
    """
    rollouts: int
    rollouts_after_flaky: int
    fresh_env: bool
    physics_margin: bool
    use_journal: bool

    @classmethod
    def from_cfg(cls) -> "ValidationConfig":
        """Read the validation flags from the live ``CFG``."""
        # Flags keep their names for experiment-yaml compatibility.
        return cls(
            rollouts=CFG.agent_plan_validation_rollouts,
            rollouts_after_flaky=(
                CFG.agent_plan_validation_rollouts_after_flaky),
            fresh_env=CFG.agent_plan_validation_fresh_env,
            physics_margin=CFG.agent_plan_validation_physics_margin,
            use_journal=CFG.agent_solve_use_journal,
        )


@dataclass(frozen=True)
class ToolSurfaceConfig:
    """Which optional tools a session offers, and their surface knobs.

    Consumed by the tool builders in tools.py (gates + descriptions
    baked at build time), the proposal handlers (call-time gates), image
    sizing, and the sandbox CLAUDE.md builder (sandbox_prompts.py).
    """
    use_explore_python: bool
    explore_python_keep_replaced_tools: bool
    use_base_simulator: bool
    explore_python_call_timeout: float
    image_max_px: int
    propose_types: bool
    propose_predicates: bool
    propose_processes: bool
    propose_options: bool
    propose_objects: bool

    @classmethod
    def from_cfg(cls) -> "ToolSurfaceConfig":
        """Read the tool-surface flags from the live ``CFG``."""
        # Flags keep their names for experiment-yaml compatibility.
        return cls(
            use_explore_python=CFG.agent_planner_use_explore_python,
            explore_python_keep_replaced_tools=(
                CFG.agent_planner_explore_python_keep_replaced_tools),
            use_base_simulator=CFG.agent_planner_use_base_simulator,
            explore_python_call_timeout=(
                CFG.agent_sdk_explore_python_call_timeout),
            image_max_px=CFG.agent_sdk_image_max_px,
            propose_types=CFG.agent_sdk_propose_types,
            propose_predicates=CFG.agent_sdk_propose_predicates,
            propose_processes=CFG.agent_sdk_propose_processes,
            propose_options=CFG.agent_sdk_propose_options,
            propose_objects=CFG.agent_sdk_propose_objects,
        )
