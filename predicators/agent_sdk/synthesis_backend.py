"""Structural typing contract between synthesis tools and the approach.

:class:`SynthesisBackend` declares exactly the approach surface that the
synthesis tool factories in :mod:`predicators.agent_sdk.tools`
(``create_synthesis_tools``, ``create_predicate_synthesis_tools``,
``create_sampler_synthesis_tools``) and the approach-layer validation
glue in :mod:`predicators.approaches.synthesis_validation` dereference.
It exists so those modules can be typed against the contract instead of
importing the concrete ``AgentSimLearningApproach`` - the import that
previously made ``agent_sdk`` depend on the approach layer.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, \
    Protocol, Sequence, Set, Tuple

if TYPE_CHECKING:
    from predicators.code_sim_learning.fit_space import FitResult, ParamSpec
    from predicators.code_sim_learning.rollout_env import RolloutTrajectory
    from predicators.code_sim_learning.utils import LearnedSimulator
    from predicators.option_model import _OracleOptionModel
    from predicators.structs import Action, LowLevelTrajectory, \
        ParameterizedOption, ParameterizedSampler, Predicate, State, Task, \
        Type


class SynthesisBackend(Protocol):
    """Structural contract ``AgentSimLearningApproach`` implements for the
    synthesis tool factories.

    The synthesis tools mutate approach state (fitted params, learned
    predicates, synthesized samplers, candidate residual rules) so the
    live session and later refinement calls see the agent's drafts;
    everything else is read-only access to the approach's env, tasks,
    vocabulary, and fitting engine. Member names mirror the (mostly
    protected) names on the approach - this protocol is the one place
    that documents them as a public-for-tools surface.

    Implemented (structurally - no inheritance) by
    ``predicators.approaches.agent_sim_learning_approach.
    AgentSimLearningApproach`` and its subclasses.
    """

    # ── State read by the tools ──────────────────────────────────
    # The planning base env (physical-param registry + overrides).
    _base_env: Any
    # Mutated in place (clear + update) on every re-fit; predicate
    # classifiers hold a live view onto this exact dict object.
    _fitted_params: Dict[str, float]
    _train_tasks: List[Task]
    _types: Set[Type]
    _initial_predicates: Set[Predicate]
    _initial_options: Set[ParameterizedOption]
    # Per-fit cache of best-achievable per-segment RMS (system ID).
    _explainability_cache: Dict[Tuple, Tuple[List[float], List[Dict[str,
                                                                    float]]]]

    # ── State written by the tools ───────────────────────────────
    # Per-skill samplers keyed by option name.
    _synthesized_samplers: Dict[str, ParameterizedSampler]
    # Candidate simulator state published during validation so the
    # recurrent combined simulator sees the rules under evaluation.
    _residual_rules: Optional[List]
    _latent_init: Any

    # ── Vocabulary / engine accessors ────────────────────────────
    def _get_all_predicates(self) -> Set[Predicate]:
        ...

    def _get_all_options(self) -> Set[ParameterizedOption]:
        ...

    def _get_all_samplers(self) -> Dict[str, ParameterizedSampler]:
        ...

    def _group_triples_by_trajectory(
        self,
        triples: List[Tuple[State, Action, State]],
    ) -> List[List[Tuple[State, Action, State]]]:
        ...

    def _rollout_fit_trajectories(
        self,
        residual_features: Optional[Dict[str, List[str]]] = None,
        traj_idxs: Optional[Sequence[int]] = None,
    ) -> List[RolloutTrajectory]:
        ...

    def _get_rollout_fit_env(self) -> Any:
        ...

    def _apply_identified_physical_params(
            self, identified: Dict[str, float]) -> None:
        ...

    def _record_sysid_diagnostics(self, report: Dict[str, Dict[str, Any]],
                                  physical_names: Sequence[str],
                                  num_survivors: int, num_segments: int,
                                  rms: List[float]) -> None:
        ...

    def _fit_parameters_recurrent(
        self,
        rules: List,
        specs: List[ParamSpec],
        base_pred_triples: List[Tuple[State, Action, State]],
        residual_features: Dict[str, List[str]],
        num_steps: Optional[int] = None,
    ) -> Tuple[FitResult, float]:
        ...

    def _fit_parameters_joint_rollout(
        self,
        rules: List,
        rule_specs: List[ParamSpec],
        residual_features: Dict[str, List[str]],
    ) -> Tuple[FitResult, float]:
        ...

    def _build_combined_simulator(
        self,
        learned_simulator: LearnedSimulator,
    ) -> Callable[[State, Action], State]:
        ...

    def _build_option_model(
        self,
        simulator_fn: Callable[[State, Action], State],
    ) -> _OracleOptionModel:
        ...

    def materialise_latent(
        self,
        traj: LowLevelTrajectory,
    ) -> List[Optional[Dict[str, Any]]]:
        """Per-step latent dicts for a trajectory, if the rule has one."""


class PredicateSynthesisBackend(SynthesisBackend, Protocol):
    """The extra surface ``create_predicate_synthesis_tools`` needs.

    Only the predicate-invention subclass provides these, so they live
    off the core protocol.
    """

    # Agent-invented predicates (read back through
    # ``_get_all_predicates``); replaced wholesale on each validation.
    _learned_predicates: Set[Predicate]
    # Initial predicates that survived retraction, used to build the
    # exec namespace the agent's predicate code runs in.
    _kept_initial_predicates: Set[Predicate]


class SamplerSynthesisBackend(Protocol):
    """The narrow surface ``create_sampler_synthesis_tools`` needs.

    ``SamplerLearningMixin`` satisfies this directly (its declared host-
    class contract covers every member), so the mixin can pass ``self``
    without seeing the full backend.
    """

    _fitted_params: Dict[str, float]
    _train_tasks: List[Task]
    _types: Set[Type]
    _synthesized_samplers: Dict[str, ParameterizedSampler]

    def _get_all_predicates(self) -> Set[Predicate]:
        ...

    def _get_all_options(self) -> Set[ParameterizedOption]:
        ...
