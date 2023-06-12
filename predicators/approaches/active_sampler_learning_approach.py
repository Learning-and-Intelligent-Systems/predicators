"""An approach that performs active sampler learning.

The current implementation assumes for convenience that NSRTs and options are 1:1 and
share the same parameters (like a PDDL environment). It is straightforward conceptually
to remove this assumption, because the approach uses its own NSRTs to select options,
but it is difficult implementation-wise, so we're punting for now.


Example commands
----------------

Bumpy cover easy:
    python predicators/main.py --approach active_sampler_learning --env bumpy_cover \
        --seed 0 \
        --strips_learner oracle \
        --sampler_learner oracle \
        --sampler_disable_classifier True \
        --bilevel_plan_without_sim True \
        --offline_data_bilevel_plan_without_sim False \
        --explorer random_nsrts \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 4 \
        --bumpy_cover_num_bumps 2 \
        --bumpy_cover_spaces_per_bump 1 \
        --mlp_regressor_max_itr 100000 \
        --pytorch_train_print_every 10000


Bumpy cover with shifted targets:
    python predicators/main.py --approach active_sampler_learning --env bumpy_cover \
        --seed 0 \
        --strips_learner oracle \
        --sampler_learner oracle \
        --sampler_disable_classifier True \
        --bilevel_plan_without_sim True \
        --offline_data_bilevel_plan_without_sim False \
        --explorer random_nsrts \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 4 \
        --bumpy_cover_num_bumps 2 \
        --bumpy_cover_spaces_per_bump 1 \
        --mlp_regressor_max_itr 1000000 \
        --pytorch_train_print_every 10000 \
        --bumpy_cover_right_targets True
"""

import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
from gym.spaces import Box
from scipy.special import logsumexp

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.explorers import create_explorer
from predicators.ml_models import MLPRegressor
from predicators.settings import CFG
from predicators.structs import NSRT, Array, Dataset, GroundAtom, \
    GroundAtomsHoldQuery, GroundAtomsHoldResponse, InteractionRequest, \
    InteractionResult, LowLevelTrajectory, NSRTSampler, Object, \
    ParameterizedOption, Predicate, Query, State, Task, Type, Variable, \
    _GroundNSRT, _Option

_SamplerRegressorInput = Tuple[State, Sequence[Object], Array]


class ActiveSamplerLearningApproach(OnlineNSRTLearningApproach):
    """Performs active sampler learning."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)

        assert CFG.sampler_disable_classifier
        assert CFG.strips_learner

        self._sampler_data: Dict[ParameterizedOption,
                                 List[Tuple[_SamplerRegressorInput,
                                            float]]] = {
                                                option: []
                                                for option in initial_options
                                            }
        self._current_sampler_noise = 0.0  # updated later

    @classmethod
    def get_name(cls) -> str:
        return "active_sampler_learning"

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     online_learning_cycle: Optional[int],
                     annotations: Optional[List[Any]]) -> None:
        # Start by learning NSRTs in the usual way.
        super()._learn_nsrts(trajectories, online_learning_cycle, annotations)
        # Check the assumption that operators and options are 1:1.
        # This is just an implementation convenience.
        assert len({nsrt.option for nsrt in self._nsrts}) == len(self._nsrts)
        for nsrt in self._nsrts:
            assert nsrt.option_vars == nsrt.parameters
        # Update the sampler data.
        self._update_sampler_data(trajectories)
        # Re-learn sampler regressors. Updates the NSRTs.
        self._learn_sampler_regressors(online_learning_cycle)

    def _update_sampler_data(self, trajectories: List[LowLevelTrajectory]):
        # TODO: deal with multi-step options; refactor to use segments.
        preds = self._get_current_predicates()
        for traj in trajectories:
            # Label the steps according to whether the operators succeeded.
            refinement_successes: List[bool] = []
            atom_seq = [utils.abstract(state, preds) for state in traj.states]
            for t in range(len(atom_seq) - 1):
                atoms = atom_seq[t]
                next_atoms = atom_seq[t + 1]
                objects = set(traj.states[t])
                option = traj.actions[t].get_option()
                ground_nsrt = self._option_to_ground_nsrt(
                    option, objects, atoms, next_atoms)
                success = self._check_nsrt_success(ground_nsrt, next_atoms)
                refinement_successes.append(success)
            # Create regressor data.
            for t in range(len(atom_seq) - 1):
                # Find the first failure.
                first_failure_step: Optional[int] = None
                for i in range(t, len(atom_seq) - 1):
                    if not refinement_successes[i]:
                        first_failure_step = i
                        break
                if first_failure_step is None:
                    score = 0.0
                else:
                    dt = first_failure_step - t
                    score = -CFG.active_sampler_learning_score_gamma**dt
                # Set up the input.
                atoms = atom_seq[t]
                next_atoms = atom_seq[t + 1]
                state = traj.states[t]
                objects = set(state)
                option = traj.actions[t].get_option()
                ground_nsrt = self._option_to_ground_nsrt(
                    option, objects, atoms, next_atoms)
                regressor_input = (state, ground_nsrt.objects, option.params)
                self._sampler_data[option.parent].append(
                    (regressor_input, score))

    def _option_to_ground_nsrt(self, option: _Option, objects: Set[Object],
                               atoms: Set[GroundAtom],
                               next_atoms: Set[GroundAtom]) -> _GroundNSRT:
        nsrt_matches = [n for n in self._nsrts if n.option == option.parent]
        assert len(nsrt_matches) == 1
        nsrt = nsrt_matches[0]
        return nsrt.ground(option.objects)

    def _check_nsrt_success(self, ground_nsrt: _GroundNSRT,
                            next_atoms: Set[GroundAtom]) -> bool:
        return ground_nsrt.add_effects.issubset(
            next_atoms) and not ground_nsrt.delete_effects.issubset(next_atoms)

    def _learn_sampler_regressors(
            self, online_learning_cycle: Optional[int]) -> None:
        """Learn regressors to re-weight the base samplers.

        Update the NSRTs in place.
        """
        new_nsrts = set()
        for option, data in self._sampler_data.items():
            logging.info(f"Fitting residual regressor for {option.name}...")
            X_regressor: List[List[Array]] = []
            y_regressor: List[List[Array]] = []
            for (state, objects, params), target in data:
                # input is state features and option parameters
                X_regressor.append([np.array(1.0)])  # start with bias term
                for obj in objects:
                    X_regressor[-1].extend(state[obj])
                X_regressor[-1].extend(params)
                assert not CFG.sampler_learning_use_goals
                y_regressor.append([target])
            X_arr_regressor = np.array(X_regressor)
            y_arr_regressor = np.array(y_regressor)
            regressor = MLPRegressor(
                seed=CFG.seed,
                hid_sizes=CFG.mlp_regressor_hid_sizes,
                max_train_iters=CFG.mlp_regressor_max_itr,
                clip_gradients=CFG.mlp_regressor_clip_gradients,
                clip_value=CFG.mlp_regressor_gradient_clip_value,
                learning_rate=CFG.learning_rate,
                weight_decay=CFG.weight_decay,
                use_torch_gpu=CFG.use_torch_gpu,
                train_print_every=CFG.pytorch_train_print_every)
            regressor.fit(X_arr_regressor, y_arr_regressor)

            # Save the sampler regressor for external analysis.
            approach_save_path = utils.get_approach_save_path_str()
            save_path = f"{approach_save_path}_{option.name}_{online_learning_cycle}.sampler_regressor"
            with open(save_path, "wb") as f:
                pkl.dump(regressor, f)
            logging.info(f"Saved sampler regressor to {save_path}.")

            nsrt = next(n for n in self._nsrts if n.option == option)
            base_sampler = nsrt._sampler
            wrapped_sampler = _WrappedSampler(base_sampler, regressor,
                                              nsrt.parameters, nsrt.option,
                                              self._get_sampler_noise)
            # Create new NSRT with wrapped sampler.
            new_nsrt = NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                            nsrt.add_effects, nsrt.delete_effects,
                            nsrt.ignore_effects, nsrt.option, nsrt.option_vars,
                            wrapped_sampler.sampler)
            new_nsrts.add(new_nsrt)
        self._nsrts = new_nsrts

    def _get_sampler_noise(self) -> float:
        """Change the sampler noise depending on whether we're training on
        evaluating."""
        return self._current_sampler_noise

    def get_interaction_requests(self) -> List[InteractionRequest]:
        self._current_sampler_noise = 0.25  # high noise
        return super().get_interaction_requests()

    def learn_from_interaction_results(
            self, results: Sequence[InteractionResult]) -> None:
        self._current_sampler_noise = 0.01  # low noise
        return super().learn_from_interaction_results(results)


@dataclass(frozen=True, eq=False, repr=False)
class _WrappedSampler:
    """Wraps a base sampler with a regressor.

    The outputs of the regressor are used to select among multiple
    candidate samples from the base sampler.
    """
    _base_sampler: NSRTSampler
    _regressor: MLPRegressor
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption
    _get_explore_noise: Callable[[], float]

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object]) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        x_lst: List[Any] = [1.0]  # start with bias term
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
        assert not CFG.sampler_learning_use_goals
        x = np.array(x_lst)

        samples = []
        scores = []
        for _ in range(CFG.active_sampler_learning_num_samples):
            params = self._base_sampler(state, goal, rng, objects)
            assert self._param_option.params_space.contains(params)
            score = self._regressor.predict(np.r_[x, params])[0]
            samples.append(params)
            scores.append(score)

        # Add a little bit of noise to promote exploration.
        eps = self._get_explore_noise()
        scores = scores + np.array(rng.uniform(-eps, eps, size=len(scores)))

        idx = np.argmax(scores)
        return samples[idx]
