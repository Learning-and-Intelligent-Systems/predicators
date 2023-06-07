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
        --max_num_steps_interaction_request 10 \
        --bumpy_cover_num_bumps 2 \
        --bumpy_cover_spaces_per_bump 1 \
        --sampler_mlp_classifier_max_itr 100000


Bumpy cover medium:
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
        --max_num_steps_interaction_request 100 \
        --bumpy_cover_num_bumps 2 \
        --bumpy_cover_spaces_per_bump 5 \
        --sampler_mlp_classifier_max_itr 100000
"""

from dataclasses import dataclass
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
from gym.spaces import Box
from scipy.special import logsumexp

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import OnlineNSRTLearningApproach
from predicators.explorers import create_explorer
from predicators.ml_models import BinaryClassifierEnsemble, \
    KNeighborsClassifier, LearnedPredicateClassifier, MLPBinaryClassifier, BinaryClassifier
from predicators.settings import CFG
from predicators.structs import Dataset, GroundAtom, GroundAtomsHoldQuery, \
    GroundAtomsHoldResponse, InteractionRequest, InteractionResult, \
    LowLevelTrajectory, Predicate, Query, State, Task, \
    Type, ParameterizedOption, Object, Array, _Option, _GroundNSRT, NSRTSampler, Variable, NSRT


_SamplerClassifierInput = Tuple[State, Sequence[Object], Array]


class ActiveSamplerLearningApproach(OnlineNSRTLearningApproach):
    """Performs active sampler learning."""
    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)

        assert CFG.sampler_disable_classifier
        assert CFG.strips_learner

        # For each option, record all sampler inputs and parameters and whether the immediate execution of the sampler led to a success or failure.
        self._sampler_data: Dict[ParameterizedOption, List[Tuple[_SamplerClassifierInput, bool]]] = {
            option: [] for option in initial_options
        }

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
        # Re-learn sampler classifiers. Updates the NSRTs.
        self._learn_sampler_classifiers()

    def _update_sampler_data(self, trajectories: List[LowLevelTrajectory]):
        # TODO: deal with multi-step options; refactor to use segments.
        preds = self._get_current_predicates()
        for traj in trajectories:
            for state, action, next_state in zip(traj.states[:-1], traj.actions, traj.states[1:]):
                option = action.get_option()
                atoms = utils.abstract(state, preds)
                next_atoms = utils.abstract(next_state, preds)
                ground_nsrt = self._option_to_ground_nsrt(option, set(state), atoms, next_atoms)
                classifier_input = (state, ground_nsrt.objects, option.params)
                classifier_output = self._check_nsrt_success(ground_nsrt, next_atoms)
                self._sampler_data[option.parent].append((classifier_input, classifier_output))

    def _option_to_ground_nsrt(self, option: _Option, objects: Set[Object], atoms: Set[GroundAtom], next_atoms: Set[GroundAtom]) -> _GroundNSRT:
        nsrt_matches = [n for n in self._nsrts if n.option == option.parent]
        assert len(nsrt_matches) == 1
        nsrt = nsrt_matches[0]
        return nsrt.ground(option.objects)

    def _check_nsrt_success(self, ground_nsrt: _GroundNSRT, next_atoms: Set[GroundAtom]) -> bool:
        return ground_nsrt.add_effects.issubset(next_atoms) and not ground_nsrt.delete_effects.issubset(next_atoms)

    def _learn_sampler_classifiers(self) -> None:
        """Learn classifiers to re-weight the base samplers. Update the NSRTs in place."""
        new_nsrts = set()
        for option, data in self._sampler_data.items():
            logging.info(f"Fitting residual classifier for {option.name}...")
            X_classifier: List[List[Array]] = []
            y_classifier: List[int] = []
            for (state, objects, params), label in data:
                # input is state features and option parameters
                X_classifier.append([np.array(1.0)])  # start with bias term
                for obj in objects:
                    X_classifier[-1].extend(state[obj])
                X_classifier[-1].extend(params)
                assert not CFG.sampler_learning_use_goals
                y_classifier.append(label)
            X_arr_classifier = np.array(X_classifier)
            # output is binary signal
            y_arr_classifier = np.array(y_classifier)
            classifier = MLPBinaryClassifier(
                seed=CFG.seed,
                balance_data=CFG.mlp_classifier_balance_data,
                max_train_iters=CFG.sampler_mlp_classifier_max_itr,
                learning_rate=CFG.learning_rate,
                weight_decay=CFG.weight_decay,
                use_torch_gpu=CFG.use_torch_gpu,
                train_print_every=CFG.pytorch_train_print_every,
                n_iter_no_change=CFG.mlp_classifier_n_iter_no_change,
                hid_sizes=CFG.mlp_classifier_hid_sizes,
                n_reinitialize_tries=CFG.sampler_mlp_classifier_n_reinitialize_tries,
                weight_init="default")
            classifier.fit(X_arr_classifier, y_arr_classifier)
            nsrt = next(n for n in self._nsrts if n.option == option)
            base_sampler = nsrt._sampler
            wrapped_sampler = _WrappedSampler(base_sampler, classifier, nsrt.parameters, nsrt.option)
            # Create new NSRT with wrapped sampler.
            new_nsrt = NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions, nsrt.add_effects,
                    nsrt.delete_effects, nsrt.ignore_effects, nsrt.option,
                    nsrt.option_vars, wrapped_sampler.sampler)
            new_nsrts.add(new_nsrt)
        self._nsrts = new_nsrts



@dataclass(frozen=True, eq=False, repr=False)
class _WrappedSampler:
    """Wraps a base sampler with a classifier.
    
    The class probabilities of the classifier are used to select among
    multiple candidate samples from the base sampler.
    """
    _base_sampler: NSRTSampler
    _classifier: BinaryClassifier
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption

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
            score = self._classifier.predict_proba(np.r_[x, params])
            samples.append(params)
            scores.append(score)
        
        # Add a little bit of noise to promote exploration.
        eps = CFG.active_sampler_learning_score_eps
        scores = scores + rng.uniform(-eps, eps, size=len(scores))

        idx = np.argmax(scores)
        return samples[idx]
