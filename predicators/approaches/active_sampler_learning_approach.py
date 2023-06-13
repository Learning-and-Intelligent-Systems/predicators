"""An approach that performs active sampler learning.

The current implementation assumes for convenience that NSRTs and options are
1:1 and share the same parameters (like a PDDL environment). It is
straightforward conceptually to remove this assumption, because the approach
uses its own NSRTs to select options, but it is difficult implementation-wise,
so we're punting for now.


Example commands
----------------

Bumpy cover easy:
    python predicators/main.py --approach active_sampler_learning \
        --env bumpy_cover \
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
    python predicators/main.py --approach active_sampler_learning \
        --env bumpy_cover \
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

import abc
import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.ml_models import MLPRegressor
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LowLevelTrajectory, \
    NSRTSampler, Object, ParameterizedOption, Predicate, Segment, State, \
    Task, Type, Variable, _GroundNSRT, _Option

# Dataset for sampler learning: includes (s, option, s', label) per NSRT.
_NSRTSamplerDataset = List[Tuple[State, _Option, State, Any]]
_SamplerDataset = Dict[ParameterizedOption, _NSRTSamplerDataset]
# Score function used to wrap samplers.
_ScoreFn = Callable[[State, Sequence[Object], List[Array]], List[float]]


class ActiveSamplerLearningApproach(OnlineNSRTLearningApproach):
    """Performs active sampler learning."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)

        assert CFG.sampler_disable_classifier
        assert CFG.strips_learner

        self._sampler_data: _SamplerDataset = {}

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
        # Update the sampler data using the updated self._segmented_trajs.
        self._update_sampler_data()
        # Re-learn samplers. Updates the NSRTs.
        self._learn_wrapped_samplers(online_learning_cycle)

    def _update_sampler_data(self) -> None:
        for segmented_traj in self._segmented_trajs:
            for segment in segmented_traj:
                s = segment.states[0]
                o = segment.get_option()
                ns = segment.states[-1]

                # Forthcoming implementations may change the label here.
                success = self._check_option_success(option, segment)
                label = success

                # Store transition per ParameterizedOption. Don't store by
                # NSRT because those change as we re-learn.
                self._sampler_data[option.parent].append((s, o, ns, label))

    def _check_option_success(self, option: _Option, segment: Segment) -> bool:
        ground_nsrt = _option_to_ground_nsrt(option, self._nsrts)
        return ground_nsrt.add_effects.issubset(
            segment.final_atoms) and not ground_nsrt.delete_effects.issubset(
                segment.final_atoms)

    def _learn_wrapped_samplers(
            self, online_learning_cycle: Optional[int]) -> None:
        """Update the NSRTs in place."""
        # Forthcoming approaches may use a different learner.
        learner = _ClassifierWrappedSamplerLearner(self._get_current_nsrts(),
                             self._get_current_predicates(),
                             online_learning_cycle)
        # Fit with the current data.
        learner.learn(self._sampler_data)
        wrapped_samplers = learner.get_samplers()
        # Update the NSRTs.
        new_nsrts = set()
        for nsrt, sampler in wrapped_samplers.items():
            # Create new NSRT.
            new_nsrt = NSRT(nsrt.name, nsrt.parameters, nsrt.preconditions,
                            nsrt.add_effects, nsrt.delete_effects,
                            nsrt.ignore_effects, nsrt.option, nsrt.option_vars,
                            sampler)
            new_nsrts.add(new_nsrt)
        self._nsrts = new_nsrts


class _WrappedSamplerLearner(abc.ABC):
    """A base class for learning wrapped samplers."""

    def __init__(self, nsrts: Set[NSRT], predicates: Set[Predicate],
                 online_learning_cycle: Optional[int]) -> None:
        self._nsrts = nsrts
        self._predicates = predicates
        self._online_learning_cycle = online_learning_cycle
        self._rng = np.random.default_rng(CFG.seed)
        self._learned_samplers: Optional[Dict[NSRT, NSRTSampler]] = None

    @abc.abstractmethod
    def learn(self, data: _SamplerDataset) -> None:
        """Fit all of the samplers."""

    def get_samplers(self) -> Dict[NSRT, NSRTSampler]:
        """Expose the fitted samplers, organized by NSRTs."""
        assert self._learned_samplers is not None
        return self._learned_samplers


class _ClassifierWrappedSamplerLearner(_WrappedSamplerLearner):
    """Using boolean class labels on transitions, learn a classifier, and then
    use the probability of predicting True to select parameters.

    TODO: don't forget to save classifiers for external analysis.
    """"

    def learn(self, data: _SamplerDataset) -> None:
        # Learn the sampler for each NSRT independently.
        new_samplers: Dict[NSRT, NSRTSampler] = {}
        for param_opt, nsrt_data in data.items():
            nsrt = _param_option_to_nsrt(param_opt, self._nsrts)
            new_samplers[nsrt] = self._learn_nsrt_sampler(nsrt_data, nsrt)
        self._learned_samplers = new_samplers

    def _learn_nsrt_sampler(self, nsrt_data: _NSRTSamplerDataset, nsrt: NSRT) -> NSRTSampler:
        logging.info(f"Fitting wrapped sampler classifier for {nsrt.name}...")
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
            n_reinitialize_tries=CFG.
            sampler_mlp_classifier_n_reinitialize_tries,
            weight_init="default")
        classifier.fit(X_arr_classifier, y_arr_classifier)

        # Save the sampler classifier for external analysis.
        approach_save_path = utils.get_approach_save_path_str()
        save_path = f"{approach_save_path}_{nsrt.name}_{self._online_learning_cycle}.sampler_classifier"
        with open(save_path, "wb") as f:
            pkl.dump(classifier, f)
        logging.info(f"Saved sampler classifier to {save_path}.")

        # Easiest way to access the base sampler.
        base_sampler = nsrt._sampler
        score_fn = _classifier_to_score_fn(classifier, nsrt)
        wrapped_sampler = _wrap_sampler(base_sampler, score_fn)

        return wrapped_sampler



# Helper functions.
def _param_option_to_nsrt(param_option: ParameterizedOption, nsrts: Set[NSRT]) -> NSRT:
    """Assumes 1:1 options and NSRTs, see note in file docstring."""
    nsrt_matches = [n for n in nsrts if n.option == option.parent]
    assert len(nsrt_matches) == 1
    nsrt = nsrt_matches[0]
    return nsrt


def _option_to_ground_nsrt(option: _Option, nsrts: Set[NSRT]) -> _GroundNSRT:
    """Assumes 1:1 options and NSRTs, see note in file docstring."""
    nsrt = _param_option_to_nsrt(option.parent, nsrts)
    return nsrt.ground(option.objects)


def _wrap_sampler(base_sampler: NSRTSampler, score_fn: _ScoreFn) -> NSRTSampler:
    """Create a wrapped sampler that uses a score function to select among candidates from a base sampler."""

    def _sample(state: State, goal: Set[GroundAtom],
               rng: np.random.Generator, objects: Sequence[Object]) -> Array:
        samples = [
            base_sampler(state, goal, rng, objects)
            for _ in range(CFG.active_sampler_learning_num_samples)
        ]
        scores = score_fn(state, objects, samples)
        # For now, just pick the best scoring sample.
        idx = np.argmax(scores)
        return samples[idx]

    return _sample


def _classifier_to_score_fn(classifier: MLPBinaryClassifier, nsrt: NSRT) -> _ScoreFn:
    """Use predict_proba() to produce scores."""

    def _score_fn(state: State, objects: Sequence[Object], param_lst: List[Array]) -> List[float]:
        x_lst: List[Any] = [1.0]  # start with bias term
        sub = dict(zip(nsrt.parameters, objects))
        for var in nsrt.parameters:
            x_lst.extend(state[sub[var]])
        assert not CFG.sampler_learning_use_goals
        x = np.array(x_lst)
        scores = [classifier.predict_proba(np.r_[x, p]) for p in param_lst]
        return scores

    return _score_fn
