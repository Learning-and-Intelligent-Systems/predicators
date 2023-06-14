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
        --bilevel_plan_without_sim True \
        --offline_data_bilevel_plan_without_sim False \
        --explorer random_nsrts \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 4 \
        --bumpy_cover_num_bumps 2 \
        --bumpy_cover_spaces_per_bump 1


Bumpy cover with shifted targets:
    python predicators/main.py --approach active_sampler_learning \
        --env bumpy_cover \
        --seed 0 \
        --strips_learner oracle \
        --sampler_learner oracle \
        --bilevel_plan_without_sim True \
        --offline_data_bilevel_plan_without_sim False \
        --explorer random_nsrts \
        --max_initial_demos 1 \
        --num_train_tasks 1000 \
        --num_test_tasks 10 \
        --max_num_steps_interaction_request 4 \
        --bumpy_cover_num_bumps 2 \
        --bumpy_cover_spaces_per_bump 1 \
        --bumpy_cover_right_targets True
"""
from __future__ import annotations

import abc
import logging
from typing import Any, Callable, Dict, List, Optional, Sequence, Set, Tuple

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.ml_models import MLPBinaryClassifier, MLPRegressor
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LowLevelTrajectory, \
    NSRTSampler, Object, ParameterizedOption, Predicate, Segment, State, \
    Task, Type, _GroundNSRT, _Option

# Dataset for sampler learning: includes (s, option, s', label) per param opt.
_OptionSamplerDataset = List[Tuple[State, _Option, State, Any]]
_SamplerDataset = Dict[ParameterizedOption, _OptionSamplerDataset]
_ScoreFn = Callable[[State, Sequence[Object], List[Array]], List[float]]


class ActiveSamplerLearningApproach(OnlineNSRTLearningApproach):
    """Performs active sampler learning."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)

        # The current implementation assumes that NSRTs are not changing.
        assert CFG.strips_learner == "oracle"
        # The base sampler should also be unchanging and from the oracle.
        assert CFG.sampler_learner == "oracle"

        self._sampler_data: _SamplerDataset = {}
        self._last_seen_segment_traj_idx = -1

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
        new_trajs = self._segmented_trajs[self._last_seen_segment_traj_idx:]
        for segmented_traj in new_trajs:
            self._last_seen_segment_traj_idx += 1
            just_made_incorrect_pick = False
            for segment in segmented_traj:
                s = segment.states[0]
                o = segment.get_option()
                ns = segment.states[-1]
                success = self._check_option_success(o, segment)
                assert CFG.env == "bumpy_cover"
                if CFG.active_sampler_learning_use_teacher:
                    if CFG.bumpy_cover_right_targets:
                        # In bumpy cover with the 'bumpy_cover_right_targets'
                        # flag set, picking from the left is bad and can
                        # potentially lead to failures to place. We will
                        # say that picking from left fails, and also skip
                        # adding all the subsequent 'Place' actions.
                        if o.name == "Pick":
                            block = o.objects[0]
                            block_center = s.get(block, "pose")
                            block_width = s.get(block, "width")
                            if block_center - block_width < o.params[
                                    0] < block_center:
                                success = False
                                just_made_incorrect_pick = True
                            elif success:
                                just_made_incorrect_pick = False
                        if o.name == "Place" and just_made_incorrect_pick:
                            continue

                if CFG.active_sampler_learning_model == "myopic_classifier":
                    label: Any = success
                else:
                    assert CFG.active_sampler_learning_model == "fitted_q"
                    label = 0.0 if success else -1.0

                # Store transition per ParameterizedOption. Don't store by
                # NSRT because those change as we re-learn.
                if o.parent not in self._sampler_data:
                    self._sampler_data[o.parent] = []
                self._sampler_data[o.parent].append((s, o, ns, label))

    def _check_option_success(self, option: _Option, segment: Segment) -> bool:
        ground_nsrt = _option_to_ground_nsrt(option, self._nsrts)
        return ground_nsrt.add_effects.issubset(
            segment.final_atoms) and not ground_nsrt.delete_effects.issubset(
                segment.final_atoms)

    def _learn_wrapped_samplers(self,
                                online_learning_cycle: Optional[int]) -> None:
        """Update the NSRTs in place."""
        if CFG.active_sampler_learning_model == "myopic_classifier":
            learner: _WrappedSamplerLearner = _ClassifierWrappedSamplerLearner(
                self._get_current_nsrts(), self._get_current_predicates(),
                online_learning_cycle)
        else:
            assert CFG.active_sampler_learning_model == "fitted_q"
            learner = _FittedQWrappedSamplerLearner(
                self._get_current_nsrts(), self._get_current_predicates(),
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
        # Re-save the NSRTs now that we've updated them.
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.NSRTs", "wb") as f:
            pkl.dump(self._nsrts, f)


class _WrappedSamplerLearner(abc.ABC):
    """A base class for learning wrapped samplers."""

    def __init__(self, nsrts: Set[NSRT], predicates: Set[Predicate],
                 online_learning_cycle: Optional[int]) -> None:
        self._nsrts = nsrts
        self._predicates = predicates
        self._online_learning_cycle = online_learning_cycle
        self._rng = np.random.default_rng(CFG.seed)
        self._learned_samplers: Optional[Dict[NSRT, NSRTSampler]] = None

    def learn(self, data: _SamplerDataset) -> None:
        """Fit all of the samplers."""
        new_samplers: Dict[NSRT, NSRTSampler] = {}
        for param_opt, nsrt_data in data.items():
            nsrt = _param_option_to_nsrt(param_opt, self._nsrts)
            logging.info(f"Fitting wrapped sampler for {nsrt.name}...")
            new_samplers[nsrt] = self._learn_nsrt_sampler(nsrt_data, nsrt)
        self._learned_samplers = new_samplers

    def get_samplers(self) -> Dict[NSRT, NSRTSampler]:
        """Expose the fitted samplers, organized by NSRTs."""
        assert self._learned_samplers is not None
        return self._learned_samplers

    @abc.abstractmethod
    def _learn_nsrt_sampler(self, nsrt_data: _OptionSamplerDataset,
                            nsrt: NSRT) -> NSRTSampler:
        """Learn the sampler for a single NSRT and return it."""


class _ClassifierWrappedSamplerLearner(_WrappedSamplerLearner):
    """Using boolean class labels on transitions, learn a classifier, and then
    use the probability of predicting True to select parameters."""

    def _learn_nsrt_sampler(self, nsrt_data: _OptionSamplerDataset,
                            nsrt: NSRT) -> NSRTSampler:
        X_classifier: List[List[Array]] = []
        y_classifier: List[int] = []
        for state, option, _, label in nsrt_data:
            objects = option.objects
            params = option.params
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
        save_path = f"{approach_save_path}_{nsrt.name}_" + \
            f"{self._online_learning_cycle}.sampler_classifier"
        with open(save_path, "wb") as f:
            pkl.dump(classifier, f)
        logging.info(f"Saved sampler classifier to {save_path}.")

        # Easiest way to access the base sampler.
        base_sampler = nsrt._sampler  # pylint: disable=protected-access
        score_fn = _classifier_to_score_fn(classifier, nsrt)
        wrapped_sampler = _wrap_sampler(base_sampler, score_fn)

        return wrapped_sampler


class _FittedQWrappedSamplerLearner(_WrappedSamplerLearner):
    """Perform fitted Q iteration to learn all samplers from batch data."""

    def __init__(self, nsrts: Set[NSRT], predicates: Set[Predicate],
                 online_learning_cycle: Optional[int]) -> None:
        super().__init__(nsrts, predicates, online_learning_cycle)
        self._nsrt_score_fns: Optional[Dict[NSRT, _ScoreFn]] = None
        self._next_nsrt_score_fns: Dict[NSRT, _ScoreFn] = {}

    def learn(self, data: _SamplerDataset) -> None:
        # Override parent so that the learning loop is run for multiple iters.
        for it in range(CFG.active_sampler_learning_fitted_q_iters):
            logging.info(f"Starting fitted Q learning iter {it}")
            # Run one iteration of learning. Calls _learn_nsrt_sampler().
            super().learn(data)
            # Update the score functions now that all children are processed.
            self._nsrt_score_fns = self._next_nsrt_score_fns

    def _learn_nsrt_sampler(self, nsrt_data: _OptionSamplerDataset,
                            nsrt: NSRT) -> NSRTSampler:
        # Build targets.
        gamma = CFG.active_sampler_learning_score_gamma
        num_a_samp = CFG.active_sampler_learning_num_next_option_samples
        targets: List[float] = []
        for _, _, ns, r in nsrt_data:
            # Sample actions to estimate Q in infinite action space.
            next_as = self._sample_options_from_state(ns, num=num_a_samp)
            next_q = max(self._predict(ns, na) for na in next_as)
            # NOTE: there is no terminal state because we're in a lifelong
            # (reset-free) setup.
            target = r + gamma * next_q
            targets.append(target)
        # Build regressor dataset.
        regressor_data = [(s, a, ns, target)
                          for (s, a, ns, _), target in zip(nsrt_data, targets)]
        # Run regression.
        regressor = self._fit_regressor(regressor_data)
        # Save the sampler regressor for external analysis.
        approach_save_path = utils.get_approach_save_path_str()
        save_path = f"{approach_save_path}_{nsrt.name}_" + \
            f"{self._online_learning_cycle}.sampler_regressor"
        with open(save_path, "wb") as f:
            pkl.dump(regressor, f)
        logging.info(f"Saved sampler regressor to {save_path}.")
        # Wrap and return sampler.
        base_sampler = nsrt._sampler  # pylint: disable=protected-access
        score_fn = _regressor_to_score_fn(regressor, nsrt)
        # Save the score function for use in later target computation.
        self._next_nsrt_score_fns[nsrt] = score_fn
        wrapped_sampler = _wrap_sampler(base_sampler, score_fn)
        return wrapped_sampler

    def _predict(self, state: State, option: _Option) -> float:
        """Predict Q(s, a)."""
        if self._nsrt_score_fns is None:
            return 0.0  # initialize to 0.0
        ground_nsrt = _option_to_ground_nsrt(option, self._nsrts)
        score_fn = self._nsrt_score_fns[ground_nsrt.parent]
        return score_fn(state, ground_nsrt.objects, [option.params])[0]

    def _sample_options_from_state(self,
                                   state: State,
                                   num: int = 1) -> List[_Option]:
        """Use NSRTs to sample options in the current state."""
        # Create all applicable ground NSRTs.
        ground_nsrts: List[_GroundNSRT] = []
        for nsrt in sorted(self._nsrts):
            ground_nsrts.extend(utils.all_ground_nsrts(nsrt, list(state)))

        sampled_options: List[_Option] = []
        for _ in range(num):
            # Sample an applicable NSRT.
            ground_nsrt = utils.sample_applicable_ground_nsrt(
                state, ground_nsrts, self._predicates, self._rng)
            assert ground_nsrt is not None
            assert all(a.holds for a in ground_nsrt.preconditions)

            # Sample an option.
            option = ground_nsrt.sample_option(
                state,
                goal=set(),  # goal not used
                rng=self._rng)
            assert option.initiable(state)
            sampled_options.append(option)

        return sampled_options

    def _fit_regressor(self, nsrt_data: _OptionSamplerDataset) -> MLPRegressor:
        X_regressor: List[List[Array]] = []
        y_regressor: List[Array] = []
        for state, option, _, target in nsrt_data:
            objects = option.objects
            params = option.params
            # input is state features and option parameters
            X_regressor.append([np.array(1.0)])  # start with bias term
            for obj in objects:
                X_regressor[-1].extend(state[obj])
            X_regressor[-1].extend(params)
            assert not CFG.sampler_learning_use_goals
            y_regressor.append(np.array([target]))
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
            train_print_every=CFG.pytorch_train_print_every,
            n_iter_no_change=CFG.active_sampler_learning_n_iter_no_change)
        regressor.fit(X_arr_regressor, y_arr_regressor)
        return regressor


# Helper functions.
def _param_option_to_nsrt(param_option: ParameterizedOption,
                          nsrts: Set[NSRT]) -> NSRT:
    """Assumes 1:1 options and NSRTs, see note in file docstring."""
    nsrt_matches = [n for n in nsrts if n.option == param_option]
    assert len(nsrt_matches) == 1
    nsrt = nsrt_matches[0]
    return nsrt


def _option_to_ground_nsrt(option: _Option, nsrts: Set[NSRT]) -> _GroundNSRT:
    """Assumes 1:1 options and NSRTs, see note in file docstring."""
    nsrt = _param_option_to_nsrt(option.parent, nsrts)
    return nsrt.ground(option.objects)


def _wrap_sampler(
    base_sampler: NSRTSampler,
    score_fn: _ScoreFn,
) -> NSRTSampler:
    """Create a wrapped sampler that uses a score function to select among
    candidates from a base sampler."""

    def _sample(state: State, goal: Set[GroundAtom], rng: np.random.Generator,
                objects: Sequence[Object]) -> Array:
        samples = [
            base_sampler(state, goal, rng, objects)
            for _ in range(CFG.active_sampler_learning_num_samples)
        ]
        scores = score_fn(state, objects, samples)
        # For now, just pick the best scoring sample.
        idx = np.argmax(scores)
        return samples[idx]

    return _sample


def _vector_score_fn_to_score_fn(vector_fn: Callable[[Array], float],
                                 nsrt: NSRT) -> _ScoreFn:
    """Helper for _classifier_to_score_fn() and _regressor_to_score_fn()."""

    def _score_fn(state: State, objects: Sequence[Object],
                  param_lst: List[Array]) -> List[float]:
        x_lst: List[Any] = [1.0]  # start with bias term
        sub = dict(zip(nsrt.parameters, objects))
        for var in nsrt.parameters:
            x_lst.extend(state[sub[var]])
        assert not CFG.sampler_learning_use_goals
        x = np.array(x_lst)
        scores = [vector_fn(np.r_[x, p]) for p in param_lst]
        return scores

    return _score_fn


def _classifier_to_score_fn(classifier: MLPBinaryClassifier,
                            nsrt: NSRT) -> _ScoreFn:
    return _vector_score_fn_to_score_fn(classifier.predict_proba, nsrt)


def _regressor_to_score_fn(regressor: MLPRegressor, nsrt: NSRT) -> _ScoreFn:
    fn = lambda v: regressor.predict(v)[0]
    return _vector_score_fn_to_score_fn(fn, nsrt)
