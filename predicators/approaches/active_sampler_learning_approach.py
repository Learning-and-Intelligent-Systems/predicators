"""An approach that performs active sampler learning.

The current implementation assumes for convenience that NSRTs and options are
1:1 and share the same parameters (like a PDDL environment). It is
straightforward conceptually to remove this assumption, because the approach
uses its own NSRTs to select options, but it is difficult implementation-wise,
so we're punting for now.

See scripts/configs/active_sampler_learning.yaml for examples.
"""
from __future__ import annotations

import abc
import logging
from collections import defaultdict
from typing import Any, Callable, DefaultDict, Dict, List, Optional, \
    Sequence, Set, Tuple
from typing import Type as TypingType

import dill as pkl
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches.online_nsrt_learning_approach import \
    OnlineNSRTLearningApproach
from predicators.competence_models import SkillCompetenceModel
from predicators.explorers import BaseExplorer, create_explorer
from predicators.ml_models import BinaryClassifier, BinaryClassifierEnsemble, \
    KNeighborsClassifier, MLPBinaryClassifier, MLPRegressor
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LowLevelTrajectory, \
    Metrics, NSRTSampler, NSRTSamplerWithEpsilonIndicator, Object, \
    ParameterizedOption, Predicate, Segment, State, Task, Type, _GroundNSRT, \
    _GroundSTRIPSOperator, _Option

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
        # Maps used ground operators to all historical outcomes (whether they
        # successfully reached their effects or not). Updated in-place by the
        # explorer when CFG.explorer is active_sampler_explorer.
        self._ground_op_hist: Dict[_GroundSTRIPSOperator, List[bool]] = {}
        self._competence_models: Dict[_GroundSTRIPSOperator,
                                      SkillCompetenceModel] = {}
        self._last_seen_segment_traj_idx = -1

        # For certain methods, we may want the NSRTs used for exploration to
        # differ from those used for execution (they will differ precisely
        # in their sampler). Thus, we will keep around a separate mapping from
        # NSRTs to samplers to be used at exploration time.
        self._nsrt_to_explorer_sampler: Dict[
            NSRT, NSRTSamplerWithEpsilonIndicator] = {}

        # Record what train tasks have been seen during exploration so far.
        self._seen_train_task_idxs: Set[int] = set()

        # Set the default cost for skills.
        alpha, beta = CFG.skill_competence_default_alpha_beta
        c = utils.beta_bernoulli_posterior([], alpha=alpha, beta=beta).mean()
        self._default_cost = -np.log(c)

    @classmethod
    def get_name(cls) -> str:
        return "active_sampler_learning"

    def _run_task_plan(
        self, task: Task, nsrts: Set[NSRT], preds: Set[Predicate],
        timeout: float, seed: int, **kwargs: Any
    ) -> Tuple[List[_GroundNSRT], List[Set[GroundAtom]], Metrics]:
        # Add ground operator competence for competence-aware planning.
        ground_op_costs = {
            o: -np.log(m.get_current_competence())
            for o, m in self._competence_models.items()
        }
        return super()._run_task_plan(task,
                                      nsrts,
                                      preds,
                                      timeout,
                                      seed,
                                      ground_op_costs=ground_op_costs,
                                      default_cost=self._default_cost,
                                      **kwargs)

    def _create_explorer(self) -> BaseExplorer:
        # Geometrically increase the length of exploration.
        b = CFG.active_sampler_learning_explore_length_base
        max_steps = b**(1 + self._online_learning_cycle)
        preds = self._get_current_predicates()
        # Pursue the task goal during exploration periodically.
        n = CFG.active_sampler_learning_explore_pursue_goal_interval
        pursue_task_goal_first = False
        if self._online_learning_cycle < \
            CFG.active_sampler_learning_init_cycles_to_pursue_goal or (
                self._online_learning_cycle % n == 0):
            pursue_task_goal_first = True
        explorer = create_explorer(
            CFG.explorer,
            preds,
            self._initial_options,
            self._types,
            self._action_space,
            self._train_tasks,
            self._get_current_nsrts(),
            self._option_model,
            ground_op_hist=self._ground_op_hist,
            competence_models=self._competence_models,
            max_steps_before_termination=max_steps,
            nsrt_to_explorer_sampler=self._nsrt_to_explorer_sampler,
            seen_train_task_idxs=self._seen_train_task_idxs,
            pursue_task_goal_first=pursue_task_goal_first)
        return explorer

    def load(self, online_learning_cycle: Optional[int]) -> None:
        super().load(online_learning_cycle)
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.DATA", "rb") as f:
            save_dict = pkl.load(f)
        self._dataset = save_dict["dataset"]
        self._sampler_data = save_dict["sampler_data"]
        self._ground_op_hist = save_dict["ground_op_hist"]
        self._competence_models = save_dict["competence_models"]
        self._last_seen_segment_traj_idx = save_dict[
            "last_seen_segment_traj_idx"]
        self._nsrt_to_explorer_sampler = save_dict["nsrt_to_explorer_sampler"]
        self._seen_train_task_idxs = save_dict["seen_train_task_idxs"]
        self._train_tasks = save_dict["train_tasks"]
        self._online_learning_cycle = 0 if online_learning_cycle is None \
            else online_learning_cycle + 1

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
        # Advance the competence models.
        for ground_op, competence_model in self._competence_models.items():
            competence_model.advance_cycle()
            # Log the current competences.
            competence = competence_model.get_current_competence()
            logging.info("Current competence for "
                         f"{ground_op.name}{ground_op.objects}: {competence}")
        # Sanity check that the ground op histories and sampler data are sync.
        op_to_num_ground_op_hist: Dict[str, int] = {
            n.name: 0
            for n in self._sampler_data
        }
        for ground_op, examples in self._ground_op_hist.items():
            num = len(examples)
            name = ground_op.parent.name
            assert name in op_to_num_ground_op_hist
            op_to_num_ground_op_hist[name] += num
        for op, op_sampler_data in self._sampler_data.items():
            if CFG.explorer == "active_sampler":
                # The only case where there should be more sampler data than
                # ground op hist is if we started out with a nontrivial
                # dataset. That dataset is not included in the ground op hist.
                num_ground_op = op_to_num_ground_op_hist[op.name]
                num_sampler = len(op_sampler_data)
                assert num_ground_op == num_sampler or \
                    (num_sampler > num_ground_op and CFG.max_initial_demos > 0)
        # Save the things we need other than the NSRTs, which were already
        # saved in the above call to self._learn_nsrts()
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.DATA", "wb") as f:
            pkl.dump(
                {
                    "dataset": self._dataset,
                    "sampler_data": self._sampler_data,
                    "ground_op_hist": self._ground_op_hist,
                    "competence_models": self._competence_models,
                    "last_seen_segment_traj_idx":
                    self._last_seen_segment_traj_idx,
                    "nsrt_to_explorer_sampler": self._nsrt_to_explorer_sampler,
                    "seen_train_task_idxs": self._seen_train_task_idxs,
                    # We need to save train tasks because they get modified
                    # in the explorer. The original sin is that tasks are
                    # generated before reset with default init states, which
                    # are subsequently overwritten after reset is called.
                    "train_tasks": self._train_tasks,
                },
                f)

    def _update_sampler_data(self) -> None:
        start_idx = self._last_seen_segment_traj_idx + 1
        new_trajs = self._segmented_trajs[start_idx:]
        ground_op_to_num_data: DefaultDict[_GroundSTRIPSOperator,
                                           int] = defaultdict(int)
        for segmented_traj in new_trajs:
            self._last_seen_segment_traj_idx += 1
            just_made_incorrect_pick = False
            for segment in segmented_traj:
                s = segment.states[0]
                o = segment.get_option()
                ns = segment.states[-1]
                success = self._check_option_success(o, segment)
                if CFG.active_sampler_learning_use_teacher:
                    assert CFG.env in ("bumpy_cover", "regional_bumpy_cover")
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
                            continue  # pragma: no cover

                if CFG.active_sampler_learning_model in [
                        "myopic_classifier_mlp", "myopic_classifier_ensemble",
                        "myopic_classifier_knn"
                ]:
                    label: Any = success
                else:
                    assert CFG.active_sampler_learning_model == "fitted_q"
                    label = 0.0 if success else -1.0

                # Store transition per ParameterizedOption. Don't store by
                # NSRT because those change as we re-learn.
                if o.parent not in self._sampler_data:
                    self._sampler_data[o.parent] = []
                self._sampler_data[o.parent].append((s, o, ns, label))
                ground_nsrt = utils.option_to_ground_nsrt(o, self._nsrts)
                ground_op_to_num_data[ground_nsrt.op] += 1
        # Save competence models.
        for ground_op, model in self._competence_models.items():
            approach_save_path = utils.get_approach_save_path_str()
            save_path = "_".join([
                approach_save_path, f"{ground_op.name}{ground_op.objects}",
                f"{self._online_learning_cycle}.competence"
            ])
            with open(save_path, "wb") as f:
                pkl.dump(model, f)
            logging.info(f"Saved competence model to {save_path}.")

    def _check_option_success(self, option: _Option, segment: Segment) -> bool:
        ground_nsrt = utils.option_to_ground_nsrt(option, self._nsrts)
        # Only the add effects are checked to determine option success. This
        # is fine for our cover environments, but will probably break in
        # other environments (for instance, if we accidentally delete
        # atoms that are actually necessary for a future operator in the
        # plan). The right thing to do here is check the necessary atoms,
        # which we will do in a forthcoming PR.
        return ground_nsrt.add_effects.issubset(segment.final_atoms)

    def _learn_wrapped_samplers(self,
                                online_learning_cycle: Optional[int]) -> None:
        """Update the NSRTs in place."""
        if CFG.active_sampler_learning_model in [
                "myopic_classifier_mlp", "myopic_classifier_knn"
        ]:
            learner_cls: TypingType[_WrappedSamplerLearner] = \
                _ClassifierWrappedSamplerLearner
        elif CFG.active_sampler_learning_model == "myopic_classifier_ensemble":
            learner_cls = _ClassifierEnsembleWrappedSamplerLearner
        else:
            assert CFG.active_sampler_learning_model == "fitted_q"
            learner_cls = _FittedQWrappedSamplerLearner
        if CFG.active_sampler_learning_object_specific_samplers:
            learner: _WrappedSamplerLearner = \
                _ObjectSpecificSamplerLearningWrapper(
                learner_cls, self._get_current_nsrts(),
                self._get_current_predicates(), online_learning_cycle)
        else:
            learner = learner_cls(self._get_current_nsrts(),
                                  self._get_current_predicates(),
                                  online_learning_cycle)
        # Fit with the current data.
        learner.learn(self._sampler_data)
        wrapped_samplers = learner.get_samplers()
        # Update the NSRTs.
        new_test_nsrts: Set[NSRT] = set()
        self._nsrt_to_explorer_sampler.clear()
        for nsrt, (test_sampler, explore_sampler) in wrapped_samplers.items():
            # Create new test NSRT.
            new_test_nsrt = NSRT(nsrt.name, nsrt.parameters,
                                 nsrt.preconditions, nsrt.add_effects,
                                 nsrt.delete_effects, nsrt.ignore_effects,
                                 nsrt.option, nsrt.option_vars, test_sampler)
            new_test_nsrts.add(new_test_nsrt)
            # Update the dictionary mapping NSRTs to exploration samplers.
            self._nsrt_to_explorer_sampler[nsrt] = explore_sampler
        # Special case, especially on the first iteration: if there was no
        # data for the sampler, then we didn't learn a wrapped sampler, so
        # we should just use the original NSRT.
        new_nsrt_options = {n.option for n in new_test_nsrts}
        for old_nsrt in self._nsrts:
            if old_nsrt.option not in new_nsrt_options:
                new_test_nsrts.add(old_nsrt)
                # Since we don't have a learned score function, just make
                # a lambda function that returns the same score (1.0)
                # for every input that gets passed in.
                self._nsrt_to_explorer_sampler[
                    old_nsrt] = _wrap_sampler_exploration(
                        old_nsrt._sampler,  # pylint: disable=protected-access
                        lambda o, _, params: [1.0] * len(params),
                        "greedy")
        self._nsrts = new_test_nsrts
        # Re-save the NSRTs now that we've updated them.
        save_path = utils.get_approach_save_path_str()
        with open(f"{save_path}_{online_learning_cycle}.NSRTs", "wb") as f:
            pkl.dump(self._nsrts, f)


class _WrappedSamplerLearner(abc.ABC):
    """A base class for learning wrapped samplers."""

    def __init__(self,
                 nsrts: Set[NSRT],
                 predicates: Set[Predicate],
                 online_learning_cycle: Optional[int],
                 save_id_suffix: Optional[str] = None) -> None:
        self._nsrts = nsrts
        self._predicates = predicates
        self._online_learning_cycle = online_learning_cycle
        self._save_id_suffix = save_id_suffix
        self._rng = np.random.default_rng(CFG.seed)
        # We keep track of two samplers per NSRT: one to use at test time
        # and another to use during exploration/play time.
        self._learned_samplers: Optional[Dict[NSRT, Tuple[
            NSRTSampler, NSRTSamplerWithEpsilonIndicator]]] = None

    def _get_save_id(self, nsrt_name: str) -> str:
        """A unique ID used for saving data."""
        approach_path = utils.get_approach_save_path_str()
        save_id = f"{approach_path}_{nsrt_name}_{self._online_learning_cycle}"
        if self._save_id_suffix is not None:
            save_id = f"{save_id}_{self._save_id_suffix}"
        return save_id

    def learn(self, data: _SamplerDataset) -> None:
        """Fit all of the samplers."""
        new_samplers: Dict[NSRT, Tuple[NSRTSampler,
                                       NSRTSamplerWithEpsilonIndicator]] = {}
        for param_opt, nsrt_data in data.items():
            nsrt = utils.param_option_to_nsrt(param_opt, self._nsrts)
            logging.info(f"Fitting wrapped sampler for {nsrt.name}...")
            new_samplers[nsrt] = self._learn_nsrt_sampler(nsrt_data, nsrt)
        self._learned_samplers = new_samplers

    def get_samplers(
        self
    ) -> Dict[NSRT, Tuple[NSRTSampler, NSRTSamplerWithEpsilonIndicator]]:
        """Expose the fitted samplers, organized by NSRTs."""
        assert self._learned_samplers is not None
        return self._learned_samplers

    @abc.abstractmethod
    def _learn_nsrt_sampler(
            self, nsrt_data: _OptionSamplerDataset,
            nsrt: NSRT) -> Tuple[NSRTSampler, NSRTSamplerWithEpsilonIndicator]:
        """Learn the new test-time and exploration samplers for a single NSRT
        and return them."""


class _ClassifierWrappedSamplerLearner(_WrappedSamplerLearner):
    """Using boolean class labels on transitions, learn a classifier, and then
    use the probability of predicting True to select parameters."""

    def _learn_nsrt_sampler(
            self, nsrt_data: _OptionSamplerDataset,
            nsrt: NSRT) -> Tuple[NSRTSampler, NSRTSamplerWithEpsilonIndicator]:
        X_classifier: List[Array] = []
        y_classifier: List[int] = []
        for state, option, _, label in nsrt_data:
            objects = option.objects
            params = option.params
            x_arr = utils.construct_active_sampler_input(
                state, objects, params, option.parent)
            X_classifier.append(x_arr)
            y_classifier.append(label)
        X_arr_classifier = np.array(X_classifier)
        # output is binary signal
        y_arr_classifier = np.array(y_classifier)
        if CFG.active_sampler_learning_model.endswith("mlp"):
            classifier: BinaryClassifier = MLPBinaryClassifier(
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
        else:
            assert CFG.active_sampler_learning_model.endswith("knn")
            n_neighbors = min(len(X_arr_classifier),
                              CFG.active_sampler_learning_knn_neighbors)
            classifier = KNeighborsClassifier(seed=CFG.seed,
                                              n_neighbors=n_neighbors)
        classifier.fit(X_arr_classifier, y_arr_classifier)

        # Save the sampler classifier for external analysis.
        save_id = self._get_save_id(nsrt.name)
        save_path = f"{save_id}.sampler_classifier"
        with open(save_path, "wb") as f:
            pkl.dump(classifier, f)
        logging.info(f"Saved sampler classifier to {save_path}.")
        save_path = f"{save_id}.sampler_classifier_data"
        with open(save_path, "wb") as f:
            pkl.dump((X_arr_classifier, y_arr_classifier), f)
        logging.info(f"Saved sampler classifier data to {save_path}.")

        # Easiest way to access the base sampler.
        base_sampler = nsrt._sampler  # pylint: disable=protected-access
        score_fn = _classifier_to_score_fn(classifier, nsrt)
        wrapped_sampler_test = _wrap_sampler_test(base_sampler, score_fn)
        wrapped_sampler_exploration = _wrap_sampler_exploration(
            base_sampler,
            score_fn,
            strategy=CFG.active_sampler_learning_exploration_sample_strategy)
        return (wrapped_sampler_test, wrapped_sampler_exploration)


class _ClassifierEnsembleWrappedSamplerLearner(_WrappedSamplerLearner):
    """Using boolean class labels on transitions, learn an ensemble of
    classifiers, and then use the entropy among the predictions, as well as the
    probability of predicting True to select parameters."""

    def _learn_nsrt_sampler(
            self, nsrt_data: _OptionSamplerDataset,
            nsrt: NSRT) -> Tuple[NSRTSampler, NSRTSamplerWithEpsilonIndicator]:
        X_classifier: List[Array] = []
        y_classifier: List[int] = []
        for state, option, _, label in nsrt_data:
            objects = option.objects
            params = option.params
            x_arr = utils.construct_active_sampler_input(
                state, objects, params, option.parent)
            X_classifier.append(x_arr)
            y_classifier.append(label)
        X_arr_classifier = np.array(X_classifier)
        # output is binary signal
        y_arr_classifier = np.array(y_classifier)
        classifier = BinaryClassifierEnsemble(
            seed=CFG.seed,
            ensemble_size=CFG.active_sampler_learning_num_ensemble_members,
            member_cls=MLPBinaryClassifier,
            balance_data=CFG.mlp_classifier_balance_data,
            max_train_iters=CFG.sampler_mlp_classifier_max_itr,
            learning_rate=CFG.learning_rate,
            n_iter_no_change=CFG.mlp_classifier_n_iter_no_change,
            hid_sizes=CFG.mlp_classifier_hid_sizes,
            n_reinitialize_tries=CFG.
            sampler_mlp_classifier_n_reinitialize_tries,
            weight_init=CFG.predicate_mlp_classifier_init,
            weight_decay=CFG.weight_decay,
            use_torch_gpu=CFG.use_torch_gpu,
            train_print_every=CFG.pytorch_train_print_every)
        classifier.fit(X_arr_classifier, y_arr_classifier)

        # Save the sampler classifier for external analysis.
        save_id = self._get_save_id(nsrt.name)
        save_path = f"{save_id}.sampler_classifier"
        with open(save_path, "wb") as f:
            pkl.dump(classifier, f)
        logging.info(f"Saved sampler classifier to {save_path}.")

        # Easiest way to access the base sampler.
        base_sampler = nsrt._sampler  # pylint: disable=protected-access
        test_score_fn = _classifier_ensemble_to_score_fn(classifier,
                                                         nsrt,
                                                         test_time=True)
        wrapped_sampler_test = _wrap_sampler_test(base_sampler, test_score_fn)
        explore_score_fn = _classifier_ensemble_to_score_fn(classifier,
                                                            nsrt,
                                                            test_time=False)
        wrapped_sampler_exploration = _wrap_sampler_exploration(
            base_sampler,
            explore_score_fn,
            strategy=CFG.active_sampler_learning_exploration_sample_strategy)

        return (wrapped_sampler_test, wrapped_sampler_exploration)


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

    def _learn_nsrt_sampler(
            self, nsrt_data: _OptionSamplerDataset,
            nsrt: NSRT) -> Tuple[NSRTSampler, NSRTSamplerWithEpsilonIndicator]:
        # Build targets.
        gamma = CFG.active_sampler_learning_score_gamma
        num_a_samp = CFG.active_sampler_learning_num_lookahead_samples
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
        save_id = self._get_save_id(nsrt.name)
        save_path = f"{save_id}.sampler_regressor"
        with open(save_path, "wb") as f:
            pkl.dump(regressor, f)
        logging.info(f"Saved sampler regressor to {save_path}.")
        # Wrap and return sampler.
        base_sampler = nsrt._sampler  # pylint: disable=protected-access
        score_fn = _regressor_to_score_fn(regressor, nsrt)
        # Save the score function for use in later target computation.
        self._next_nsrt_score_fns[nsrt] = score_fn
        wrapped_sampler_test = _wrap_sampler_test(base_sampler, score_fn)
        wrapped_sampler_exploration = _wrap_sampler_exploration(
            base_sampler,
            score_fn,
            strategy=CFG.active_sampler_learning_exploration_sample_strategy)
        return (wrapped_sampler_test, wrapped_sampler_exploration)

    def _predict(self, state: State, option: _Option) -> float:
        """Predict Q(s, a)."""
        if self._nsrt_score_fns is None:
            return 0.0  # initialize to 0.0
        ground_nsrt = utils.option_to_ground_nsrt(option, self._nsrts)
        # Special case: we haven't seen any data for the parent NSRT, so we
        # haven't learned a score function for it.
        if ground_nsrt.parent not in self._nsrt_score_fns:  # pragma: no cover
            return 0.0
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
        X_regressor: List[Array] = []
        y_regressor: List[Array] = []
        for state, option, _, target in nsrt_data:
            objects = option.objects
            params = option.params
            x_arr = utils.construct_active_sampler_input(
                state, objects, params, option.parent)
            X_regressor.append(x_arr)
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


class _ObjectSpecificSamplerLearningWrapper(_WrappedSamplerLearner):
    """Wrapper for learning multiple object-specific samplers for each NSRT."""

    def __init__(self,
                 base_sampler_learner_cls: TypingType[_WrappedSamplerLearner],
                 nsrts: Set[NSRT], predicates: Set[Predicate],
                 online_learning_cycle: Optional[int]) -> None:
        super().__init__(nsrts, predicates, online_learning_cycle)
        self._base_sampler_learner_cls = base_sampler_learner_cls

    def _learn_nsrt_sampler(
            self, nsrt_data: _OptionSamplerDataset,
            nsrt: NSRT) -> Tuple[NSRTSampler, NSRTSamplerWithEpsilonIndicator]:
        """Learn the new test-time and exploration samplers for a single NSRT
        and return them."""
        # Organize data by groundings.
        grounding_to_data: Dict[Tuple[Object, ...], _OptionSamplerDataset] = {}
        for datum in nsrt_data:
            option = datum[1]
            objects = tuple(option.objects)
            if objects not in grounding_to_data:
                grounding_to_data[objects] = []
            grounding_to_data[objects].append(datum)
        # Call the base class once per grounding.
        grounding_to_sampler: Dict[Tuple[Object, ...], NSRTSampler] = {}
        grounding_to_sampler_with_eps: Dict[Tuple[
            Object, ...], NSRTSamplerWithEpsilonIndicator] = {}
        for grounding, data in grounding_to_data.items():
            base_sampler_learner = self._base_sampler_learner_cls(
                self._nsrts,
                self._predicates,
                self._online_learning_cycle,
                save_id_suffix=str(grounding))
            sampler_dataset = {nsrt.option: data}
            logging.info(f"Fitting object specific sampler: {grounding}...")
            base_sampler_learner.learn(sampler_dataset)
            samplers = base_sampler_learner.get_samplers()
            assert len(samplers) == 1
            sampler, sampler_with_eps = samplers[nsrt]
            grounding_to_sampler[grounding] = sampler
            grounding_to_sampler_with_eps[grounding] = sampler_with_eps
        # Create wrapped samplers and samplers with epsilon indicator.
        base_sampler = nsrt._sampler  # pylint: disable=protected-access
        nsrt_sampler = _wrap_object_specific_samplers(grounding_to_sampler,
                                                      base_sampler)
        nsrt_sampler_with_eps = _wrap_object_specific_samplers_with_epsilon(
            grounding_to_sampler_with_eps, base_sampler)
        return nsrt_sampler, nsrt_sampler_with_eps


# Helper functions.
def _wrap_sampler_test(base_sampler: NSRTSampler,
                       score_fn: _ScoreFn) -> NSRTSampler:
    """Create a wrapped sampler that uses a score function to select among
    candidates from a base sampler."""

    def _sample(state: State, goal: Set[GroundAtom], rng: np.random.Generator,
                objects: Sequence[Object]) -> Array:
        samples = [
            base_sampler(state, goal, rng, objects)
            for _ in range(CFG.active_sampler_learning_num_samples)
        ]
        scores = score_fn(state, objects, samples)
        idx = int(np.argmax(scores))
        return samples[idx]

    return _sample


def _wrap_sampler_exploration(
        base_sampler: NSRTSampler, score_fn: _ScoreFn,
        strategy: str) -> NSRTSamplerWithEpsilonIndicator:

    def _sample(state: State, goal: Set[GroundAtom], rng: np.random.Generator,
                objects: Sequence[Object]) -> Tuple[Array, bool]:
        samples = [
            base_sampler(state, goal, rng, objects)
            for _ in range(CFG.active_sampler_learning_num_samples)
        ]
        scores = score_fn(state, objects, samples)
        if strategy in ["greedy", "epsilon_greedy"]:
            idx = int(np.argmax(scores))
            epsilon_bool = False
            if strategy == "epsilon_greedy" and rng.uniform(
            ) <= CFG.active_sampler_learning_exploration_epsilon:
                # Randomly select a sample to pick, following the epsilon
                # greedy strategy!
                idx = rng.integers(0, len(scores))
                epsilon_bool = True
        else:
            raise NotImplementedError('Exploration strategy ' +
                                      f'{strategy} ' + 'is not implemented.')
        return (samples[idx], epsilon_bool)

    return _sample


def _wrap_object_specific_samplers(
    object_specific_samplers: Dict[Tuple[Object, ...], NSRTSampler],
    base_sampler: NSRTSampler,
) -> NSRTSampler:

    def _wrapped_sampler(
            state: State, goal: Set[GroundAtom], rng: np.random.Generator,
            objects: Sequence[Object]) -> Array:  # pragma: no cover
        objects_tuple = tuple(objects)
        # If we haven't yet learned a object-specific sampler for these objects
        # then use the base sampler.
        if objects_tuple not in object_specific_samplers:
            return base_sampler(state, goal, rng, objects)
        sampler = object_specific_samplers[objects_tuple]
        return sampler(state, goal, rng, objects)

    return _wrapped_sampler


def _wrap_object_specific_samplers_with_epsilon(
    object_specific_samplers: Dict[Tuple[Object, ...],
                                   NSRTSamplerWithEpsilonIndicator],
    base_sampler: NSRTSampler,
) -> NSRTSamplerWithEpsilonIndicator:

    def _wrapped_sampler(
            state: State, goal: Set[GroundAtom], rng: np.random.Generator,
            objects: Sequence[Object]
    ) -> Tuple[Array, bool]:  # pragma: no cover
        objects_tuple = tuple(objects)
        # If we haven't yet learned a object-specific sampler for these objects
        # then use the base sampler. Treat the output as if it was greedy
        # (epsilon = True).
        if objects_tuple not in object_specific_samplers:  # pragma: no cover
            return base_sampler(state, goal, rng, objects), True
        sampler = object_specific_samplers[objects_tuple]
        return sampler(state, goal, rng, objects)

    return _wrapped_sampler


def _vector_score_fn_to_score_fn(vector_fn: Callable[[Array], float],
                                 nsrt: NSRT) -> _ScoreFn:
    """Helper for _classifier_to_score_fn() and _regressor_to_score_fn()."""

    def _score_fn(state: State, objects: Sequence[Object],
                  param_lst: List[Array]) -> List[float]:
        xs = [
            utils.construct_active_sampler_input(state, objects, p,
                                                 nsrt.option)
            for p in param_lst
        ]
        scores = [vector_fn(x) for x in xs]
        return scores

    return _score_fn


def _classifier_to_score_fn(classifier: BinaryClassifier,
                            nsrt: NSRT) -> _ScoreFn:
    return _vector_score_fn_to_score_fn(classifier.predict_proba, nsrt)


def _classifier_ensemble_to_score_fn(classifier: BinaryClassifierEnsemble,
                                     nsrt: NSRT, test_time: bool) -> _ScoreFn:
    if test_time:
        return _vector_score_fn_to_score_fn(
            lambda x: np.mean(classifier.predict_member_probas(x), dtype=float
                              ), nsrt)
    # If we want the exploration score function, then we need to compute the
    # entropy.
    return _vector_score_fn_to_score_fn(
        lambda x: utils.entropy(
            float(np.mean(classifier.predict_member_probas(x)))), nsrt)


def _regressor_to_score_fn(regressor: MLPRegressor, nsrt: NSRT) -> _ScoreFn:
    fn = lambda v: regressor.predict(v)[0]
    return _vector_score_fn_to_score_fn(fn, nsrt)
