"""Code for learning the samplers within NSRTs."""

import logging
from dataclasses import dataclass
from typing import Any, List, Sequence, Set, Tuple

import numpy as np

from predicators.src import utils
from predicators.src.envs import get_or_create_env
from predicators.src.ground_truth_nsrts import get_gt_nsrts
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Array, Datastore, EntToEntSub, \
    GroundAtom, LiftedAtom, NSRTSampler, Object, OptionSpec, \
    ParameterizedOption, SamplerDatapoint, State, STRIPSOperator, Variable
from predicators.src.torch_models import Classifier, MLPClassifier, \
    NeuralGaussianRegressor, GNNRegressor


def learn_samplers(strips_ops: List[STRIPSOperator],
                   datastores: List[Datastore], option_specs: List[OptionSpec],
                   sampler_learner: str) -> List[NSRTSampler]:
    """Learn all samplers for each operator's option parameters."""
    if sampler_learner == "oracle":
        return _extract_oracle_samplers(strips_ops, option_specs)
    samplers = []
    for i, op in enumerate(strips_ops):
        param_option, _ = option_specs[i]
        if sampler_learner == "random" or \
           param_option.params_space.shape == (0,):
            sampler: NSRTSampler = _RandomSampler(param_option).sampler
        elif sampler_learner == "neural":
            sampler = _learn_neural_sampler(datastores, op.name, op.parameters,
                                            op.preconditions, op.add_effects,
                                            op.delete_effects, param_option, i)
        else:
            raise NotImplementedError("Unknown sampler_learner: "
                                      f"{CFG.sampler_learner}")
        samplers.append(sampler)
    return samplers


def _extract_oracle_samplers(
    strips_ops: List[STRIPSOperator],
    option_specs: List[OptionSpec],
) -> List[NSRTSampler]:
    """Extract the oracle samplers matching the given STRIPSOperator objects
    from the ground truth operators defined in approaches/oracle_approach.py.

    We require every ground truth operator to match one of the given
    operators, but some of the given operators can match no ground truth
    operator, in which case such an operator is given a random sampler.
    """
    env = get_or_create_env(CFG.env)
    # We don't need to match ground truth NSRTs with no continuous
    # parameters, so we filter them out.
    gt_nsrts = {
        nsrt
        for nsrt in get_gt_nsrts(env.predicates, env.options)
        if nsrt.option.params_space.shape != (0, )
    }
    assert len(strips_ops) == len(option_specs)
    # Initialize all samplers to random.
    samplers: List[NSRTSampler] = [
        _RandomSampler(param_option).sampler
        for param_option, _ in option_specs
    ]
    # Go through the ground truth NSRTs. For each one, if we find a
    # matching to a given operator, extract the NSRT's sampler.
    for nsrt in gt_nsrts:
        # Use unification to find a matching.
        for idx, (op, (param_option,
                       option_vars)) in enumerate(zip(strips_ops,
                                                      option_specs)):
            # If option learning, the names of the option will not match.
            # Ignore this by allowing the learned NSRT option to just be
            # the ground truth one.
            if CFG.option_learner != "no_learning":
                param_option = nsrt.option
            suc, sub = utils.unify_preconds_effects_options(
                frozenset(nsrt.preconditions), frozenset(op.preconditions),
                frozenset(nsrt.add_effects), frozenset(op.add_effects),
                frozenset(nsrt.delete_effects), frozenset(op.delete_effects),
                nsrt.option, param_option, tuple(nsrt.option_vars),
                tuple(option_vars))
            if suc:  # match found!
                samplers[idx] = _make_reordered_sampler(nsrt, op, sub)
                break
        else:
            raise Exception("Can't use oracle samplers, no match for "
                            f"ground truth NSRT: {nsrt}")
    return samplers


def _make_reordered_sampler(nsrt: NSRT, op: STRIPSOperator,
                            sub: EntToEntSub) -> NSRTSampler:
    """Helper for _extract_oracle_samplers()."""

    def _reordered_sampler(state: State, goal: Set[GroundAtom],
                           rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        # Use the sub dictionary to correctly order the arguments
        # to the NSRT sampler.
        reordered_objs = []
        for param in nsrt.parameters:
            param_idx = op.parameters.index(sub[param])
            reordered_objs.append(objs[param_idx])
        return nsrt.sampler(state, goal, rng, reordered_objs)

    return _reordered_sampler


def _learn_neural_sampler(datastores: List[Datastore], nsrt_name: str,
                          variables: Sequence[Variable],
                          preconditions: Set[LiftedAtom],
                          add_effects: Set[LiftedAtom],
                          delete_effects: Set[LiftedAtom],
                          param_option: ParameterizedOption,
                          datastore_idx: int) -> NSRTSampler:
    """Learn a neural network sampler given data.

    Transitions are clustered, so that they can be used for generating
    negative data. Integer datastore_idx represents the index into
    transitions corresponding to the datastore that this sampler is
    being learned for.
    """
    logging.info(f"\nLearning neural sampler for NSRT {nsrt_name}")

    positive_data, negative_data = _create_sampler_data(
        datastores, variables, preconditions, add_effects, delete_effects,
        param_option, datastore_idx)

    # Fit classifier to data
    logging.info("Fitting classifier...")
    X_classifier: List[List[Array]] = []
    for state, sub, option, goal in positive_data + negative_data:
        # input is state features and option parameters
        X_classifier.append([np.array(1.0)])  # start with bias term
        for var in variables:
            X_classifier[-1].extend(state[sub[var]])
        X_classifier[-1].extend(option.params)
        # For sampler learning, we currently make the extremely limiting
        # assumption that there is one goal atom, with one goal object. This
        # will not be true in most cases. This is a placeholder for better
        # methods to come.
        if CFG.sampler_learning_use_goals:
            assert goal is not None
            # assert len(goal) == 1
            goal_atom = next(iter(goal))
            # assert len(goal_atom.objects) == 1
            goal_obj = goal_atom.objects[0]
            X_classifier[-1].extend(state[goal_obj])
    X_arr_classifier = np.array(X_classifier)
    # output is binary signal
    y_arr_classifier = np.array([1 for _ in positive_data] +
                                [0 for _ in negative_data])
    classifier = MLPClassifier(X_arr_classifier.shape[1],
                               CFG.sampler_mlp_classifier_max_itr)
    classifier.fit(X_arr_classifier, y_arr_classifier)

    gnn_model = None
    regressor = None
    if CFG.sampler_use_gnn: 
        assert CFG.sampler_learning_use_goals # goals used for graph
        # Fit GNN to data 
        logging.info("Fitting GNN...")    
        gnn_model = GNNRegressor({param_option}) 
        gnn_model.setup_fields(positive_data)

        X = [] 
        Y = []
        for state, sub, option, goal in positive_data: 
            # state features for global vector 
            state_feature = np.array([1.0])
            for var in variables: 
                state_feature = np.concatenate((state_feature, state[sub[var]]))
            # goal objects and their states
            goal_objs_to_states = {} 
            for atom in goal: 
                for obj in atom.objects:
                    goal_objs_to_states[obj] = state[obj]
            X.append((state_feature, goal, goal_objs_to_states))
            Y.append(option)
        graph_data = gnn_model.graphify_data(X,Y)
        gnn_model.learn_from_graph_data(graph_data)

    else: 
        # Fit regressor to data
        logging.info("Fitting regressor...")
        X_regressor: List[List[Array]] = []
        Y_regressor = []

        for state, sub, option, goal in positive_data:  # don't use negative data!
            # input is state features
            X_regressor.append([np.array(1.0)])  # start with bias term
            for var in variables:
                X_regressor[-1].extend(state[sub[var]])
            # Above, we made the assumption that there is one goal atom with one
            # goal object, which must also be used when assembling data for the
            # regressor.
            if CFG.sampler_learning_use_goals:
                assert goal is not None
                assert len(goal) == 1
                goal_atom = next(iter(goal))
                assert len(goal_atom.objects) == 1
                goal_obj = goal_atom.objects[0]
                X_regressor[-1].extend(state[goal_obj])
            # output is option parameters
            Y_regressor.append(option.params)
        X_arr_regressor = np.array(X_regressor)
        Y_arr_regressor = np.array(Y_regressor)
        regressor = NeuralGaussianRegressor()
        regressor.fit(X_arr_regressor, Y_arr_regressor)

    # Construct and return sampler
    return _LearnedSampler(classifier, regressor, gnn_model, variables,
                           param_option).sampler


def _create_sampler_data(
    datastores: List[Datastore], variables: Sequence[Variable],
    preconditions: Set[LiftedAtom], add_effects: Set[LiftedAtom],
    delete_effects: Set[LiftedAtom], param_option: ParameterizedOption,
    datastore_idx: int
) -> Tuple[List[SamplerDatapoint], List[SamplerDatapoint]]:
    """Generate positive and negative data for training a sampler."""
    positive_data = []
    negative_data = []
    for idx, datastore in enumerate(datastores):
        for (segment, var_to_obj) in datastore:
            assert segment.has_option()
            option = segment.get_option()
            state = segment.states[0]
            if CFG.sampler_learning_use_goals:
                # Right now, we're making the assumption that all data is
                # demonstration data when we're learning samplers with goals.
                # In the future, we may weaken this assumption.
                assert segment.has_goal()
                goal = segment.get_goal()
            else:
                goal = None
            trans_add_effects = segment.add_effects
            trans_delete_effects = segment.delete_effects
            if option.parent != param_option:
                continue
            var_types = [var.type for var in variables]
            objects = list(state)
            for grounding in utils.get_object_combinations(objects, var_types):
                # If we are currently at the datastore that we're learning a
                # sampler for, and this datapoint matches the actual grounding,
                # add it to the positive data and continue.
                if idx == datastore_idx:
                    actual_grounding = [var_to_obj[var] for var in variables]
                    if grounding == actual_grounding:
                        assert all(
                            pre.predicate.holds(
                                state, [var_to_obj[v] for v in pre.variables])
                            for pre in preconditions)
                        positive_data.append((state, var_to_obj, option, goal))
                        continue
                if CFG.sampler_disable_classifier:
                    # We disable the classifier by not providing it any
                    # negative examples, so that it always outputs 1.
                    continue
                sub = dict(zip(variables, grounding))
                # When building data for a datastore with effects X, if we
                # encounter a transition with effects Y, and if Y is a superset
                # of X, then we do not want to include the transition as a
                # negative example, because if Y was achieved, then X was also
                # achieved. So for now, we just filter out such examples.
                ground_add_effects = {e.ground(sub) for e in add_effects}
                ground_delete_effects = {e.ground(sub) for e in delete_effects}
                if ground_add_effects.issubset(trans_add_effects) and \
                   ground_delete_effects.issubset(trans_delete_effects):
                    continue
                # Add this datapoint to the negative data.
                negative_data.append((state, sub, option, goal))
    logging.info(f"Generated {len(positive_data)} positive and "
                 f"{len(negative_data)} negative examples")
    assert len(positive_data) == len(datastores[datastore_idx])
    return positive_data, negative_data


@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned
    sampler."""
    _classifier: Classifier
    _regressor: NeuralGaussianRegressor
    _gnnregressor: GNNRegressor
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
        # For sampler learning, we currently make the extremely limiting
        # assumption that there is one goal atom, with one goal object. This
        # will not be true in most cases. This is a placeholder for better
        # methods to come.
        if CFG.sampler_learning_use_goals:
            # assert len(goal) == 1
            goal_atom = next(iter(goal))
            # assert len(goal_atom.objects) == 1
            goal_obj = goal_atom.objects[0]
            x_lst.extend(state[goal_obj])  # add goal state
        x = np.array(x_lst)
        num_rejections = 0
        if CFG.sampler_disable_classifier:
            if CFG.sampler_use_gnn: 
                # state features for global vector 
                state_feature = np.array([1.0])
                for var in self._variables: 
                    state_feature = np.concatenate((state_feature, state[sub[var]]))

                # goal objects and their states
                goal_objs_to_states = {} 
                for atom in goal: 
                    for obj in atom.objects:
                        goal_objs_to_states[obj] = state[obj]

                sample = self._gnnregressor.predict_sample(state_feature, goal, goal_objs_to_states)
                params = np.array(sample, dtype=self._param_option.params_space.dtype)
                low = self._param_option.params_space.low
                high = self._param_option.params_space.high
                params = np.clip(params, low, high)
                return params
            else: 
                params = np.array(self._regressor.predict_sample(x, rng),
                                  dtype=self._param_option.params_space.dtype)
            low = self._param_option.params_space.low
            high = self._param_option.params_space.high
            params = np.clip(params, low, high)
            return params
        while num_rejections <= CFG.max_rejection_sampling_tries:
            params = np.array(self._regressor.predict_sample(x, rng),
                              dtype=self._param_option.params_space.dtype)
            if self._param_option.params_space.contains(params) and \
               self._classifier.classify(np.r_[x, params]):
                break
            num_rejections += 1
        else:
            # Edge case: we exceeded the number of sampling tries
            # and we might be left with a params that is not in
            # bounds. If so, fall back to sampling from the space.
            if not self._param_option.params_space.contains(params):
                params = self._param_option.params_space.sample()
        return params


@dataclass(frozen=True, eq=False, repr=False)
class _RandomSampler:
    """A convenience class for implementing a random sampler."""
    _param_option: ParameterizedOption

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object]) -> Array:
        """A random sampler for this option.

        Ignores all arguments.
        """
        del state, goal, rng, objects  # unused
        return self._param_option.params_space.sample()
