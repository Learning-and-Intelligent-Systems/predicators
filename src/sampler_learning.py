"""Code for learning the samplers within NSRTs."""

from dataclasses import dataclass
from typing import Set, Tuple, List, Sequence, Dict, Any
import numpy as np
from predicators.src.structs import ParameterizedOption, LiftedAtom, Variable, \
    Object, Array, State, _Option, Datastore, STRIPSOperator, OptionSpec, \
    NSRTSampler, NSRT
from predicators.src import utils
from predicators.src.torch_models import MLPClassifier, NeuralGaussianRegressor
from predicators.src.settings import CFG
from predicators.src.envs import create_env
from predicators.src.approaches.oracle_approach import get_gt_nsrts


def learn_samplers(strips_ops: List[STRIPSOperator],
                   datastores: List[Datastore], option_specs: List[OptionSpec],
                   sampler_learner: str) -> List[NSRTSampler]:
    """Learn all samplers for each operator's option parameters."""
    samplers = []
    if sampler_learner == "oracle":
        env = create_env(CFG.env)
        # We don't need to match ground truth NSRTs with no
        # continuous parameters, so filter them out.
        gt_nsrts = {nsrt for nsrt in get_gt_nsrts(env.predicates, env.options)
                    if nsrt.option.params_space.shape != (0,)}
    for i, op in enumerate(strips_ops):
        param_option, option_vars = option_specs[i]
        if sampler_learner == "random" or \
           param_option.params_space.shape == (0,):
            sampler: NSRTSampler = _RandomSampler(param_option).sampler
        elif sampler_learner == "oracle":
            sampler = _extract_oracle_sampler(
                op, param_option, option_vars, gt_nsrts)
        elif sampler_learner == "neural":
            sampler = _learn_neural_sampler(datastores, op.name, op.parameters,
                                            op.preconditions, op.add_effects,
                                            op.delete_effects, param_option, i)
        else:
            raise NotImplementedError("Unknown sampler_learner: "
                                      f"{CFG.sampler_learner}")
        samplers.append(sampler)
    if sampler_learner == "oracle":
        assert not gt_nsrts, f"Can't use oracle samplers, {len(gt_nsrts)} " \
            "oracle operator(s) were not matched to a learned operator!"
    return samplers


def _extract_oracle_sampler(op: STRIPSOperator,
                            param_option: ParameterizedOption,
                            option_vars: Sequence[Variable],
                            gt_nsrts: Set[NSRT]) -> NSRTSampler:
    """Extract the oracle sampler matching the given STRIPSOperator
    from the ground truth operators defined in approaches/oracle_approach.py.
    If no ground truth operator can be unified with the given one, just
    returns a random sampler.
    """
    matching_nsrt = None
    matching_sub = None
    for nsrt in gt_nsrts:
        suc, sub = utils.unify_preconds_effects_options(
            frozenset(op.preconditions),
            frozenset(nsrt.preconditions),
            frozenset(op.add_effects),
            frozenset(nsrt.add_effects),
            frozenset(op.delete_effects),
            frozenset(nsrt.delete_effects),
            param_option,
            nsrt.option,
            tuple(option_vars),
            tuple(nsrt.option_vars))
        if suc:
            matching_nsrt = nsrt
            matching_sub = sub
            break
    if matching_nsrt is None:
        # Fall back to random sampler.
        return _RandomSampler(param_option).sampler
    # We found a matching operator. Use the matching_sub dictionary to
    # correctly order the arguments to the sampler.
    def _reordered_sampler(state: State, rng: np.random.Generator,
                           objs: Sequence[Object]) -> Array:
        assert matching_nsrt is not None
        assert matching_sub is not None
        reordered_objs = []
        rev_sub = {v: k for k, v in matching_sub.items()}
        for param in matching_nsrt.parameters:
            idx = op.parameters.index(rev_sub[param])
            reordered_objs.append(objs[idx])
        return matching_nsrt.get_sampler()(state, rng, reordered_objs)
    gt_nsrts.remove(matching_nsrt)  # remove this NSRT from the set
    return _reordered_sampler


def _learn_neural_sampler(datastores: List[Datastore],
                          nsrt_name: str,
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
    print(f"\nLearning neural sampler for NSRT {nsrt_name}")

    positive_data, negative_data = _create_sampler_data(
        datastores, variables, preconditions, add_effects, delete_effects,
        param_option, datastore_idx)

    # Fit classifier to data
    print("Fitting classifier...")
    X_classifier: List[List[Array]] = []
    for state, sub, option in positive_data + negative_data:
        # input is state features and option parameters
        X_classifier.append([np.array(1.0)])  # start with bias term
        for var in variables:
            X_classifier[-1].extend(state[sub[var]])
        X_classifier[-1].extend(option.params)
    X_arr_classifier = np.array(X_classifier)
    # output is binary signal
    y_arr_classifier = np.array([1 for _ in positive_data] +
                                [0 for _ in negative_data])
    classifier = MLPClassifier(X_arr_classifier.shape[1],
                               CFG.classifier_max_itr_sampler)
    classifier.fit(X_arr_classifier, y_arr_classifier)

    # Fit regressor to data
    print("Fitting regressor...")
    X_regressor: List[List[Array]] = []
    Y_regressor = []
    for state, sub, option in positive_data:  # don't use negative data!
        # input is state features
        X_regressor.append([np.array(1.0)])  # start with bias term
        for var in variables:
            X_regressor[-1].extend(state[sub[var]])
        # output is option parameters
        Y_regressor.append(option.params)
    X_arr_regressor = np.array(X_regressor)
    Y_arr_regressor = np.array(Y_regressor)
    regressor = NeuralGaussianRegressor()
    regressor.fit(X_arr_regressor, Y_arr_regressor)

    # Construct and return sampler
    return _LearnedSampler(classifier, regressor, variables,
                           param_option).sampler


def _create_sampler_data(
    datastores: List[Datastore], variables: Sequence[Variable],
    preconditions: Set[LiftedAtom], add_effects: Set[LiftedAtom],
    delete_effects: Set[LiftedAtom], param_option: ParameterizedOption,
    datastore_idx: int
) -> Tuple[List[Tuple[State, Dict[Variable, Object], _Option]], ...]:
    """Generate positive and negative data for training a sampler."""
    positive_data = []
    negative_data = []
    for idx, datastore in enumerate(datastores):
        for (segment, obj_to_var) in datastore:
            assert segment.has_option()
            option = segment.get_option()
            state = segment.states[0]
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
                    var_to_obj = {v: k for k, v in obj_to_var.items()}
                    actual_grounding = [var_to_obj[var] for var in variables]
                    if grounding == actual_grounding:
                        assert all(
                            pre.predicate.holds(
                                state, [var_to_obj[v] for v in pre.variables])
                            for pre in preconditions)
                        positive_data.append((state, var_to_obj, option))
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
                negative_data.append((state, sub, option))
    print(f"Generated {len(positive_data)} positive and {len(negative_data)} "
          f"negative examples")
    assert len(positive_data) == len(datastores[datastore_idx])
    return positive_data, negative_data


@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned sampler.
    """
    _classifier: MLPClassifier
    _regressor: NeuralGaussianRegressor
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption

    def sampler(self, state: State, rng: np.random.Generator,
                objects: Sequence[Object]) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        x_lst: List[Any] = [1.0]  # start with bias term
        sub = dict(zip(self._variables, objects))
        for var in self._variables:
            x_lst.extend(state[sub[var]])
        x = np.array(x_lst)
        num_rejections = 0
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
    """A convenience class for implementing a random sampler.
    """
    _param_option: ParameterizedOption

    def sampler(self, state: State, rng: np.random.Generator,
                objects: Sequence[Object]) -> Array:
        """A random sampler for this option.

        Ignores all arguments.
        """
        del state, rng, objects  # unused
        return self._param_option.params_space.sample()
