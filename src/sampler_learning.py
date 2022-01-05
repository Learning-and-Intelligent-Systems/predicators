"""Code for learning the samplers within NSRTs.
"""

from dataclasses import dataclass
from typing import Set, Tuple, List, Sequence, Dict, Any
import numpy as np
from predicators.src.structs import ParameterizedOption, LiftedAtom, Variable, \
    Object, Array, State, _Option, Partition, STRIPSOperator, OptionSpec, \
    NSRTSampler
from predicators.src import utils
from predicators.src.torch_models import MLPClassifier, NeuralGaussianRegressor
from predicators.src.settings import CFG


def learn_samplers(
    strips_ops: List[STRIPSOperator],
    partitions: List[Partition],
    option_specs: List[OptionSpec],
    do_sampler_learning: bool
    ) -> List[NSRTSampler]:
    """Learn all samplers for each operator's option parameters.
    """
    samplers = []
    for i, op in enumerate(strips_ops):
        sampler = _learn_sampler(
            partitions, op.name, op.parameters, op.preconditions,
            op.add_effects, op.delete_effects, option_specs[i][0], i,
            do_sampler_learning)
        samplers.append(sampler)
    return samplers


def _learn_sampler(partitions: List[Partition],
                   nsrt_name: str,
                   variables: Sequence[Variable],
                   preconditions: Set[LiftedAtom],
                   add_effects: Set[LiftedAtom],
                   delete_effects: Set[LiftedAtom],
                   param_option: ParameterizedOption,
                   partition_idx: int, do_sampler_learning: bool
                   ) -> NSRTSampler:
    """Learn a sampler given data. Transitions are partitioned, so
    that they can be used for generating negative data. Integer partition_idx
    represents the index into transitions corresponding to the partition that
    this sampler is being learned for. If do_sampler_learning is False,
    just returns a random sampler.
    """
    if not do_sampler_learning or param_option.params_space.shape == (0,):
        return _RandomSampler(param_option).sampler
    print(f"\nLearning sampler for NSRT {nsrt_name}")

    positive_data, negative_data = _create_sampler_data(
        partitions, variables, preconditions, add_effects, delete_effects,
        param_option, partition_idx)

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
        partitions: List[Partition],
        variables: Sequence[Variable],
        preconditions: Set[LiftedAtom],
        add_effects: Set[LiftedAtom],
        delete_effects: Set[LiftedAtom],
        param_option: ParameterizedOption,
        partition_idx: int) -> Tuple[List[Tuple[State,
                                     Dict[Variable, Object], _Option]], ...]:
    """Generate positive and negative data for training a sampler.
    """
    positive_data = []
    negative_data = []
    for idx, partition in enumerate(partitions):
        for (segment, obj_to_var) in partition:
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
                # If we are currently at the partition that we're learning a
                # sampler for, and this datapoint matches the actual grounding,
                # add it to the positive data and continue.
                if idx == partition_idx:
                    var_to_obj = {v: k for k, v in obj_to_var.items()}
                    actual_grounding = [var_to_obj[var] for var in variables]
                    if grounding == actual_grounding:
                        assert all(pre.predicate.holds(
                            state, [var_to_obj[v] for v in pre.variables])
                                   for pre in preconditions)
                        positive_data.append((state, var_to_obj, option))
                        continue
                sub = dict(zip(variables, grounding))
                # When building data for a partition with effects X, if we
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
    assert len(positive_data) == len(partitions[partition_idx])
    return positive_data, negative_data


@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned sampler.
    Prefer to use this because it is pickleable.
    """
    _classifier: MLPClassifier
    _regressor: NeuralGaussianRegressor
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption

    def sampler(self, state: State, rng: np.random.Generator,
                objects: Sequence[Object]) -> Array:
        """The sampler corresponding to the given models. May be used
        as the _sampler field in an NSRT.
        """
        x_lst : List[Any] = [1.0]  # start with bias term
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
    """A convenience class for implementing a random sampler. Prefer
    to use this over a lambda function because it is pickleable.
    """
    _param_option: ParameterizedOption

    def sampler(self, state: State, rng: np.random.Generator,
                objects: Sequence[Object]) -> Array:
        """A random sampler for this option. Ignores all arguments.
        """
        del state, rng, objects  # unused
        return self._param_option.params_space.sample()
