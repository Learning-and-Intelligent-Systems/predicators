"""Code for learning the samplers within NSRTs."""

import logging
from dataclasses import dataclass
from math import trunc
from typing import Any, List, Sequence, Set, Tuple

import numpy as np
import pybullet as p
from predicators import utils
from predicators.envs import get_or_create_env
from predicators.ground_truth_models import get_gt_nsrts, get_gt_options
from predicators.ml_models import BinaryClassifier, \
    DegenerateMLPDistributionRegressor, DistributionRegressor, \
    MLPBinaryClassifier, NeuralGaussianRegressor, DiffusionRegressor
from predicators.nsrt_learning import sampler_visualizer
from predicators.settings import CFG
from predicators.structs import NSRT, Array, Datastore, EntToEntSub, \
    GroundAtom, LiftedAtom, NSRTSampler, Object, OptionSpec, \
    ParameterizedOption, SamplerDatapoint, State, STRIPSOperator, Variable
from matplotlib import pyplot as plt


def learn_samplers(strips_ops: List[STRIPSOperator],
                   datastores: List[Datastore], option_specs: List[OptionSpec],
                   sampler_learner: str) -> List[NSRTSampler]:
    """Learn all samplers for each operator's option parameters."""
    if sampler_learner == "oracle":
        return _extract_oracle_samplers(strips_ops, option_specs)
    samplers = []
    for i, op in enumerate(strips_ops):
        logging.info(strips_ops)
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

    If a given operator does not match any ground truth operator, it is
    given a random sampler. If a ground truth operator has no given
    operator match, a warning is generated.
    """
    env = get_or_create_env(CFG.env)
    env_options = get_gt_options(env.get_name())
    # We don't need to match ground truth NSRTs with no continuous
    # parameters, so we filter them out.
    gt_nsrts = {
        nsrt
        for nsrt in get_gt_nsrts(env.get_name(), env.predicates, env_options)
        if nsrt.option.params_space.shape != (0,)
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
            logging.warning("Oracle sampler learning found no match for "
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
    logging.info(f"Generated {len(positive_data)} positive and "
                 f"{len(negative_data)} negative examples")

    # Fit classifier to data
    classifier = None

    if not CFG.sampler_disable_classifier:
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
                assert len(goal) == 1
                goal_atom = next(iter(goal))
                assert len(goal_atom.objects) == 1
                goal_obj = goal_atom.objects[0]
                X_classifier[-1].extend(state[goal_obj])

        if nsrt_name == "PutOnTableInZone":
            X_arr_classifier = np.array(X_classifier)[:, :-1]
        else:
            X_arr_classifier = np.array(X_classifier)
        logging.info(f'classifier: {X_arr_classifier.shape}')
        # output is binary signal
        y_arr_classifier = np.array([1 for _ in positive_data] +
                                    [0 for _ in negative_data])
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
    X_arr_regressor = np.array(X_regressor)[:, :]
    np.set_printoptions(threshold=np.inf)

    Y_arr_regressor = np.array(Y_regressor)

    if CFG.sampler_learning_regressor_model == "neural_gaussian":
        regressor: DistributionRegressor = NeuralGaussianRegressor(
            seed=CFG.seed,
            hid_sizes=CFG.neural_gaus_regressor_hid_sizes,
            max_train_iters=CFG.neural_gaus_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate)
    elif CFG.sampler_learning_regressor_model == "diffusion":
        assert CFG.sampler_disable_classifier
        regressor = DiffusionRegressor(
            seed=CFG.seed,
            hid_sizes=CFG.neural_gaus_regressor_hid_sizes,
            max_train_iters=CFG.neural_gaus_regressor_max_itr,
            timesteps=100,
            learning_rate=CFG.learning_rate,
        )
    else:
        assert CFG.sampler_learning_regressor_model == "degenerate_mlp"
        regressor = DegenerateMLPDistributionRegressor(
            seed=CFG.seed,
            hid_sizes=CFG.mlp_regressor_hid_sizes,
            max_train_iters=CFG.mlp_regressor_max_itr,
            clip_gradients=CFG.mlp_regressor_clip_gradients,
            clip_value=CFG.mlp_regressor_gradient_clip_value,
            learning_rate=CFG.learning_rate)

    regressor.fit(X_arr_regressor, Y_arr_regressor)

    if nsrt_name == "PickFromTable":

        ys = []
        xs = []
        x_true = []
        y_true = []
        rng = np.random.default_rng(CFG.seed)
        for x in X_arr_regressor:
            y_hat = regressor.predict_sample(x, rng)

            if not CFG.sampler_disable_classifier:
               if classifier.classify(np.r_[x, y_hat]):
                    x_cart = y_hat[0] * np.cos(y_hat[1] * 2 * np.pi)
                    y_cart = y_hat[0] * np.sin(y_hat[1] * 2 * np.pi)

                    xs.append(x_cart)
                    ys.append(y_cart)
            else:
                x_cart = y_hat[0] * np.cos(y_hat[1] * 2 * np.pi)
                y_cart = y_hat[0] * np.sin(y_hat[1] * 2 * np.pi)

                xs.append(x_cart)
                ys.append(y_cart)

        for y in Y_arr_regressor:
            x_cart_true = y[0] * np.cos(y[1] * 2 * np.pi)
            y_cart_true = y[0] * np.sin(y[1] * 2 * np.pi)

            x_true.append(x_cart_true)
            y_true.append(y_cart_true)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figsize as needed


        axes[0].scatter(x_true, y_true, alpha=0.3)
        axes[0].set_title(f"{nsrt_name} Option: Ground Truth",fontsize=20)
        axes[0].set_xlabel("x",fontsize=18)
        axes[0].set_ylabel("y",fontsize=18)
        axes[0].tick_params(axis='x', labelsize=15)
        axes[0].tick_params(axis='y', labelsize=15)

        # First subplot: Gaussian Sampler With Rejection Classifier
        axes[1].scatter(xs, ys, color='green', alpha=0.3)
        axes[1].set_xlabel("x",fontsize=18)
        axes[1].set_ylabel("y",fontsize=18)
        axes[1].set_title(f"{nsrt_name} Option: {CFG.sampler_learning_regressor_model}", fontsize=20)
        axes[1].tick_params(axis='x',labelsize=15)
        axes[1].tick_params(axis='y',labelsize=15)

        # Third subplot: Overlay
        axes[2].scatter(xs, ys, color='green', alpha=0.3, label=CFG.sampler_learning_regressor_model)
        axes[2].scatter(x_true, y_true, alpha=0.3, label="GT")
        axes[2].set_xlabel("x",fontsize=18)
        axes[2].set_ylabel("y",fontsize=18)
        axes[2].set_title(f"{nsrt_name} Option: Comparison", fontsize=20)
        axes[2].tick_params(axis='x',labelsize=15)
        axes[2].tick_params(axis='y',labelsize=15)
        axes[2].legend(fontsize=18)

        # Adjust layout and save the combined figure
        plt.tight_layout()
        plt.savefig(f"combined_row_{nsrt_name}_{CFG.sampler_learning_regressor_model}.pdf", format="pdf")
        plt.close()


    if nsrt_name == "PutOnTableAroundPole":
        nsrt_name = "PutAroundPole"
        ys = []
        xs = []
        x_true = []
        y_true = []
        rng = np.random.default_rng(CFG.seed)
        for x in X_arr_regressor:
            y_hat = regressor.predict_sample(x, rng)

            if not CFG.sampler_disable_classifier:
               if classifier.classify(np.r_[x, y_hat]):
                    x_cart = y_hat[0]
                    y_cart = y_hat[1]

                    xs.append(x_cart)
                    ys.append(y_cart)
            else:
                x_cart = y_hat[0]
                y_cart = y_hat[1]

                xs.append(x_cart)
                ys.append(y_cart)

        for y in Y_arr_regressor:
            x_cart_true = y[0]
            y_cart_true = y[1]

            x_true.append(x_cart_true)
            y_true.append(y_cart_true)

        fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Adjust figsize as needed


        axes[0].scatter(x_true, y_true, alpha=0.3)
        axes[0].set_title(f"{nsrt_name} Option: Ground Truth",fontsize=20)
        axes[0].set_xlabel("x",fontsize=18)
        axes[0].set_ylabel("y",fontsize=18)
        axes[0].tick_params(axis='x', labelsize=15)
        axes[0].tick_params(axis='y', labelsize=15)

        # First subplot: Gaussian Sampler With Rejection Classifier
        axes[1].scatter(xs, ys, color='green', alpha=0.3)
        axes[1].set_xlabel("x",fontsize=18)
        axes[1].set_ylabel("y",fontsize=18)
        axes[1].set_title(f"{nsrt_name} Option: {CFG.sampler_learning_regressor_model}", fontsize=20)
        axes[1].tick_params(axis='x',labelsize=15)
        axes[1].tick_params(axis='y',labelsize=15)

        # Third subplot: Overlay
        axes[2].scatter(xs, ys, color='green', alpha=0.3, label=CFG.sampler_learning_regressor_model)
        axes[2].scatter(x_true, y_true, alpha=0.3, label="GT")
        axes[2].set_xlabel("x",fontsize=18)
        axes[2].set_ylabel("y",fontsize=18)
        axes[2].set_title(f"{nsrt_name} Option: Comparison", fontsize=20)
        axes[2].tick_params(axis='x',labelsize=15)
        axes[2].tick_params(axis='y',labelsize=15)
        axes[2].legend(fontsize=18)

        # Adjust layout and save the combined figure
        plt.tight_layout()
        plt.savefig(f"combined_row_{nsrt_name}_{CFG.sampler_learning_regressor_model}.pdf", format="pdf")
        plt.close()

    # Construct and return sampler
    return _LearnedSampler(classifier, regressor, variables,
                           param_option).sampler


def _create_sampler_data(
        datastores: List[Datastore], variables: Sequence[Variable],
        preconditions: Set[LiftedAtom], add_effects: Set[LiftedAtom],
        delete_effects: Set[LiftedAtom], param_option: ParameterizedOption,
        datastore_idx: int
) -> Tuple[List[SamplerDatapoint], List[SamplerDatapoint]]:
    """Generate positive and negative data for training a sampler."""
    # Populate all positive data.
    positive_data: List[SamplerDatapoint] = []
    for (segment, var_to_obj) in datastores[datastore_idx]:
        option = segment.get_option()
        state = segment.states[0]
        if CFG.sampler_learning_use_goals:
            # Right now, we're making the assumption that all data is
            # demonstration data when we're learning samplers with goals.
            # In the future, we may weaken this assumption.
            goal = segment.get_goal()
        else:
            goal = None
        assert all(
            pre.predicate.holds(state, [var_to_obj[v] for v in pre.variables])
            for pre in preconditions)
        positive_data.append((state, var_to_obj, option, goal))

    # Populate all negative data.
    negative_data: List[SamplerDatapoint] = []

    if CFG.sampler_disable_classifier:
        # If we disable the classifier, then we never provide
        # negative examples, so that it always outputs 1.
        return positive_data, negative_data

    for idx, datastore in enumerate(datastores):
        for (segment, var_to_obj) in datastore:
            option = segment.get_option()
            state = segment.states[0]
            if CFG.sampler_learning_use_goals:
                # Right now, we're making the assumption that all data is
                # demonstration data when we're learning samplers with goals.
                # In the future, we may weaken this assumption.
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
                if len(negative_data
                       ) >= CFG.sampler_learning_max_negative_data:
                    # If we already have more negative examples
                    # than the maximum specified in the config,
                    # we don't add any more negative examples.
                    return positive_data, negative_data

                # If we are currently at the datastore that we're learning a
                # sampler for, and this datapoint matches the positive
                # grounding, this was already added to the positive data, so
                # we can continue.
                if idx == datastore_idx:
                    positive_grounding = [var_to_obj[var] for var in variables]
                    if grounding == positive_grounding:
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

    return positive_data, negative_data


@dataclass(frozen=True, eq=False, repr=False)
class _LearnedSampler:
    """A convenience class for holding the models underlying a learned
    sampler."""
    _classifier: BinaryClassifier
    _regressor: DistributionRegressor
    _variables: Sequence[Variable]
    _param_option: ParameterizedOption

    def sampler(self, state: State, goal: Set[GroundAtom],
                rng: np.random.Generator, objects: Sequence[Object]) -> Array:
        """The sampler corresponding to the given models.

        May be used as the _sampler field in an NSRT.
        """
        x_lst: List[Any] = [1.0]  # start with bias term
        sub = dict(zip(self._variables, objects))
        x_names = []
        for var in self._variables:
            x_lst.extend(state[sub[var]])
            x_names.append(sub[var])
        if CFG.sampler_learning_use_goals:
            # For goal-conditioned sampler learning, we currently make the
            # extremely limiting assumption that there is one goal atom, with
            # one goal object. This will not be true in most cases. This is a
            # placeholder for better methods to come.
            assert len(goal) == 1
            goal_atom = next(iter(goal))
            assert len(goal_atom.objects) == 1
            goal_obj = goal_atom.objects[0]
            x_lst.extend(state[goal_obj])  # add goal state

        np.set_printoptions(threshold=np.inf)

        x = np.array(x_lst)
        truncated_x = np.array(x_lst)[:-1]

        num_rejections = 0

        if CFG.sampler_disable_classifier:
            try:
                params = np.array(self._regressor.predict_sample(truncated_x, rng),
                                  dtype=self._param_option.params_space.dtype)
            except:
                params = np.array(self._regressor.predict_sample(x, rng),
                                  dtype=self._param_option.params_space.dtype)

            file_path = f"output_dist_direct_all.txt"

            # Open the file in append mode
            with open(file_path, "a") as file:
                # Iterate through the array and write each element to the file
                if self._param_option.name == "Pick":
                    file.write(f"({params[0]}, {params[1]})\n")  # Each item on a new line

            return params

        while num_rejections <= CFG.max_rejection_sampling_tries:
            # sampler_visualizer.visualize(self._regressor, x, rng)
            try:

                params = np.array(self._regressor.predict_sample(x, rng),
                                  dtype=self._param_option.params_space.dtype)
                if not CFG.sampler_disable_classifier and self._param_option.params_space.contains(params) and \
                        self._classifier.classify(np.r_[x, params]):
                    break

            except AssertionError:
                params = np.array(self._regressor.predict_sample(truncated_x, rng),
                                  dtype=self._param_option.params_space.dtype)
                if not CFG.sampler_disable_classifier and self._param_option.params_space.contains(params) and \
                        self._classifier.classify(np.r_[truncated_x, params]):
                    break

            num_rejections += 1


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
