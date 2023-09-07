"""Various utils for running RL approaches.

This file is adapted from the MAPLE codebase (https://github.com/UT-
Austin-RPL/maple) by Nasiriany et. al.
"""

import abc
import logging
from collections import OrderedDict
from numbers import Number
from typing import Callable, Dict, List, Set

import numpy as np
import torch
from gym.spaces import Box, Discrete, Tuple

from predicators import utils
from predicators.settings import CFG
from predicators.structs import Action, Array, GroundAtom, Predicate, State, \
    _GroundNSRT, _Option


def add_prefix(log_dict: OrderedDict, prefix: str, divider=''):
    with_prefix = OrderedDict()
    for key, val in log_dict.items():
        with_prefix[prefix + divider + key] = val
    return with_prefix


def identity(x):
    return x


_str_to_activation = {
    'identity': identity,
    'relu': torch.nn.ReLU(),
    'tanh': torch.nn.Tanh(),
    'leaky_relu': torch.nn.LeakyReLU(),
    'sigmoid': torch.nn.Sigmoid(),
    'selu': torch.nn.SELU(),
    'softplus': torch.nn.Softplus(),
}


def activation_from_string(string):
    return _str_to_activation[string]


def soft_update_from_to(source, target, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def copy_model_params_from_to(source, target):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def maximum_2d(t1, t2):
    # noinspection PyArgumentList
    return torch.max(
        torch.cat((t1.unsqueeze(2), t2.unsqueeze(2)), dim=2),
        dim=2,
    )[0].squeeze(2)


def kronecker_product(t1, t2):
    """Computes the Kronecker product between two tensors See
    https://en.wikipedia.org/wiki/Kronecker_product."""
    t1_height, t1_width = t1.size()
    t2_height, t2_width = t2.size()
    out_height = t1_height * t2_height
    out_width = t1_width * t2_width

    # TODO(vitchyr): see if you can use expand instead of repeat
    tiled_t2 = t2.repeat(t1_height, t1_width)
    expanded_t1 = (t1.unsqueeze(2).unsqueeze(3).repeat(
        1, t2_height, t2_width, 1).view(out_height, out_width))

    return expanded_t1 * tiled_t2


def alpha_dropout(
    x,
    p=0.05,
    alpha=-1.7580993408473766,
    fixedPointMean=0,
    fixedPointVar=1,
    training=False,
):
    keep_prob = 1 - p
    if keep_prob == 1 or not training:
        return x
    a = np.sqrt(
        fixedPointVar /
        (keep_prob *
         ((1 - keep_prob) * pow(alpha - fixedPointMean, 2) + fixedPointVar)))
    b = fixedPointMean - a * (keep_prob * fixedPointMean +
                              (1 - keep_prob) * alpha)
    keep_prob = 1 - p

    random_tensor = keep_prob + torch.rand(x.size())
    binary_tensor = torch.floor(random_tensor)
    x = x.mul(binary_tensor)
    ret = x + alpha * (1 - binary_tensor)
    ret.mul_(a).add_(b)
    return ret


def alpha_selu(x, training=False):
    return alpha_dropout(torch.nn.SELU(x), training=training)


def double_moments(x, y):
    """Returns the first two moments between x and y.

    Specifically, for each vector x_i and y_i in x and y, compute their
    outer-product. Flatten this resulting matrix and return it.

    The first moments (i.e. x_i and y_i) are included by appending a `1` to x_i
    and y_i before taking the outer product.
    :param x: Shape [batch_size, feature_x_dim]
    :param y: Shape [batch_size, feature_y_dim]
    :return: Shape [batch_size, (feature_x_dim + 1) * (feature_y_dim + 1)
    """
    batch_size, x_dim = x.size()
    _, y_dim = x.size()
    x = torch.cat((x, torch.ones(batch_size, 1)), dim=1)
    y = torch.cat((y, torch.ones(batch_size, 1)), dim=1)
    x_dim += 1
    y_dim += 1
    x = x.unsqueeze(2)
    y = y.unsqueeze(1)

    outer_prod = (x.expand(batch_size, x_dim, y_dim) *
                  y.expand(batch_size, x_dim, y_dim))
    return outer_prod.view(batch_size, -1)


def batch_diag(diag_values, diag_mask=None):
    batch_size, dim = diag_values.size()
    if diag_mask is None:
        diag_mask = torch.diag(torch.ones(dim))
    batch_diag_mask = diag_mask.unsqueeze(0).expand(batch_size, dim, dim)
    batch_diag_values = diag_values.unsqueeze(1).expand(batch_size, dim, dim)
    return batch_diag_values * batch_diag_mask


def batch_square_vector(vector, M):
    """Compute x^T M x."""
    vector = vector.unsqueeze(2)
    return torch.bmm(torch.bmm(vector.transpose(2, 1), M), vector).squeeze(2)


def fanin_init(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    return tensor.data.uniform_(-bound, bound)


def fanin_init_weights_like(tensor):
    size = tensor.size()
    if len(size) == 2:
        fan_in = size[0]
    elif len(size) > 2:
        fan_in = np.prod(size[1:])
    else:
        raise Exception("Shape must be have dimension at least 2.")
    bound = 1. / np.sqrt(fan_in)
    new_tensor = FloatTensor(tensor.size())
    new_tensor.uniform_(-bound, bound)
    return new_tensor


def almost_identity_weights_like(tensor):
    """
    Set W = I + lambda * Gaussian no
    :param tensor:
    :return:
    """
    shape = tensor.size()
    init_value = np.eye(*shape)
    init_value += 0.01 * np.random.rand(*shape)
    return FloatTensor(init_value)


def clip1(x):
    return torch.clamp(x, -1, 1)


def compute_conv_output_size(h_in, w_in, kernel_size, stride, padding=0):
    h_out = (h_in + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    w_out = (w_in + 2 * padding - (kernel_size - 1) - 1) / stride + 1
    return int(np.floor(h_out)), int(np.floor(w_out))


def compute_deconv_output_size(h_in, w_in, kernel_size, stride, padding=0):
    h_out = (h_in - 1) * stride - 2 * padding + kernel_size
    w_out = (w_in - 1) * stride - 2 * padding + kernel_size
    return int(np.floor(h_out)), int(np.floor(w_out))


def compute_conv_layer_sizes(h_in, w_in, kernel_sizes, strides, paddings=None):
    if paddings == None:
        for kernel, stride in zip(kernel_sizes, strides):
            h_in, w_in = compute_conv_output_size(h_in, w_in, kernel, stride)
            print('Output Size:', (h_in, w_in))
    else:
        for kernel, stride, padding in zip(kernel_sizes, strides, paddings):
            h_in, w_in = compute_conv_output_size(h_in,
                                                  w_in,
                                                  kernel,
                                                  stride,
                                                  padding=padding)
            print('Output Size:', (h_in, w_in))


def compute_deconv_layer_sizes(h_in,
                               w_in,
                               kernel_sizes,
                               strides,
                               paddings=None):
    if paddings == None:
        for kernel, stride in zip(kernel_sizes, strides):
            h_in, w_in = compute_deconv_output_size(h_in, w_in, kernel, stride)
            print('Output Size:', (h_in, w_in))
    else:
        for kernel, stride, padding in zip(kernel_sizes, strides, paddings):
            h_in, w_in = compute_deconv_output_size(h_in,
                                                    w_in,
                                                    kernel,
                                                    stride,
                                                    padding=padding)
            print('Output Size:', (h_in, w_in))


"""
GPU wrappers
"""

_use_gpu = False
device = None


def set_gpu_mode(mode, gpu_id=0):
    global _use_gpu
    global device
    global _gpu_id
    _gpu_id = gpu_id
    _use_gpu = mode
    device = torch.device("cuda:" + str(gpu_id) if _use_gpu else "cpu")


def gpu_enabled():
    return _use_gpu


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


# noinspection PyPep8Naming
def FloatTensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.FloatTensor(*args, **kwargs, device=torch_device)


def from_numpy(*args, **kwargs):
    return torch.from_numpy(*args, **kwargs).float().to(device)


def to_numpy(tensor):
    return tensor.to('cpu').detach().numpy()


def randint(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randint(*sizes, **kwargs, device=torch_device)


def zeros(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros(*sizes, **kwargs, device=torch_device)


def ones(*sizes, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones(*sizes, **kwargs, device=torch_device)


def ones_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.ones_like(*args, **kwargs, device=torch_device)


def randn(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.randn(*args, **kwargs, device=torch_device)


def zeros_like(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.zeros_like(*args, **kwargs, device=torch_device)


def tensor(*args, torch_device=None, **kwargs):
    if torch_device is None:
        torch_device = device
    return torch.tensor(*args, **kwargs, device=torch_device)


def normal(*args, **kwargs):
    return torch.normal(*args, **kwargs).to(device)


def get_dim(space):
    if isinstance(space, Box):
        return space.low.size
    elif isinstance(space, Discrete):
        return space.n
    elif isinstance(space, Tuple):
        return sum(get_dim(subspace) for subspace in space.spaces)
    elif hasattr(space, 'flat_dim'):
        return space.flat_dim
    else:
        raise TypeError("Unknown space: {}".format(space))


# Various replay buffer implementations
class ReplayBuffer(object, metaclass=abc.ABCMeta):
    """A class used to save and replay data."""

    @abc.abstractmethod
    def add_sample(self, observation, action, reward, next_observation,
                   terminal, **kwargs):
        """Add a transition tuple."""
        pass

    @abc.abstractmethod
    def terminate_episode(self):
        """Let the replay buffer know that the episode has terminated in case
        some special book-keeping has to happen.

        :return:
        """
        pass

    @abc.abstractmethod
    def num_steps_can_sample(self, **kwargs):
        """
        :return: # of unique items that can be sampled.
        """
        pass

    def add_path(self, path):
        """Add a path to the replay buffer.

        This default implementation naively goes through every step, but you
        may want to optimize this.

        NOTE: You should NOT call "terminate_episode" after calling add_path.
        It's assumed that this function handles the episode termination.

        :param path: Dict like one outputted by maple.samplers.util.rollout
        """
        for i, (obs, action, reward, next_obs, terminal, agent_info,
                env_info) in enumerate(
                    zip(
                        path["observations"],
                        path["actions"],
                        path["rewards"],
                        path["next_observations"],
                        path["terminals"],
                        path["agent_infos"],
                        path["env_infos"],
                    )):
            self.add_sample(
                observation=obs,
                action=action,
                reward=reward,
                next_observation=next_obs,
                terminal=terminal,
                agent_info=agent_info,
                env_info=env_info,
            )
        self.terminate_episode()

    def add_paths(self, paths):
        for path in paths:
            self.add_path(path)

    @abc.abstractmethod
    def random_batch(self, batch_size):
        """Return a batch of size `batch_size`.

        :param batch_size:
        :return:
        """
        pass

    def get_diagnostics(self):
        return {}

    def get_snapshot(self):
        return {}

    def end_epoch(self, epoch):
        return


class SimpleReplayBuffer(ReplayBuffer):

    def __init__(
        self,
        max_replay_buffer_size,
        observation_dim,
        action_dim,
        env_info_sizes,
        replace=True,
    ):
        self._observation_dim = observation_dim
        self._action_dim = action_dim
        self._max_replay_buffer_size = max_replay_buffer_size
        self._observations = np.zeros(
            (max_replay_buffer_size, observation_dim))
        # It's a bit memory inefficient to save the observations twice,
        # but it makes the code *much* easier since you no longer have to
        # worry about termination conditions.
        self._next_obs = np.zeros((max_replay_buffer_size, observation_dim))
        self._actions = np.zeros((max_replay_buffer_size, action_dim))
        # Make everything a 2D np array to make it easier for other code to
        # reason about the shape of the data
        self._rewards = np.zeros((max_replay_buffer_size, 1))
        # self._terminals[i] = a terminal was received at time i
        self._terminals = np.zeros((max_replay_buffer_size, 1), dtype='uint8')
        # Define self._env_infos[key][i] to be the return value of env_info[key]
        # at time i
        self._env_infos = {}
        for key, size in env_info_sizes.items():
            self._env_infos[key] = np.zeros((max_replay_buffer_size, size))
        self._env_info_keys = env_info_sizes.keys()

        self._replace = replace

        self._top = 0
        self._size = 0

    def add_sample(self, observation, action, reward, next_observation,
                   terminal, env_info, **kwargs):
        self._observations[self._top] = observation
        self._actions[self._top] = action
        self._rewards[self._top] = reward
        self._terminals[self._top] = terminal
        self._next_obs[self._top] = next_observation

        for key in self._env_info_keys:
            self._env_infos[key][self._top] = env_info[key]
        self._advance()

    def terminate_episode(self):
        pass

    def _advance(self):
        self._top = (self._top + 1) % self._max_replay_buffer_size
        if self._size < self._max_replay_buffer_size:
            self._size += 1

    def random_batch(self, batch_size):
        indices = np.random.choice(self._size,
                                   size=batch_size,
                                   replace=self._replace
                                   or self._size < batch_size)
        if not self._replace and self._size < batch_size:
            logging.warning(
                'Replace was set to false, but is temporarily set to true because batch size is larger than current size of replay.'
            )
        batch = dict(
            observations=self._observations[indices],
            actions=self._actions[indices],
            rewards=self._rewards[indices],
            terminals=self._terminals[indices],
            next_observations=self._next_obs[indices],
        )
        for key in self._env_info_keys:
            assert key not in batch.keys()
            batch[key] = self._env_infos[key][indices]
        return batch

    def rebuild_env_info_dict(self, idx):
        return {key: self._env_infos[key][idx] for key in self._env_info_keys}

    def batch_env_info_dict(self, indices):
        return {
            key: self._env_infos[key][indices]
            for key in self._env_info_keys
        }

    def num_steps_can_sample(self):
        return self._size

    def get_diagnostics(self):
        return OrderedDict([('size', self._size)])


class EnvReplayBuffer(SimpleReplayBuffer):

    def __init__(self,
                 max_replay_buffer_size,
                 ob_space_size,
                 action_space_size,
                 env_info_sizes=dict()):
        """
        :param max_replay_buffer_size:
        :param env:
        """
        super().__init__(max_replay_buffer_size=max_replay_buffer_size,
                         observation_dim=ob_space_size,
                         action_dim=action_space_size,
                         env_info_sizes=env_info_sizes)

    def add_sample(self, observation, action, reward, terminal,
                   next_observation, **kwargs):
        return super().add_sample(observation=observation,
                                  action=action,
                                  reward=reward,
                                  next_observation=next_observation,
                                  terminal=terminal,
                                  env_info={},
                                  **kwargs)


def create_stats_ordered_dict(
    name,
    data,
    stat_prefix=None,
    always_show_all_stats=True,
    exclude_max_min=True,
):
    if stat_prefix is not None:
        name = "{}{}".format(stat_prefix, name)
    if isinstance(data, Number):
        return OrderedDict({name: data})

    if len(data) == 0:
        return OrderedDict()

    if isinstance(data, tuple):
        ordered_dict = OrderedDict()
        for number, d in enumerate(data):
            sub_dict = create_stats_ordered_dict(
                "{0}_{1}".format(name, number),
                d,
            )
            ordered_dict.update(sub_dict)
        return ordered_dict

    if isinstance(data, list):
        try:
            iter(data[0])
        except TypeError:
            pass
        else:
            data = np.concatenate(data)

    if (isinstance(data, np.ndarray) and data.size == 1
            and not always_show_all_stats):
        return OrderedDict({name: float(data)})

    stats = OrderedDict([
        (name + ' Mean', np.mean(data)),
        (name + ' Std', np.std(data)),
    ])
    if not exclude_max_min:
        stats[name + ' Max'] = np.max(data)
        stats[name + ' Min'] = np.min(data)
    return stats


def convert_policy_action_to_ground_option(
        policy_action: Array, ground_nsrts: List[_GroundNSRT],
        discrete_actions_size: int, continuous_actions_size: int) -> _Option:
    """Converts an action (in the form of a vector) output by a HybridSACAgent
    into an _Option that can be executed in the actual environment.

    Importantly, the ground_nsrts input list must have the correct order
    so that indexing it with the output from policy_action will work
    correctly.
    """
    ground_nsrt, continuous_params_for_option = get_ground_nsrt_and_params_from_maple(
        policy_action, ground_nsrts, discrete_actions_size,
        continuous_actions_size)
    logging.debug(
        f"[RL] Running {ground_nsrt.name}({ground_nsrt.objects}) with clipped params {continuous_params_for_option}"
    )
    output_ground_option = ground_nsrt.option.ground(
        ground_nsrt.option_objs, continuous_params_for_option)
    return output_ground_option


def _filter_batch(np_batch):
    for k, v in np_batch.items():
        if v.dtype == np.bool_:
            yield k, v.astype(int)
        else:
            yield k, v


def np_to_pytorch_batch(np_batch: Dict[str, Array]):
    """Provided a dict representing a numpy batch, convert to a dict
    representing a pytorch batch."""
    return {
        k: torch.from_numpy(x).float().to(device)
        for k, x in _filter_batch(np_batch)
        if x.dtype != np.dtype('O')  # ignore object (e.g. dictionaries)
    }


def env_state_to_maple_input(state: State) -> Array:
    """Convert an input state into a vector that can be input into MAPLE."""
    return state.vec(sorted(list(state)))


def get_ground_nsrt_and_params_from_maple(
        policy_action, ground_nsrts, discrete_actions_size,
        continuous_actions_size):  #-> Tuple[_GroundNSRT, Array]:
    """Gets the ground NSRT as well as the continuous params output by MAPLE
    separately so we can choose to override either."""
    assert policy_action.shape[
        0] == discrete_actions_size + continuous_actions_size
    # Discrete actions output should be 0 everywhere except for one place.
    discrete_actions_output = policy_action[:discrete_actions_size]
    discrete_action_idx = np.argmax(discrete_actions_output)
    assert discrete_actions_output[discrete_action_idx] == 1.0
    ground_nsrt = ground_nsrts[discrete_action_idx]
    continuous_params_output = policy_action[-continuous_actions_size:]
    continuous_params_for_option = continuous_params_output[:ground_nsrt.option
                                                            .params_space.
                                                            shape[0]]

    # if not ground_nsrt.option.params_space.contains(continuous_params_for_option.astype(np.float32)):
    #     import ipdb; ipdb.set_trace()

    # Clip these continuous params to ensure they're within the bounds of the
    # parameter space.
    continuous_params_for_option = np.clip(
        continuous_params_for_option, ground_nsrt.option.params_space.low,
        ground_nsrt.option.params_space.high)

    # if all(continuous_params_for_option != unclipped_params):
    #     import ipdb; ipdb.set_trace()

    return (ground_nsrt, continuous_params_for_option)


def make_executable_maple_policy(
        policy, ground_nsrts: List[_GroundNSRT], observation_size: int,
        discrete_actions_size: int,
        continuous_actions_size: int) -> Callable[[State], Action]:
    curr_option = None
    num_curr_option_steps = 0

    def _rollout_rl_policy(state: State) -> Action:
        """Execute the option policy until we get an option termination or
        timeout (i.e, we exceed the max steps for the option) and then get a
        new output from the model."""
        nonlocal policy, curr_option, num_curr_option_steps, observation_size,\
            discrete_actions_size, continuous_actions_size
        state_vec = env_state_to_maple_input(state)
        if curr_option is None or (curr_option is not None
                                   and curr_option.terminal(state)):
            # We need to produce a new ground option from the network.
            assert state_vec.shape[0] == observation_size
            num_curr_option_steps = 0
            policy_action = policy.get_action(state_vec)[0]
            curr_option = convert_policy_action_to_ground_option(
                policy_action, ground_nsrts, discrete_actions_size,
                continuous_actions_size)

            if not curr_option.initiable(state):
                num_curr_option_steps = 0
                raise utils.OptionExecutionFailure(
                    "Unsound option policy.",
                    info={"last_failed_option": curr_option})

        if CFG.max_num_steps_option_rollout is not None and \
            num_curr_option_steps >= CFG.max_num_steps_option_rollout:
            raise utils.OptionTimeoutFailure(
                "Exceeded max option steps.",
                info={"last_failed_option": curr_option})

        num_curr_option_steps += 1

        return curr_option.policy(state)

    return _rollout_rl_policy


def make_executable_qfunc_only_policy(qf1, qf2,
                                      ground_nsrts: List[_GroundNSRT],
                                      ground_nsrt_to_idx: Dict[_GroundNSRT,
                                                               int],
                                      observation_size: int,
                                      discrete_actions_size: int,
                                      continuous_actions_size: int,
                                      predicates: Set[Predicate],
                                      goal: Set[GroundAtom],
                                      rng: np.random.Generator,
                                      epsilon: float) -> Callable[[State], Action]:

    curr_option = None
    num_curr_option_steps = 0

    def _rollout_policy_based_on_qfunc(state: State) -> Action:
        """This function comes up with an action by:
        1. Abstracting state
        2. Finding the set of ground nsrts whose preconditions hold in this state
        3. For each of these ground NSRTs, taking a fixed number of samples from the base sampler
        4. Scoring all of these according to the learned q-function
        5. Picking the ground NSRT and sample with the greatest q-value. This is turned into an
        option.

        If the current option is not None, we just continue taking actions from that option.

        Note that with probability epsilon, we sample a random option instead of the best one.
        """
        nonlocal qf1, qf2, curr_option, num_curr_option_steps, observation_size,\
            discrete_actions_size, continuous_actions_size, predicates, goal, rng

        max_q_val = -float('inf')
        max_q_val_option = None

        if curr_option is None or (curr_option is not None
                                   and curr_option.terminal(state)):
            curr_atoms = utils.abstract(state, predicates)
            all_applicable_ground_nsrts = sorted(
                list(utils.get_applicable_operators(ground_nsrts, curr_atoms)))
            state_vec = torch.from_numpy(env_state_to_maple_input(state)).type(torch.float32)
            randomly_chosen_nsrt = rng.choice(all_applicable_ground_nsrts)
            randomly_chosen_sample_idx = rng.choice(CFG.active_sampler_learning_num_samples)
            randomly_chosen_option = None
            for ground_nsrt in all_applicable_ground_nsrts:
                samples = [
                    ground_nsrt._sampler(state, goal, rng, ground_nsrt.objects)
                    for _ in range(CFG.active_sampler_learning_num_samples)
                ]
                for i, sample in enumerate(samples):
                    discrete_action = np.zeros(discrete_actions_size)
                    discrete_action[ground_nsrt_to_idx[ground_nsrt]] = 1.0
                    continuous_action = np.zeros(continuous_actions_size)
                    continuous_action[:ground_nsrt.option.params_space.
                                      shape[0]] = np.array(sample)
                    maple_action = np.concatenate(
                        (discrete_action, continuous_action), axis=0)
                    q_val = torch.min(
                        qf1(state_vec.unsqueeze(0), torch.from_numpy(maple_action).unsqueeze(0).type(torch.float32)),
                        qf2(state_vec.unsqueeze(0), torch.from_numpy(maple_action).unsqueeze(0).type(torch.float32)),
                    ).item()

                    if q_val > max_q_val:
                        max_q_val = q_val
                        max_q_val_option = ground_nsrt.option.ground(
                            ground_nsrt.option_objs, sample)
                        
                    if ground_nsrt == randomly_chosen_nsrt and i == randomly_chosen_sample_idx:
                        randomly_chosen_option = ground_nsrt.option.ground(ground_nsrt.option_objs, sample)

                    # Debugging
                    # print(f"Option: {ground_nsrt.option.ground(ground_nsrt.option_objs, sample)}, with q-val {q_val}")

            curr_option = max_q_val_option

            if rng.random() < epsilon:
                curr_option = randomly_chosen_option

            logging.info(f"Running option {curr_option}")
            # import ipdb; ipdb.set_trace()

            if not curr_option.initiable(state):
                num_curr_option_steps = 0
                raise utils.OptionExecutionFailure(
                    "Unsound option policy.",
                    info={"last_failed_option": curr_option})

        if CFG.max_num_steps_option_rollout is not None and \
            num_curr_option_steps >= CFG.max_num_steps_option_rollout:
            raise utils.OptionTimeoutFailure(
                "Exceeded max option steps.",
                info={"last_failed_option": curr_option})

        num_curr_option_steps += 1

        return curr_option.policy(state)

    return _rollout_policy_based_on_qfunc
