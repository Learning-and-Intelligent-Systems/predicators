"""An approach that trains a GNN mapping states, atoms, and goals to
options."""

import time
from collections import defaultdict
from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn
import torch.optim
from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.gnn_approach import GNNApproach
from predicators.nsrt_learning.segmentation import segment_trajectory
from predicators.option_model import create_option_model
from predicators.settings import CFG
from predicators.structs import Action, Array, Dataset, DummyOption, \
    GroundAtom, Object, ParameterizedOption, Predicate, State, Task, Type, \
    _Option


class GNNOptionPolicyApproach(GNNApproach):
    """Trains and uses a goal-conditioned GNN policy that produces options."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._sorted_options = sorted(self._initial_options,
                                      key=lambda o: o.name)
        self._option_model = create_option_model(CFG.option_model_name)
        self._max_option_objects = 0
        self._max_option_params = 0
        self._bce_loss = torch.nn.BCEWithLogitsLoss()
        self._crossent_loss = torch.nn.CrossEntropyLoss()
        self._mse_loss = torch.nn.MSELoss()

    def _generate_data_from_dataset(
        self, dataset: Dataset
    ) -> List[Tuple[State, Set[GroundAtom], Set[GroundAtom], _Option]]:
        data = []
        ground_atom_dataset = utils.create_ground_atom_dataset(
            dataset.trajectories, self._initial_predicates)
        # In this approach, we never learned any NSRTs, so we just call
        # segment_trajectory() to segment the given dataset.
        segmented_trajs = [
            segment_trajectory(traj) for traj in ground_atom_dataset
        ]
        for segment_traj, (ll_traj, _) in zip(segmented_trajs,
                                              ground_atom_dataset):
            if not ll_traj.is_demo:
                continue
            goal = self._train_tasks[ll_traj.train_task_idx].goal
            for segment in segment_traj:
                state = segment.states[0]  # the segment's initial state
                atoms = segment.init_atoms  # the segment's initial atoms
                target = segment.get_option()  # the segment's option
                data.append((state, atoms, goal, target))
        return data

    def _setup_output_specific_fields(
        self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom],
                               _Option]]
    ) -> None:
        # Go through the data, identifying the maximum number of option
        # objects and parameters.
        max_option_objects = 0
        max_option_params = 0
        for _, _, _, option in data:
            assert len(option.params.shape) == 1
            max_option_objects = max(max_option_objects, len(option.objects))
            max_option_params = max(max_option_params, option.params.shape[0])
        self._max_option_objects = max_option_objects
        self._max_option_params = max_option_params

    def _graphify_single_target(self, target: _Option, graph_input: Dict,
                                object_to_node: Dict) -> Dict:
        # First, copy over all unchanged fields.
        graph_target = {
            "n_node": graph_input["n_node"],
            "n_edge": graph_input["n_edge"],
            "edges": graph_input["edges"],
            "senders": graph_input["senders"],
            "receivers": graph_input["receivers"],
        }
        # Next, set up the target node features. The target is an _Option.
        object_mask = np.zeros((len(object_to_node), self._max_option_objects),
                               dtype=np.int64)
        for i, obj in enumerate(target.objects):
            object_mask[object_to_node[obj], i] = 1
        graph_target["nodes"] = object_mask
        # Finally, set up the target globals.
        option_index = self._sorted_options.index(target.parent)
        onehot_target = np.zeros(len(self._sorted_options))
        onehot_target[option_index] = 1
        assert len(target.params.shape) == 1
        params_target = np.zeros(self._max_option_params)
        params_target[:target.params.shape[0]] = target.params
        graph_target["globals"] = np.r_[onehot_target, params_target]
        return graph_target

    def _criterion(self, output: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        if self._max_option_objects == 0:
            return torch.tensor(0.0)
        return self._bce_loss(output, target)

    def _global_criterion(self, output: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        onehot_output, params_output = torch.split(  # type: ignore
            output, [len(self._sorted_options), self._max_option_params],
            dim=1)
        onehot_target, params_target = torch.split(  # type: ignore
            target, [len(self._sorted_options), self._max_option_params],
            dim=1)
        onehot_loss = self._crossent_loss(onehot_output,
                                          onehot_target.argmax(dim=1))
        if self._max_option_params > 0:
            params_loss = self._mse_loss(params_output, params_target)
        else:
            params_loss = torch.tensor(0.0)
        return onehot_loss + params_loss

    def _add_output_specific_fields_to_save_info(self, info: Dict) -> None:
        info["max_option_objects"] = self._max_option_objects
        info["max_option_params"] = self._max_option_params

    def _load_output_specific_fields_from_save_info(self, info: Dict) -> None:
        self._max_option_objects = info["max_option_objects"]
        self._max_option_params = info["max_option_params"]

    def _extract_output_from_graph(
        self, graph_output: Dict, object_to_node: Dict
    ) -> Tuple[ParameterizedOption, List[Object], Array]:
        """The output is a parameterized option from self._sorted_options,
        discrete object arguments, and continuous arguments."""
        node_to_object = {v: k for k, v in object_to_node.items()}
        type_to_node = defaultdict(set)
        for obj, node in object_to_node.items():
            type_to_node[obj.type.name].add(node)
        # Extract parameterized option and continuous parameters.
        onehot_output, params = np.split(graph_output["globals"],
                                         [len(self._sorted_options)])
        param_opt = self._sorted_options[np.argmax(onehot_output)]
        # Pad and clip parameters.
        params = params[:param_opt.params_space.shape[0]]
        params = params.clip(param_opt.params_space.low,
                             param_opt.params_space.high)
        # Extract objects, making sure to enforce the typing constraints.
        objects = []
        for i, obj_type in enumerate(param_opt.types):
            scores = graph_output["nodes"][:, i]
            allowed_idxs = type_to_node[obj_type.name]
            for j in range(len(scores)):
                if j not in allowed_idxs:
                    scores[j] = float("-inf")  # set its score to be really bad
            if np.max(scores) == float("-inf"):
                # If all scores are -inf, we failed to select an object.
                raise ApproachFailure(
                    "GNN option policy could not select an object")
            objects.append(node_to_object[np.argmax(scores)])
        return param_opt, objects, params

    @classmethod
    def get_name(cls) -> str:
        return "gnn_option_policy"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        assert self._gnn is not None, "Learning hasn't happened yet!"
        if CFG.gnn_option_policy_solve_with_shooting:
            return self._solve_with_shooting(task, timeout)
        return self._solve_without_shooting(task)

    def _solve_without_shooting(self, task: Task) -> Callable[[State], Action]:
        cur_option = DummyOption
        memory: Dict = {}  # optionally updated by predict()

        def _policy(state: State) -> Action:
            atoms = utils.abstract(state, self._initial_predicates)
            nonlocal cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                param_opt, objects, params_mean = self._predict(
                    state, atoms, task.goal, memory)
                # Just use the mean parameters to ground the option.
                cur_option = param_opt.ground(objects, params_mean)
                if not cur_option.initiable(state):
                    raise ApproachFailure(
                        "GNN option policy chose a non-initiable option")
            act = cur_option.policy(state)
            return act

        return _policy

    def _solve_with_shooting(self, task: Task,
                             timeout: int) -> Callable[[State], Action]:
        start_time = time.perf_counter()
        memory: Dict = {}  # optionally updated by predict()
        # Keep trying until the timeout.
        while time.perf_counter() - start_time < timeout:
            total_num_act = 0
            state = task.init
            plan: List[_Option] = []
            # A single shooting try goes up to the environment's horizon.
            while total_num_act < CFG.horizon:
                if task.goal_holds(state):
                    # We found a plan that achieves the goal under the
                    # option model, so return it.
                    option_policy = utils.option_plan_to_policy(plan)

                    def _policy(s: State) -> Action:
                        try:
                            return option_policy(s)
                        except utils.OptionExecutionFailure as e:
                            raise ApproachFailure(e.args[0], e.info)

                    return _policy
                atoms = utils.abstract(state, self._initial_predicates)
                param_opt, objects, params_mean = self._predict(
                    state, atoms, task.goal, memory)
                low = param_opt.params_space.low
                high = param_opt.params_space.high
                # Sample an initiable option.
                for _ in range(CFG.gnn_option_policy_shooting_max_samples):
                    params = np.array(self._rng.normal(
                        params_mean, CFG.gnn_option_policy_shooting_variance),
                                      dtype=np.float32)
                    params = np.clip(params, low, high)
                    opt = param_opt.ground(objects, params)
                    if opt.initiable(state):
                        break
                else:
                    break  # out of the while loop for this shooting try
                plan.append(opt)
                # Use the option model to determine the next state.
                try:
                    state, num_act = \
                        self._option_model.get_next_state_and_num_actions(
                            state, opt)
                except utils.EnvironmentFailure:
                    break
                # If num_act is zero, that means that the option is stuck in
                # the state, so we should break to avoid infinite loops.
                if num_act == 0:
                    break
                total_num_act += num_act
                # Break early if we have timed out.
                if time.perf_counter() - start_time < timeout:
                    break
        raise ApproachTimeout("Shooting timed out!")
