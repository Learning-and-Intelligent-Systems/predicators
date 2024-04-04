"""An approach that trains a GNN mapping states, atoms, and goals to
actions."""

from typing import Callable, Dict, List, Set, Tuple

import numpy as np
import torch
import torch.nn
import torch.optim
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches.gnn_approach import GNNApproach
from predicators.src.structs import Action, Dataset, GroundAtom, \
    ParameterizedOption, Predicate, State, Task, Type


class GNNActionPolicyApproach(GNNApproach):
    """Trains and uses a goal-conditioned GNN policy that produces actions."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        del self._initial_options  # ensure that options aren't used!
        assert len(self._action_space.shape) == 1
        self._act_dims = self._action_space.shape[0]
        self._mse_loss = torch.nn.MSELoss()

    def _generate_data_from_dataset(
        self, dataset: Dataset
    ) -> List[Tuple[State, Set[GroundAtom], Set[GroundAtom], Action]]:
        data = []
        for ll_traj in dataset.trajectories:
            if not ll_traj.is_demo:
                continue
            goal = self._train_tasks[ll_traj.train_task_idx].goal
            for i in range(len(ll_traj.states) - 1):
                state = ll_traj.states[i]
                atoms = utils.abstract(state, self._initial_predicates)
                target = ll_traj.actions[i]
                data.append((state, atoms, goal, target))
        return data

    def _setup_output_specific_fields(
        self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom],
                               Action]]
    ) -> None:
        pass  # nothing to do here

    def _graphify_single_target(self, target: Action, graph_input: Dict,
                                object_to_node: Dict) -> Dict:
        # First, copy over all unchanged fields.
        graph_target = {
            "n_node": graph_input["n_node"],
            "n_edge": graph_input["n_edge"],
            "nodes": graph_input["nodes"],
            "edges": graph_input["edges"],
            "senders": graph_input["senders"],
            "receivers": graph_input["receivers"],
        }
        # Finally, set up the target globals. The target is an Action.
        graph_target["globals"] = target.arr
        return graph_target

    def _criterion(self, output: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        return torch.tensor(0.0)  # unused

    def _global_criterion(self, output: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        return self._mse_loss(output, target)

    def _add_output_specific_fields_to_save_info(self, info: Dict) -> None:
        pass  # nothing to do here

    def _load_output_specific_fields_from_save_info(self, info: Dict) -> None:
        pass  # nothing to do here

    def _extract_output_from_graph(self, graph_output: Dict,
                                   object_to_node: Dict) -> Action:
        """The output is an Action."""
        arr = graph_output["globals"]
        arr = np.clip(arr, self._action_space.low, self._action_space.high)
        return Action(arr)

    @classmethod
    def get_name(cls) -> str:
        return "gnn_action_policy"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        assert self._gnn is not None, "Learning hasn't happened yet!"

        def _policy(state: State) -> Action:
            atoms = utils.abstract(state, self._initial_predicates)
            act = self._predict(state, atoms, task.goal)
            return act

        return _policy
