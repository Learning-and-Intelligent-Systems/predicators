"""An approach that learns NSRTs, then learns a metacontroller to select ground
NSRTs sequentially on evaluation tasks.

For each ground NSRT in this sequence, we sample continuous parameters
until the expected atoms check is satisfied. This approach can be
understood as an ablation of bilevel planning that uses a
metacontroller, instead of task planning, to generate skeletons.
"""

import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.gnn_approach import GNNApproach
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Action, Dataset, DummyOption, \
    GroundAtom, LowLevelTrajectory, ParameterizedOption, Predicate, Segment, \
    State, Task, Type, _GroundNSRT, _Option


class GNNMetacontrollerApproach(NSRTLearningApproach, GNNApproach):
    """GNNMetacontrollerApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        NSRTLearningApproach.__init__(self, initial_predicates,
                                      initial_options, types, action_space,
                                      train_tasks)
        GNNApproach.__init__(self, initial_predicates, initial_options, types,
                             action_space, train_tasks)
        self._sorted_nsrts: List[NSRT] = []
        self._max_nsrt_objects = 0
        self._bce_loss = torch.nn.BCEWithLogitsLoss()
        self._crossent_loss = torch.nn.CrossEntropyLoss()

    def _extract_target_from_data(self, segment: Segment,
                                  segment_traj: List[Segment],
                                  ll_traj: LowLevelTrajectory) -> _GroundNSRT:
        seg_option = segment.get_option()
        objects = list(segment.states[0])
        poss_ground_nsrts = []
        # Find the ground NSRT that matches the given segment.
        for nsrt in self._sorted_nsrts:
            if nsrt.option != seg_option.parent:
                continue
            for ground_nsrt in utils.all_ground_nsrts(nsrt, objects):
                # If the option objects, preconditions, and effects
                # match, add this ground NSRT to poss_ground_nsrts.
                if ground_nsrt.option_objs != seg_option.objects:
                    continue
                if not ground_nsrt.preconditions.issubset(segment.init_atoms):
                    continue
                atoms = utils.apply_operator(ground_nsrt, segment.init_atoms)
                if not atoms.issubset(segment.final_atoms):
                    continue
                poss_ground_nsrts.append(ground_nsrt)
        # Verify that there is exactly one grounding that matches.
        assert len(poss_ground_nsrts) == 1
        return poss_ground_nsrts[0]

    def _setup_output_specific_fields(
        self, data: List[Tuple[State, Set[GroundAtom], Set[GroundAtom],
                               _GroundNSRT]]
    ) -> None:
        max_nsrt_objects = 0
        for _, _, _, ground_nsrt in data:
            max_nsrt_objects = max(max_nsrt_objects, len(ground_nsrt.objects))
        assert max_nsrt_objects > 0
        self._max_nsrt_objects = max_nsrt_objects

    def _graphify_single_target(self, target: _GroundNSRT, graph_input: Dict,
                                object_to_node: Dict) -> Dict:
        # First, copy over all unchanged fields.
        graph_target = {
            "n_node": graph_input["n_node"],
            "n_edge": graph_input["n_edge"],
            "edges": graph_input["edges"],
            "senders": graph_input["senders"],
            "receivers": graph_input["receivers"],
        }
        # Next, set up the target node features. The target is a _GroundNSRT.
        object_mask = np.zeros((len(object_to_node), self._max_nsrt_objects),
                               dtype=np.int64)
        for i, obj in enumerate(target.objects):
            object_mask[object_to_node[obj], i] = 1
        graph_target["nodes"] = object_mask
        # Finally, set up the target globals.
        nsrt_index = self._sorted_nsrts.index(target.parent)
        onehot_target = np.zeros(len(self._sorted_nsrts))
        onehot_target[nsrt_index] = 1
        graph_target["globals"] = onehot_target
        return graph_target

    def _criterion(self, output: torch.Tensor,
                   target: torch.Tensor) -> torch.Tensor:
        return self._bce_loss(output, target)

    def _global_criterion(self, output: torch.Tensor,
                          target: torch.Tensor) -> torch.Tensor:
        return self._crossent_loss(output, target.argmax(dim=1))

    def _add_output_specific_fields_to_save_info(self, info: Dict) -> None:
        info["max_nsrt_objects"] = self._max_nsrt_objects

    def _load_output_specific_fields_from_save_info(self, info: Dict) -> None:
        self._max_nsrt_objects = info["max_nsrt_objects"]

    def _extract_output_from_graph(self, graph_output: Dict,
                                   object_to_node: Dict) -> _GroundNSRT:
        """The output is a ground NSRT."""
        node_to_object = {v: k for k, v in object_to_node.items()}
        type_to_node = defaultdict(set)
        for obj, node in object_to_node.items():
            type_to_node[obj.type.name].add(node)
        # Extract NSRT.
        output = graph_output["globals"]
        nsrt = self._sorted_nsrts[np.argmax(output)]
        # Extract objects, making sure to enforce the typing constraints.
        types = [param.type for param in nsrt.parameters]
        objects = []
        for i, obj_type in enumerate(types):
            scores = graph_output["nodes"][:, i]
            allowed_idxs = type_to_node[obj_type.name]
            for j in range(len(scores)):
                if j not in allowed_idxs:
                    scores[j] = float("-inf")  # set its score to be really bad
            if np.max(scores) == float("-inf"):  # type: ignore
                # If all scores are -inf, we failed to select an object.
                raise ApproachFailure(
                    "GNN metacontroller could not select an object")
            objects.append(node_to_object[np.argmax(scores)])
        return nsrt.ground(objects)

    @classmethod
    def get_name(cls) -> str:
        return "gnn_metacontroller"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        assert self._gnn is not None, "Learning hasn't happened yet!"
        cur_option = DummyOption

        def _policy(state: State) -> Action:
            atoms = utils.abstract(state, self._initial_predicates)
            nonlocal cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                ground_nsrt = self._predict(state, atoms, task.goal)
                cur_option = self._sample_option_from_nsrt(
                    ground_nsrt, state, atoms, task.goal)
            act = cur_option.policy(state)
            return act

        return _policy

    def _sample_option_from_nsrt(self, ground_nsrt: _GroundNSRT, state: State,
                                 atoms: Set[GroundAtom],
                                 goal: Set[GroundAtom]) -> _Option:
        """Given a ground NSRT, try invoking its sampler repeatedly until we
        find an option that produces the expected next atoms under the ground
        NSRT."""
        for _ in range(CFG.gnn_metacontroller_max_samples):
            # Invoke the ground NSRT's sampler to produce an option.
            opt = ground_nsrt.sample_option(state, goal, self._rng)
            if not opt.initiable(state):
                # The option is not initiable. Continue on to the next sample.
                continue
            next_state, _ = self._option_model.get_next_state_and_num_actions(
                state, opt)
            next_atoms = utils.abstract(next_state, self._initial_predicates)
            expected_next_atoms = utils.apply_operator(ground_nsrt, atoms)
            if not expected_next_atoms.issubset(next_atoms):
                # Some expected atom is not achieved. Continue on to the
                # next sample.
                continue
            return opt
        raise ApproachFailure("GNN metacontroller could not sample an option")

    def load(self, online_learning_cycle: Optional[int]) -> None:
        NSRTLearningApproach.load(self, online_learning_cycle)
        self._sorted_nsrts = sorted(self._nsrts)
        assert self._sorted_nsrts
        del self._nsrts  # henceforth, we'll use self._sorted_nsrts
        GNNApproach.load(self, online_learning_cycle)

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First learn NSRTs.
        NSRTLearningApproach.learn_from_offline_dataset(self, dataset)
        self._sorted_nsrts = sorted(self._nsrts)
        assert self._sorted_nsrts
        del self._nsrts  # henceforth, we'll use self._sorted_nsrts
        # Then learn the GNN metacontroller.
        logging.info("Learning metacontroller...")
        GNNApproach.learn_from_offline_dataset(self, dataset)
