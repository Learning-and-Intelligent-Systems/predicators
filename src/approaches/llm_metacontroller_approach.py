"""An approach that uses a large language model as a metacontroller.

Example command:
    python src/main.py --approach llm_metacontroller \
        --env pddl_easy_delivery_procedural_tasks --seed 0 \
        --strips_learner oracle --num_train_tasks 3
"""

from typing import Callable, List, Sequence, Set, Union

import numpy as np
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches import ApproachFailure
from predicators.src.approaches.nsrt_learning_approach import \
    NSRTLearningApproach
from predicators.src.settings import CFG
from predicators.src.structs import Action, Dataset, DummyOption, GroundAtom, \
    ParameterizedOption, Predicate, State, Task, Type, _GroundNSRT, _Option


class LLMMetacontrollerApproach(NSRTLearningApproach):
    """LLMMetacontrollerApproach definition."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # The prefix used for prompting the LLM.
        self._prompt_prefix = ""  # overwritten when data is received
        # Ground NSRTs reset with every call to solve.
        self._all_ground_nsrts: List[_GroundNSRT] = []

    @classmethod
    def get_name(cls) -> str:
        return "llm_metacontroller"

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:
        cur_option = DummyOption
        # Ground all NSRTs with the objects in the task.
        task_objects = list(task.init)
        self._all_ground_nsrts = [
            ground_nsrt for nsrt in sorted(self._nsrts)
            for ground_nsrt in utils.all_ground_nsrts(nsrt, task_objects)
        ]
        # The history of previously selected options, used for prompting.
        option_history: List[_Option] = []

        def _policy(state: State) -> Action:
            atoms = utils.abstract(state, self._initial_predicates)
            nonlocal cur_option
            if cur_option is DummyOption or cur_option.terminal(state):
                ground_nsrt = self._predict(atoms, task.goal, option_history)
                cur_option = self._sample_option_from_nsrt(
                    ground_nsrt, state, atoms, task.goal)
                option_history.append(cur_option)
            act = cur_option.policy(state)
            return act

        return _policy

    def _predict(self, atoms: Set[GroundAtom], goal: Set[GroundAtom],
                 option_history: Sequence[_Option]) -> _GroundNSRT:
        """Select a next _GroundNSRT to execute."""
        new_prompt = self._create_prompt(atoms, goal, option_history)
        prompt = self._prompt_prefix + new_prompt
        # Score each potential next ground NSRT using the LLM.
        best_next_ground_nsrt = None
        best_score = -np.inf
        succs = utils.get_applicable_operators(self._all_ground_nsrts, atoms)
        for ground_nsrt in succs:
            # Higher is better.
            ground_nsrt_str = self._option_or_nsrt_to_str(ground_nsrt)
            score = self._score(prompt, ground_nsrt_str)
            if score > best_score:
                best_next_ground_nsrt = ground_nsrt
                best_score = score
        # No ground NSRTs applied.
        if best_next_ground_nsrt is None:
            raise ApproachFailure("LLM metacontroller reached a dead end.")
        return best_next_ground_nsrt

    def _score(self, prompt: str, candidate: str) -> float:
        """Score a candidate addition to the prompt, with higher better."""

        # TODO: Implement me!!!!! (Using LLM)
        return self._rng.uniform()

    def _sample_option_from_nsrt(self, ground_nsrt: _GroundNSRT, state: State,
                                 atoms: Set[GroundAtom],
                                 goal: Set[GroundAtom]) -> _Option:
        """Given a ground NSRT, try invoking its sampler repeatedly until we
        find an option that produces the expected next atoms under the ground
        NSRT."""
        # TODO: factor out code in common with GNN metacontroller approach.
        for _ in range(CFG.llm_metacontroller_max_samples):
            # Invoke the ground NSRT's sampler to produce an option.
            opt = ground_nsrt.sample_option(state, goal, self._rng)
            if not opt.initiable(state):
                # The option is not initiable. Continue on to the next sample.
                continue
            try:
                next_state, _ = \
                    self._option_model.get_next_state_and_num_actions(state,
                                                                      opt)
            except utils.EnvironmentFailure:
                continue
            expected_next_atoms = utils.apply_operator(ground_nsrt, atoms)
            if not all(a.holds(next_state) for a in expected_next_atoms):
                # Some expected atom is not achieved. Continue on to the
                # next sample.
                continue
            return opt
        raise ApproachFailure("LLM metacontroller could not sample an option")

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # First learn NSRTs.
        super().learn_from_offline_dataset(dataset)
        # Then parse the data into the prompting format expected by the LLM.
        self._prompt_prefix = self._data_to_prompt_prefix(dataset)

    def _data_to_prompt_prefix(self, dataset: Dataset) -> str:
        # In this approach, we learned NSRTs, so we just use the segmented
        # trajectories that NSRT learning returned to us.
        prompts = []
        assert len(self._segmented_trajs) == len(dataset.trajectories)
        for segment_traj, ll_traj in zip(self._segmented_trajs,
                                         dataset.trajectories):
            if not ll_traj.is_demo:
                continue
            init = segment_traj[0].init_atoms
            goal = self._train_tasks[ll_traj.train_task_idx].goal
            seg_options = []
            for segment in segment_traj:
                assert segment.has_option()
                seg_options.append(segment.get_option())
            prompt = self._create_prompt(init, goal, seg_options)
            prompts.append(prompt)
        return "\n\n".join(prompts) + "\n\n"

    def _create_prompt(self, init: Set[GroundAtom], goal: Set[GroundAtom],
                       options: Sequence[_Option]) -> str:
        init_str = "\n  ".join(map(str, sorted(init)))
        goal_str = "\n  ".join(map(str, sorted(goal)))
        options_str = "\n  ".join(map(self._option_or_nsrt_to_str, options))
        prompt = f"""
(:init
  {init_str}
)
(:goal
  {goal_str}
)
solution:
  {options_str}"""
        return prompt

    @staticmethod
    def _option_or_nsrt_to_str(option: Union[_Option, _GroundNSRT]) -> str:
        objects_str = ", ".join(map(str, option.objects))
        return f"{option.name}({objects_str})"
