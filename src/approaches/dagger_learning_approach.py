"""An approach that imlements DAgger for option learning."""

from typing import Set, List, Optional, Tuple, Callable, Sequence
import dill as pkl
import numpy as np
from gym.spaces import Box
from predicators.src import utils
from predicators.src.approaches import NSRTLearningApproach, \
    ApproachTimeout, ApproachFailure, RandomOptionsApproach
from predicators.src.structs import State, Predicate, ParameterizedOption, \
    Type, Task, Dataset, GroundAtom, LowLevelTrajectory, InteractionRequest, \
    InteractionResult, Action, GroundAtomsHoldQuery, GroundAtomsHoldResponse, \
    Query, DemonstrationQuery, PathToStateQuery
from predicators.src.torch_models import LearnedPredicateClassifier, \
    MLPClassifier
from predicators.src.settings import CFG
from typing import Set, List, Sequence, Optional
import dill as pkl
from gym.spaces import Box
from predicators.src.approaches import TAMPApproach
from predicators.src.structs import Dataset, NSRT, ParameterizedOption, \
    Predicate, Type, Task, LowLevelTrajectory
from predicators.src.nsrt_learning.nsrt_learning_main import \
    learn_nsrts_from_data
from predicators.src.settings import CFG
from predicators.src import utils
from predicators.src.envs import create_env, BaseEnv

class DaggerLearningApproach(NSRTLearningApproach):
    """An approach that implements DAgger for option learning."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self.online_learning_cycle = 0

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # save trajectories so that we can add more through DAgger
        self.dataset = dataset
        self._learn_nsrts(self.dataset.trajectories, online_learning_cycle=None)

    # def _make_query_policy(self, train_task_idx: int) -> Callable[[State], Query]:
    #     def _query_policy(s: State) -> DemonstrationQuery:
    #         del s  # not used
    #         return DemonstrationQuery(train_task_idx)
    #     return _query_policy

    def _make_query_policy(self, goal_state: State) -> Callable[[State], Query]:
        def _query_policy(s: State) -> PathToStateQuery:
            del s  # not used
            return PathToStateQuery(goal_state)
        return _query_policy

    def _make_termination_fn(self, goal: Set[GroundAtom]) -> Callable[[State], bool]:
        def _termination_fn(s: State) -> bool:
            return all(goal_atom.holds(s) for goal_atom in goal)
        return _termination_fn

    def get_interaction_requests(self) -> List[InteractionRequest]:
        requests = []

        for i in range(len(self._train_tasks)):
            task = self._train_tasks[i]
            try:
                _act_policy = self.solve(task, CFG.timeout)
            except (ApproachTimeout, ApproachFailure) as e:
                partial_refinements = e.info.get("partial_refinements")
                _, plan = max(partial_refinements, key=lambda x: len(x[1]))
                _act_policy = utils.option_plan_to_policy(plan)
            # random_options_approach = RandomOptionsApproach(self._get_current_predicates(), self._initial_options, self._types, self._action_space, self._train_tasks)
            # _act_policy = random_options_approach.solve(task, CFG.timeout)
            request = InteractionRequest(
                train_task_idx = i,
                act_policy = _act_policy,
                query_policy = self._make_query_policy(),
                termination_function = self._make_termination_fn(task.goal)
            )
            requests.append(request)
            break
        return requests

    def learn_from_interaction_results(self, results: Sequence[InteractionResult]) -> None:
        # make videos of expert trajectories for debugging
        for result in results:  # one result per training task
            responses = result.responses  # one response per state in trajectory
            print("LENGTH OF RESPONSES: ", len(responses))
            for res in responses:
                teacher_traj = res.teacher_traj
                # if teacher_traj is not None:
                #     print("TEACHER TRAJ STATES: ", teacher_traj.states)
                video: Video = []
                env = create_env(CFG.env)
                for s in teacher_traj.states:
                    dummy_task = Task(s, set())
                    video.extend(env.render(s, dummy_task))
                video_prefix = utils.get_config_path_str()
                n = np.random.randint(0, 1000)
                outfile = f"{video_prefix}_teacher_{n}.mp4"
                utils.save_video(outfile, video)

        for result in results:  # one result per training task
            responses = result.responses  # one response per state in trajectory
            # print("responses: ", responses)
            # for res in responses[:1]:
            responses = [responses[0], responses[3], responses[4]]
            for res in responses:
                teacher_traj = res.teacher_traj
                if teacher_traj is None:  # oracle approach shouldn't fail, but...
                    continue
                for act in res.teacher_traj.actions:
                    act.unset_option()
                traj = LowLevelTrajectory(res.teacher_traj.states,
                                          res.teacher_traj.actions,
                                          _is_demo=True,
                                          _train_task_idx=res.teacher_traj.train_task_idx)
                self.dataset.append(traj)
        self._learn_nsrts(self.dataset.trajectories, online_learning_cycle=self.online_learning_cycle)
        self.online_learning_cycle += 1
