"""Toy cluttered table domain. This environment is created to test our
planner's ability to handle failures reported by the environment.
"""

from typing import List, Set, Sequence, Dict
import numpy as np
from gym.spaces import Box
from predicators.src.envs import BaseEnv, EnvironmentFailure
from predicators.src.structs import Type, Predicate, State, Task, \
    ParameterizedOption, Object, Action, GroundAtom, Image, Array
from predicators.src.settings import CFG


class ClutteredTableEnv(BaseEnv):
    """Toy cluttered table domain.
    """
    def __init__(self) -> None:
        super().__init__()
        # Types
        self._can_type = Type(
            "can", ["pose_x", "pose_y", "radius", "is_grasped", "is_trashed"])
        # Predicates
        self._HandEmpty = Predicate(
            "HandEmpty", [], self._HandEmpty_holds)
        self._Holding = Predicate(
            "Holding", [self._can_type], self._Holding_holds)
        # Options
        self._Grasp = ParameterizedOption(
            "Grasp", [self._can_type], params_space=Box(0, 1, (4,)),
            _policy=lambda s, o, p: Action(p),  # action is just parameter
            _initiable=lambda s, o, p: True,  # can be run from anywhere
            _terminal=lambda s, o, p: True)  # always 1 timestep
        self._Dump = ParameterizedOption(
            "Dump", [], params_space=Box(0, 1, (0,)),
            _policy=lambda s, o, p: Action(
                np.array([-1.0, -1.0, -1.0, -1.0], dtype=np.float32)),
            _initiable=lambda s, o, p: True,  # can be run from anywhere
            _terminal=lambda s, o, p: True)  # always 1 timestep
        # Objects
        self._cans = []
        for i in range(max(CFG.cluttered_table_num_cans_train,
                           CFG.cluttered_table_num_cans_test)):
            self._cans.append(Object(f"can{i}", self._can_type))

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        # Figure out which can is currently grasped, if any.
        grasped_can = None
        for can in state:
            if state.get(can, "is_grasped") > 0.5:
                assert grasped_can is None, "Multiple cans grasped?"
                assert state.get(can, "is_trashed") < 0.5, \
                    "Grasped a can that has been trashed?"
                grasped_can = can
        if np.all(action.arr == -1):
            # Handle dumping action.
            if grasped_can is not None:
                next_state.set(grasped_can, "pose_x", -999)
                next_state.set(grasped_can, "pose_y", -999)
                next_state.set(grasped_can, "is_grasped", 0.0)
                next_state.set(grasped_can, "is_trashed", 1.0)
            return next_state
        # Handle grasping action.
        if grasped_can is not None:
            return next_state  # can't grasp while already grasping
        start_x, start_y, end_x, end_y = action.arr
        desired_can = None
        for can in state:
            this_x = state.get(can, "pose_x")
            this_y = state.get(can, "pose_y")
            this_radius = state.get(can, "radius")
            if np.linalg.norm([end_x-this_x,
                               end_y-this_y]) < this_radius:  # type: ignore
                assert desired_can is None
                desired_can = can
        if desired_can is None:
            return next_state  # end point wasn't at any can
        vec1 = np.array([end_x-start_x, end_y-start_y])
        # Handle collision checking.
        colliding_can = None
        colliding_can_max_dist = float("-inf")
        for can in state:
            if can == desired_can:
                continue
            this_x = state.get(can, "pose_x")
            this_y = state.get(can, "pose_y")
            vec2 = np.array([end_x-this_x, end_y-this_y])
            angle = np.arccos(
                vec1.dot(vec2) / (np.linalg.norm(vec1)*
                                  np.linalg.norm(vec2)))  # type: ignore
            if angle < CFG.cluttered_table_collision_angle_thresh:
                dist = np.linalg.norm(vec2)  # type: ignore
                if dist > colliding_can_max_dist:
                    colliding_can_max_dist = dist
                    colliding_can = can
        if colliding_can is not None:
            raise EnvironmentFailure("collision", {colliding_can})
        # No collisions, update state and return.
        next_state.set(desired_can, "is_grasped", 1.0)
        return next_state

    def get_train_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_train_tasks,
                               train_or_test="train")

    def get_test_tasks(self) -> List[Task]:
        return self._get_tasks(num=CFG.num_test_tasks,
                               train_or_test="test")

    @property
    def predicates(self) -> Set[Predicate]:
        return {self._HandEmpty, self._Holding}

    @property
    def types(self) -> Set[Type]:
        return {self._can_type}

    @property
    def options(self) -> Set[ParameterizedOption]:
        return {self._Grasp, self._Dump}

    @property
    def action_space(self) -> Box:
        # The action_space is 4-dimensional. The first two dimensions are the
        # start point of the vector corresponding to the grasp approach. The
        # last two dimensions are the end point. Dumping is a special action
        # where all 4 dimensions are -1.
        return Box(-1, 1, (4,))

    def render(self, state: State) -> Image:
        raise NotImplementedError

    def _get_tasks(self, num: int, train_or_test: str) -> List[Task]:
        tasks = []
        goal1 = {GroundAtom(self._Holding, [self._cans[0]])}
        goal2 = {GroundAtom(self._Holding, [self._cans[1]])}
        goal3 = {GroundAtom(self._Holding, [self._cans[2]])}
        goals = [goal1, goal2, goal3]
        for i in range(num):
            tasks.append(Task(self._create_initial_state(train_or_test),
                              goals[i%len(goals)]))
        return tasks

    def _create_initial_state(self, train_or_test: str) -> State:
        data: Dict[Object, Array] = {}
        assert train_or_test in ("train", "test")
        if train_or_test == "train":
            num_cans = CFG.cluttered_table_num_cans_train
            rng = self._train_rng
        elif train_or_test == "test":
            num_cans = CFG.cluttered_table_num_cans_test
            rng = self._test_rng
        radius = CFG.cluttered_table_can_radius
        for i in range(num_cans):
            can = self._cans[i]
            while True:
                # multiply radius by 4 to keep cans away from edge of table
                pose = np.array(rng.uniform(radius*4, 1.0-radius*4, size=2),
                                dtype=np.float32)
                if not self._any_intersection(pose, radius, data):
                    break
            # [pose_x, pose_y, radius, is_grasped, is_trashed]
            data[can] = np.array([pose[0], pose[1], radius, 0.0, 0.0])
        return State(data)

    @staticmethod
    def _HandEmpty_holds(state: State, objects: Sequence[Object]) -> bool:
        assert not objects
        for can in state:
            if state.get(can, "is_grasped") > 0.5:
                return False
        return True

    @staticmethod
    def _Holding_holds(state: State, objects: Sequence[Object]) -> bool:
        can, = objects
        return state.get(can, "is_grasped") > 0.5

    @staticmethod
    def _any_intersection(pose: Array, radius: float,
                          data: Dict[Object, Array]) -> bool:
        for other in data:
            other_feats = data[other]
            other_x = other_feats[0]
            other_y = other_feats[1]
            other_radius = other_feats[2]
            distance = np.linalg.norm([other_x-pose[0],
                                       other_y-pose[1]])  # type: ignore
            if distance <= (radius+other_radius):
                return True
        return False
