"""A MiniGrid environment wrapping https://github.com/mpSchrader/gym-sokoban."""
import sys
from typing import Callable, ClassVar, Dict, List, Optional, Sequence, Set

import gymnasium as gym
import matplotlib
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, Image, Object, \
    Observation, Predicate, State, Type, Video

from minigrid.core.constants import (
    OBJECT_TO_IDX,
)
from minigrid.core.world_object import Ball as BallObj, Goal, Key as KeyObj, Box as BoxObj
from minigrid.wrappers import FullyObsWrapper

class MiniGridEnv(BaseEnv):
    """MiniGrid environment wrapping gym-sokoban."""

    name_to_enum: ClassVar[Dict[str, int]] = OBJECT_TO_IDX

    object_type = Type("obj", ["row", "column", "type", "state", "color"])
    location_type = Type("loc", ["row", "column", "type", "state", "color"])

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Predicates
        self._Above = Predicate("Above", [self.location_type, self.location_type],
                                self._Above_holds)
        self._Below = Predicate("Below", [self.location_type, self.location_type],
                                self._Below_holds)
        self._RightOf = Predicate("RightOf",
                                  [self.location_type, self.location_type],
                                  self._RightOf_holds)
        self._LeftOf = Predicate("LeftOf",
                                 [self.location_type, self.location_type],
                                 self._LeftOf_holds)
        self._ObjAbove = Predicate("ObjAbove", [self.object_type, self.location_type],
                                   self._Above_holds)
        self._ObjBelow = Predicate("ObjBelow", [self.object_type, self.location_type],
                                      self._Below_holds)
        self._ObjRightOf = Predicate("ObjRightOf",
                                        [self.object_type, self.location_type],
                                        self._RightOf_holds)
        self._ObjLeftOf = Predicate("ObjLeftOf",
                                        [self.object_type, self.location_type],
                                        self._LeftOf_holds)
        self._IsFacingUp = Predicate("IsFacingUp", [],
                                     self._IsFacingUp_holds)
        self._IsFacingDown = Predicate("IsFacingDown", [],
                                       self._IsFacingDown_holds)
        self._IsFacingLeft = Predicate("IsFacingLeft", [],
                                        self._IsFacingLeft_holds)
        self._IsFacingRight = Predicate("IsFacingRight", [],
                                        self._IsFacingRight_holds)
        self._IsNonGoalLoc = Predicate("IsNonGoalLoc", [self.location_type],
                                       self._IsNonGoalLoc_holds)
        self._Unknown = Predicate("Unknown", [self.location_type],
                                        self._Unknown_holds)
        self._Found = Predicate("Found", [self.location_type],
                                        self._Found_holds)
        self._ObjUnknown = Predicate("ObjUnknown", [self.object_type],
                                        self._Unknown_holds)
        self._ObjFound = Predicate("ObjFound", [self.object_type],
                                        self._Found_holds)
        self._AgentAt, self._At, self._IsGoal, self._IsBall, \
        self._IsKey, self._IsBox, self._IsRed, self._IsGreen, \
        self._IsBlue, self._IsPurple, self._IsYellow, self._IsGrey, \
        self._Holding, self._Near = self.get_goal_predicates()

        self.last_action = None

        # NOTE: we can change the level by modifying what we pass

        # into gym.make here.
        if CFG.minigrid_gym_fully_observable:
            self._gym_env = FullyObsWrapper(gym.make(CFG.minigrid_gym_name))
        else:
            self._gym_env = gym.make(CFG.minigrid_gym_name)

    @classmethod
    def get_goal_predicates(cls) -> list[Predicate]:
        """Defined public so that the perceiver can use it."""
        return [Predicate("AgentAt", [cls.location_type], cls._AgentAt_holds),
                Predicate("At", [cls.object_type, cls.location_type], cls._At_holds),
                Predicate("IsGoal", [cls.location_type], cls._IsGoal_holds),
                Predicate("IsBall", [cls.object_type], cls._IsBall_holds),
                Predicate("IsKey", [cls.object_type], cls._IsKey_holds),
                Predicate("IsBox", [cls.object_type], cls._IsBox_holds),
                Predicate("IsRed", [cls.object_type], cls._IsRed_holds),
                Predicate("IsGreen", [cls.object_type], cls._IsGreen_holds),
                Predicate("IsBlue", [cls.object_type], cls._IsBlue_holds),
                Predicate("IsPurple", [cls.object_type], cls._IsPurple_holds),
                Predicate("IsYellow", [cls.object_type], cls._IsYellow_holds),
                Predicate("IsGrey", [cls.object_type], cls._IsGrey_holds),
                Predicate("Holding", [cls.object_type], cls._Holding_holds),
                Predicate("Near", [cls.object_type, cls.object_type], cls._Near_holds)]


    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, train_or_test="train")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, train_or_test="test")

    @classmethod
    def get_name(cls) -> str:
        return "minigrid_env"

    def get_observation(self) -> Observation:
        return self._copy_observation(self._current_observation)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError("This env does not use Matplotlib")

    def render_state(self,
                     state: State,
                     task: EnvironmentTask,
                     action: Optional[Action] = None,
                     caption: Optional[str] = None) -> Video:
        raise NotImplementedError("A gym environment cannot render "
                                  "arbitrary states.")

    def render(self,
               action: Optional[Action] = None,
               caption: Optional[str] = None) -> Video:
        assert caption is None
        arr: Image = self._gym_env.get_frame()
        import matplotlib.pyplot as plt
        plt.imsave('visual_image.png', arr.astype('uint8'))
        return [arr]

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._AgentAt, self._At, self._Above, self._Below,
            self._RightOf, self._LeftOf, self._ObjAbove, self._ObjBelow,
            self._ObjRightOf, self._ObjLeftOf, self._IsGoal, self._IsNonGoalLoc,
            self._IsFacingUp, self._IsFacingDown, self._IsFacingLeft, self._IsFacingRight,
            self._Unknown, self._Found, self._IsBall, self._IsKey, self._IsBox, self._IsRed,
            self._IsGreen, self._IsBlue, self._IsPurple, self._IsYellow, self._IsGrey,
            self._Holding, self._Near, self._ObjUnknown, self._ObjFound
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._AgentAt, self._IsGoal}

    @property
    def types(self) -> Set[Type]:
        return {self.object_type, self.location_type}

    @property
    def action_space(self) -> Box:
        # One-hot encoding of discrete action space.
        num_actions = 7
        assert self._gym_env.action_space.n == num_actions  # type: ignore
        lowers = np.zeros(num_actions, dtype=np.float32)
        uppers = np.ones(num_actions, dtype=np.float32)
        return Box(lowers, uppers)

    def reset(self, train_or_test: str, task_idx: int) -> Observation:
        """Resets the current state to the train or test task initial state."""
        self._current_task = self.get_task(train_or_test, task_idx)
        self._current_observation = self._current_task.init_obs
        # We now need to reset the underlying gym environment to the correct
        # state.
        seed = utils.get_task_seed(train_or_test, task_idx)
        self._reset_initial_state_from_seed(seed)
        return self._copy_observation(self._current_observation)

    def simulate(self, state: State, action: Action) -> State:
        raise NotImplementedError("Simulate not implemented for gym envs. " +
                                  "Try using --bilevel_plan_without_sim True")

    def step(self, action: Action) -> Observation:
        # Convert our actions to their discrete action space.
        discrete_action = np.argmax(action.arr)

        goal_position = [
            y.cur_pos for x, y in enumerate(self._gym_env.grid.grid) if isinstance(y, Goal)
        ]
        self._current_observation = self._gym_env.step(discrete_action)
        self._gym_env.render()
        self.last_action = discrete_action
        self._current_observation[4]['last_action'] = self.last_action
        
        if CFG.minigrid_gym_render:
            # save frame to png
            visual = self._gym_env.get_frame()
            import matplotlib.pyplot as plt
            plt.imsave('render.png', visual.astype('uint8'))


        return self._copy_observation(self._current_observation)

    def goal_reached(self) -> bool:
        if len(self._current_observation) == 5:
            return self._current_observation[2]
        return False

    def _get_tasks(self, num: int,
                   train_or_test: str) -> List[EnvironmentTask]:
        tasks = []
        for task_idx in range(num):
            seed = utils.get_task_seed(train_or_test, task_idx)
            init_obs = self._reset_initial_state_from_seed(seed)
            goal_description = self._gym_env.mission
            task = EnvironmentTask(init_obs, goal_description)
            tasks.append(task)
        return tasks

    def _reset_initial_state_from_seed(self, seed: int) -> Observation:
        return self._gym_env.reset(seed=seed)

    @classmethod
    def _IsGoal_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._check_enum(state, objects, "goal")

    @classmethod
    def _IsBall_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._check_enum(state, objects, "ball")
    
    @classmethod
    def _IsKey_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._check_enum(state, objects, "key")

    @classmethod
    def _IsBox_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        return cls._check_enum(state, objects, "box")

    @classmethod
    def _IsRed_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "color") == 'red'

    @classmethod
    def _IsGreen_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "color") == 'green'

    @classmethod
    def _IsBlue_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "color") == 'blue'

    @classmethod
    def _IsPurple_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "color") == 'purple'

    @classmethod
    def _IsYellow_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "color") == 'yellow'

    @classmethod
    def _IsGrey_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return state.get(obj, "color") == 'grey'

    @classmethod
    def _IsNonGoalLoc_holds(cls, state: State,
                            objects: Sequence[Object]) -> bool:
        return cls._check_enum(state, objects, "empty")

    @classmethod
    def _At_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, _ = objects
        if cls._check_enum(state, [obj1], "agent"):
            return False
        return cls._check_spatial_relation(state, objects, 0, 0)

    @classmethod
    def _AgentAt_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        loc_obj, = objects
        obj = list(cls.get_objects_of_enum(state, "agent"))[0]
        if cls._check_enum(state, [obj], "agent"):
            return cls._check_spatial_relation(state, [obj, loc_obj], 0, 0)
        return False
    
    @classmethod
    def _Above_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, _ = objects
        if cls._check_enum(state, [obj1], "agent"):
            return False
        return cls._check_spatial_relation(state, objects, 1, 0)

    @classmethod
    def _Below_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, _ = objects
        if cls._check_enum(state, [obj1], "agent"):
            return False
        return cls._check_spatial_relation(state, objects, -1, 0)

    @classmethod
    def _RightOf_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, _ = objects
        if cls._check_enum(state, [obj1], "agent"):
            return False
        return cls._check_spatial_relation(state, objects, 0, -1)

    @classmethod
    def _LeftOf_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, _ = objects
        if cls._check_enum(state, [obj1], "agent"):
            return False
        return cls._check_spatial_relation(state, objects, 0, 1)

    @classmethod
    def _IsFacingRight_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj = list(cls.get_objects_of_enum(state, "agent"))[0]
        if cls._check_enum(state, [obj], "agent"):
            return int(state.get(obj, "state")) == 0
        return False
    
    @classmethod
    def _IsFacingDown_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj = list(cls.get_objects_of_enum(state, "agent"))[0]
        if cls._check_enum(state, [obj], "agent"):
            return int(state.get(obj, "state")) == 1
        return False
    
    @classmethod
    def _IsFacingLeft_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj = list(cls.get_objects_of_enum(state, "agent"))[0]
        if cls._check_enum(state, [obj], "agent"):
            return int(state.get(obj, "state")) == 2
        return False

    @classmethod
    def _IsFacingUp_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj = list(cls.get_objects_of_enum(state, "agent"))[0]
        if cls._check_enum(state, [obj], "agent"):
            return int(state.get(obj, "state")) == 3
        return False

    @classmethod
    def _Holding_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return int(state.get(obj, "state")) == 3

    @classmethod
    def _Near_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj1, ob2 = objects
        return cls._Above_holds(state, [obj1, ob2]) or \
                cls._Below_holds(state, [obj1, ob2]) or \
                cls._RightOf_holds(state, [obj1, ob2]) or \
                cls._LeftOf_holds(state, [obj1, ob2])
    
    @classmethod
    def _Unknown_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return int(state.get(obj, "state")) == -1
    
    @classmethod
    def _Found_holds(cls, state: State, objects: Sequence[Object]) -> bool:
        obj, = objects
        return int(state.get(obj, "state")) != -1

    @classmethod
    def get_objects_of_enum(cls, state: State, enum_name: str) -> Set[Object]:
        """Made public for use by perceiver."""
        return {
            o
            for o in state
            if int(state.get(o, "type")) == int(cls.name_to_enum[enum_name])
        }

    @classmethod
    def _check_spatial_relation(cls, state: State, objects: Sequence[Object],
                                dr: int, dc: int) -> bool:
        obj1, obj2 = objects
        obj1_r = int(state.get(obj1, "row"))
        obj1_c = int(state.get(obj1, "column"))
        obj2_r = int(state.get(obj2, "row"))
        obj2_c = int(state.get(obj2, "column"))
        if obj1_r == sys.maxsize or obj2_r == sys.maxsize or obj1_c == sys.maxsize or obj2_c == sys.maxsize:
            return False
        return ((obj1_r + dr) == obj2_r) and ((obj1_c + dc) == obj2_c)

    @classmethod
    def _check_enum(cls, state: State, objects: Sequence[Object],
                    enum_name: str) -> bool:
        obj, = objects
        obj_type = state.get(obj, "type")
        return int(obj_type) == int(cls.name_to_enum[enum_name])

    @classmethod
    def _is_static(cls, obj: Object, state: State) -> bool:
        return cls._IsGoal_holds(state, [obj]) or \
               cls._IsNonGoalLoc_holds(state, [obj])

    @classmethod
    def _is_dynamic(cls, obj: Object, state: State) -> bool:
        return not cls._is_static(obj, state)

    def _copy_observation(self, obs: Observation) -> Observation:
        return tuple(m.copy() if type(m) not in [bool, int, float] else m for m in obs)

    def get_event_to_action_fn(
            self) -> Callable[[State, matplotlib.backend_bases.Event], Action]:

        def _event_to_action(state: State,
                             event: matplotlib.backend_bases.Event) -> Action:
            del state  # unused
            #
            
            logging.info(
                "Controls: 0: left, 1: right, 2: forward, 3: pickup, 4: drop, 5: toggle, 6: done"
            )
            if event.key == "q":
                raise utils.HumanDemonstrationFailure("Human quit.")
            try:
                action_int = int(event.key)
            except ValueError:
                action_int = 0
            action_vec = self._action_space.low.copy()
            action_vec[action_int] = 1.0
            return Action(action_vec)

        return _event_to_action
