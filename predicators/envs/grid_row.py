"""Toy environment for testing active sampler learning."""

from typing import List, Optional, Sequence, Set

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, EnvironmentTask, GroundAtom, Object, \
    Predicate, State, Type


class GridRowEnv(BaseEnv):
    """An environment where a robot is in a discrete 1D grid.

    It needs to turn on a light switch in the first grid cell and then
    navigate to the last grid cell. Turning on the light switch requires
    learning a sampler.
    """

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)

        # Types
        self._robot_type = Type("robot", ["x"])
        self._cell_type = Type("cell", ["x"])
        self._light_type = Type("light", ["level", "target", "x"])

        # Predicates
        self._RobotInCell = Predicate("RobotInCell",
                                      [self._robot_type, self._cell_type],
                                      self._In_holds)
        self._LightInCell = Predicate("LightInCell",
                                      [self._light_type, self._cell_type],
                                      self._In_holds)
        self._LightOn = Predicate("LightOn", [self._light_type],
                                  self._LightOn_holds)
        self._LightOff = Predicate("LightOff", [self._light_type],
                                   self._LightOff_holds)
        self._Adjacent = Predicate("Adjacent",
                                   [self._cell_type, self._cell_type],
                                   self._Adjacent_holds)

        # Static objects (always exist no matter the settings).
        self._robot = Object("robby", self._robot_type)
        self._light = Object("light", self._light_type)
        self._cells = [
            Object(f"cell{i}", self._cell_type)
            for i in range(CFG.grid_row_num_cells)
        ]
        self._cell_to_neighbors = {
            self._cells[0]: {self._cells[1]},
            self._cells[-1]: {self._cells[-2]},
            **{
                self._cells[t]: {self._cells[t - 1], self._cells[t + 1]}
                for t in range(1,
                               len(self._cells) - 1)
            }
        }

    @classmethod
    def get_name(cls) -> str:
        return "grid_row"

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        dx, dlight = action.arr
        # Apply dlight if we're in the same cell as the light.
        robot_cells = [
            c for c in self._cells if self._In_holds(state, [self._robot, c])
        ]
        assert len(robot_cells) == 1
        robot_cell = robot_cells[0]
        light_cells = [
            c for c in self._cells if self._In_holds(state, [self._light, c])
        ]
        assert len(light_cells) == 1
        light_cell = light_cells[0]
        if robot_cell == light_cell:
            new_light_level = np.clip(
                state.get(self._light, "level") + dlight, 0.0, 1.0)
            next_state.set(self._light, "level", new_light_level)
        # Apply dx to robot.
        new_x = np.clip(
            state.get(self._robot, "x") + dx, 0.0, len(self._cells))
        next_state.set(self._robot, "x", new_x)
        return next_state

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._RobotInCell, self._LightInCell, self._LightOn,
            self._LightOff, self._Adjacent
        }

    @property
    def goal_predicates(self) -> Set[Predicate]:
        return {self._LightOn}

    @property
    def types(self) -> Set[Type]:
        return {self._robot_type, self._cell_type, self._light_type}

    @property
    def action_space(self) -> Box:
        # dx, dlight
        return Box(-np.inf, np.inf, (2, ))

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        raise NotImplementedError

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        # There is only one goal in this environment: to turn the light on.
        goal = {GroundAtom(self._LightOn, [self._light])}
        # The only variation in the initial state is the light target level.
        tasks: List[EnvironmentTask] = []
        while len(tasks) < num:
            state_dict = {
                self._robot: {
                    "x": 0.5,
                },
                self._light: {
                    "x": len(self._cells) - 0.5,
                    "level": 0.0,
                    "target": rng.uniform(0.5, 1.0),
                },
            }
            for i, cell in enumerate(self._cells):
                state_dict[cell] = {"x": i + 0.5}
            state = utils.create_state_from_dict(state_dict)
            tasks.append(EnvironmentTask(state, goal))
        return tasks

    def _In_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        x1 = state.get(obj1, "x")
        try:
            x2 = state.get(obj2, "x")
        except:
            import ipdb;ipdb.set_trace()
        # Threshold not important because dx is always discrete.
        return abs(x1 - x2) < 1e-3

    def _LightOn_holds(self, state: State, objects: Sequence[Object]) -> bool:
        light, = objects
        level = state.get(light, "level")
        target = state.get(light, "target")
        return abs(level - target) < 0.1

    def _LightOff_holds(self, state: State, objects: Sequence[Object]) -> bool:
        return not self._LightOn_holds(state, objects)

    def _Adjacent_holds(self, state: State, objects: Sequence[Object]) -> bool:
        # This is a better definition, but since Adjacent is a bottleneck
        # for speed, we use a hackier and faster definition.
        obj1, obj2 = objects
        # x1 = state.get(obj1, "x")
        # x2 = state.get(obj2, "x")
        # dist = abs(x1 - x2)
        # # Threshold not important because dx is always discrete.
        # return abs(dist - 1.0) < 1e-3
        del state  # not used
        try:
            return obj1 in self._cell_to_neighbors[obj2]
        except:
            import ipdb; ipdb.set_trace()


class GridRowDoorEnv(GridRowEnv):
    """Simple variant on GridRow where there is also a door."""

    # Properties for rendering
    cell_width = 1
    robot_width = 0.5
    robot_height = 0.75
    light_width = 0.25
    _robot_type = Type("robot", ["x"])
    _cell_type = Type("cell", ["x"])
    _light_type = Type("light", ["level", "target", "x"])
    _door_type = Type(
            "door",
            ["x", "move_key", "move_target", "turn_key", "turn_target"])
    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        # self._door_type = Type(
        #     "door",
        #     ["x", "move_key", "move_target", "turn_key", "turn_target"])
        # self._door = Object("door", self._door_type)

        # self._DoorInCell = Predicate("DoorInCell",
        #                              [self._door_type, self._cell_type],
        #                              self._In_holds)

    @classmethod
    def get_name(cls) -> str:
        return "grid_row_door"

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> \
                matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1)
        plt.xlim([0, len(self._cells)])
        plt.ylim([0, 1])

        # Draw the cells.
        for i in range(len(self._cells)):
            rect = plt.Rectangle((i, 0),
                                 self.cell_width,
                                 self.cell_width,
                                 edgecolor="gray",
                                 facecolor="gray")
            ax.add_patch(rect)

        # Draw light, door, and robot.
        if self._LightOn_holds(state, (self._light, )):
            light = plt.Rectangle(
                (len(self._cells) - (self.cell_width + \
                                     self.light_width) / 2.0,
                 1 - self.light_width),
                self.light_width,
                self.light_width,
                edgecolor="yellow",
                facecolor="yellow")
        else:
            light = plt.Rectangle(
                (len(self._cells) - (self.cell_width + \
                                     self.light_width) / 2.0,
                 1 - self.light_width),
                self.light_width,
                self.light_width,
                edgecolor="white",
                facecolor="white")
        ax.add_patch(light)
        for door in state:
            if not door.is_instance(self._door_type):
                continue
            door_pos = state.get(door, "x")
            door_move_key = state.get(door, "move_key")
            door_move_key_target = state.get(door, "move_target")
            door_turn_key = state.get(door, "turn_key")
            door_turn_key_target = state.get(door, "turn_target")
            if (door_move_key_target - 0.1 <= door_move_key <= \
                door_move_key_target + 0.1 and door_turn_key_target - 0.1 \
                    <= door_turn_key <= door_turn_key_target + 0.1):
                draw_door = plt.Rectangle((door_pos - self.cell_width / 2.0, 0),
                                    self.cell_width,
                                    self.cell_width,
                                    edgecolor="gray",
                                    facecolor="gray")
            else:
                draw_door = plt.Rectangle((door_pos - self.cell_width / 2.0, 0),
                                    self.cell_width,
                                    self.cell_width,
                                    edgecolor="darkgoldenrod",
                                    facecolor="darkgoldenrod")
            ax.add_patch(draw_door)
        robot_pos = state.get(self._robot, "x")
        robot = plt.Rectangle((robot_pos - self.robot_width / 2.0, 0),
                              self.robot_width,
                              self.robot_height,
                              edgecolor="red",
                              facecolor="red")
        ax.add_patch(robot)
        return fig

    @property
    def predicates(self) -> Set[Predicate]:
        return {
            self._RobotInCell, self._LightInCell, self._LightOn,
            self._LightOff, self._Adjacent
        }

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._cell_type, self._light_type,
            self._door_type
        }

    @property
    def action_space(self) -> Box:
        # dx, dlight, dmove, dturn
        return Box(-np.inf, np.inf, (4, ))

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        self._cells = [
            Object(f"cell{i}", self._cell_type)
            for i in range(CFG.test_grid_row_num_cells)
        ]
        self._cell_to_neighbors = {
            self._cells[0]: {self._cells[1]},
            self._cells[-1]: {self._cells[-2]},
            **{
                self._cells[t]: {self._cells[t - 1], self._cells[t + 1]}
                for t in range(1,
                               len(self._cells) - 1)
            }
        }
        num=CFG.num_test_tasks
        rng=self._test_rng
        goal = {GroundAtom(self._LightOn, [self._light])}
        tasks: List[EnvironmentTask] = []

        door_list: List[List[Object]] = []
        for door_num in range(CFG.num_doors):
            door = Object(f"door{door_num}", self._door_type)
            door_list.append(door)

        while len(tasks) < num:
            state_dict = {
                self._robot: {
                    "x": 0.5,
                },
                # Note: light level and door locations are fixed for now
                # in order to maintain consistency across train and test
                self._light: {
                    "x": len(self._cells) - 0.5,
                    "level": 0.0,
                    "target": 0.75,
                },
            }
            for i, cell in enumerate(self._cells):
                state_dict[cell] = {"x": i + 0.5}
            for i, door in enumerate(door_list):
                state_dict[door]={
                    "x": i*3+2.5,
                    "move_key": 0.0,
                    "move_target": 0.5,
                    "turn_key": 0.0,
                    "turn_target": 0.75
                }
            state = utils.create_state_from_dict(state_dict)
            tasks.append(EnvironmentTask(state, goal))
        return tasks


    def _reset_test_cells(self):
        self._cells = [
            Object(f"cell{i}", self._cell_type)
            for i in range(CFG.test_grid_row_num_cells)
        ]
        self._cell_to_neighbors = {
            self._cells[0]: {self._cells[1]},
            self._cells[-1]: {self._cells[-2]},
            **{
                self._cells[t]: {self._cells[t - 1], self._cells[t + 1]}
                for t in range(1,
                               len(self._cells) - 1)
            }
        }

    def _reset_cells(self):
        self._cells = [
            Object(f"cell{i}", self._cell_type)
            for i in range(CFG.grid_row_num_cells)
        ]
        self._cell_to_neighbors = {
            self._cells[0]: {self._cells[1]},
            self._cells[-1]: {self._cells[-2]},
            **{
                self._cells[t]: {self._cells[t - 1], self._cells[t + 1]}
                for t in range(1,
                               len(self._cells) - 1)
            }
        }
    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        # There is only one goal in this environment: to turn the light on.
        goal = {GroundAtom(self._LightOn, [self._light])}
        self._cells = [
            Object(f"cell{i}", self._cell_type)
            for i in range(CFG.grid_row_num_cells)
        ]
        self._cell_to_neighbors = {
            self._cells[0]: {self._cells[1]},
            self._cells[-1]: {self._cells[-2]},
            **{
                self._cells[t]: {self._cells[t - 1], self._cells[t + 1]}
                for t in range(1,
                               len(self._cells) - 1)
            }
        }
        tasks: List[EnvironmentTask] = []
        while len(tasks) < num:
            state_dict = {
                self._robot: {
                    "x": 0.5,
                },
                # Note: light level and door locations are fixed for now
                # in order to maintain consistency across train and test
                self._light: {
                    "x": len(self._cells) - 0.5,
                    "level": 0.0,
                    "target": 0.75,
                },
            }
            for i, cell in enumerate(self._cells):
                state_dict[cell] = {"x": i + 0.5}
            door = Object("door", self._door_type)
            state_dict[door] =     {
                     "x": rng.choice(range(3,len(self._cells)-2))-0.5,
                    "move_key": 0.0,
                    "move_target": 0.5,
                    "turn_key": 0.0,
                    "turn_target": 0.75
                }
            state = utils.create_state_from_dict(state_dict)
            tasks.append(EnvironmentTask(state, goal))
        return tasks
    

    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        dx, dlight, dmove, dturn = action.arr
        robbot_pos = state.get(self._robot, "x")
        try:
            robot_cells = [
            c for c in self._cells if self._In_holds(state, [self._robot, c])
            ]
        except:
            import ipdb;ipdb.set_trace()
        total_door_cells = []
        robot_cell = robot_cells[0]
        door_cell_to_door = {}
        for door in state:
            if not door.is_instance(self._door_type):
                continue
            door_pos = state.get(door, "x")
            door_move_key = state.get(door, "move_key")
            door_move_target = state.get(door, "move_target")
            door_turn_key = state.get(door, "turn_key")
            door_turn_target = state.get(door, "turn_target")
            door_cells = [
            c for c in self._cells if self._In_holds(state, [door, c])
            ]
            assert len(door_cells) == 1
            # Apply ddoor if we're in same cell as door
            # Can only open door, not close
            door_cell = door_cells[0]
            total_door_cells.append(door_cell)
            door_cell_to_door[door_cell] = door
            if robot_cell == door_cell and not (door_move_target - 0.1 \
                            <= door_move_key <= door_move_target + 0.1 \
    and door_turn_target - 0.1 <= door_turn_key <= door_turn_target + 0.1):
                new_door_level = np.clip(
                state.get(door, "move_key") + dmove, 0.0, 1.0)
                next_state.set(door, "move_key", new_door_level)
                new_door1_level = np.clip(
                state.get(door, "turn_key") + dturn, 0.0, 1.0)
                next_state.set(door, "turn_key", new_door1_level)

        # Apply dlight if we're in the same cell as the light.
        assert len(robot_cells) == 1
        light_cells = [
            c for c in self._cells if self._In_holds(state, [self._light, c])
        ]
        if not len(light_cells) == 1:
            import ipdb;ipdb.set_trace()
        assert len(light_cells) == 1
        light_cell = light_cells[0]
        if robot_cell == light_cell and dlight == 0.75 and self._LightOff_holds(
                state, [self._light]):
            next_state.set(self._light, "level", 0.75)


        #if we're not in the same cell as a door, we can move forward
        #elif the door is open we can move forward

        in_door_cell = False
        door_move_key = None
        door_move_target = None
        door_turn_key = None
        door_turn_target = None

        for door_cell in total_door_cells:
            if robot_cell == door_cell:
                in_door_cell = True
                door = door_cell_to_door[door_cell]
                door_move_key = state.get(door, "move_key")
                door_move_target = state.get(door, "move_target")
                door_turn_key = state.get(door, "turn_key")
                door_turn_target = state.get(door, "turn_target")
                break

        if not in_door_cell or (dx < 0) or (door_move_target - 0.1 <= door_move_key <= door_move_target + 0.1 \
    and door_turn_target - 0.1 <= door_turn_key <= door_turn_target + 0.1):
            # Apply dx to robot.
            new_x = np.clip(
                state.get(self._robot, "x") + dx, 0.0, len(self._cells))
            next_state.set(self._robot, "x", new_x)
        return next_state  
    


class GridDoorEnv(GridRowDoorEnv):

    def __init__(self, use_gui: bool = True) -> None:
        super().__init__(use_gui)
        self._robot_type = Type("robot", ["x", "y"])
        self._cell_type = Type("cell", ["x", "y"])
        self._light_type = Type("light", ["level", "target", "x", "y"])
        #door: x,y coordinate is the center of the door
        self._door_type = Type(
            "door",
            ["x", "y", "move_key", "move_target", "turn_key", "turn_target"])
                # Predicates
        self._RobotInCell = Predicate("RobotInCell",
                                      [self._robot_type, self._cell_type],
                                      self._In_holds)
        self._LightInCell = Predicate("LightInCell",
                                      [self._light_type, self._cell_type],
                                      self._In_holds)
        self._LightOn = Predicate("LightOn", [self._light_type],
                                  self._LightOn_holds)
        self._LightOff = Predicate("LightOff", [self._light_type],
                                   self._LightOff_holds)
        self._Adjacent = Predicate("Adjacent",
                                   [self._cell_type, self._cell_type],
                                   self._Adjacent_holds)
        self._robot = Object("robby", self._robot_type)
        self._light = Object("light", self._light_type)
        self._cells = []
        for j in range(CFG.grid_row_num_cells):
            self._cells.append([
            Object(f"cell{i,j}", self._cell_type)
            for i in range(CFG.grid_row_num_cells)
            ])
        self._cell_to_neighbors = {}
        for i in range(CFG.grid_row_num_cells):
            cells = self._cells[i]
            self._cell_to_neighbors.update({
                cells[0]: {cells[1]},
                cells[-1]: {cells[-2]},
                **{
                    cells[t]: {cells[t - 1], cells[t + 1]}
                    for t in range(1,
                                len(cells) - 1)
                }
            })
        for i in range(CFG.grid_row_num_cells):
            cells = [cells[i] for cells in self._cells]
            self._cell_to_neighbors[cells[0]].update({cells[1]})
            self._cell_to_neighbors[cells[-1]].update({cells[-2]})
            for t in range(1,len(cells) - 1):
                self._cell_to_neighbors[cells[t]].update({cells[t - 1], cells[t + 1]})
        
        flat_cells = []
        for cells in self._cells:
            flat_cells.extend([cell for cell in cells ])
        self._cells = flat_cells
    @classmethod
    def get_name(cls) -> str:
        return "grid_door"

    @property
    def types(self) -> Set[Type]:
        return {
            self._robot_type, self._cell_type, self._light_type,
            self._door_type
        }
    
    @property
    def action_space(self) -> Box:
        # dx, dy, dlight, ADD dmove, dturn LATER
        return Box(-np.inf, np.inf, (5, ))
    
    def _In_holds(self, state: State, objects: Sequence[Object]) -> bool:
        obj1, obj2 = objects
        x1 = state.get(obj1, "x")
        y1 = state.get(obj1, "y")
        x2 = state.get(obj2, "x")
        y2 = state.get(obj2, "y")

        # Threshold not important because dx is always discrete.
        return abs(x1 - x2) < 1e-3 and abs(y1 - y2) < 1e-3
    

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_train_tasks, rng=self._train_rng)

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        return self._get_tasks(num=CFG.num_test_tasks, rng=self._test_rng)
    

    def _get_tasks(self, num: int,
                   rng: np.random.Generator) -> List[EnvironmentTask]:
        # There is only one goal in this environment: to turn the light on.
        goal = {GroundAtom(self._LightOn, [self._light])}
        # The only variation in the initial state is the light target level.
        tasks: List[EnvironmentTask] = []
        while len(tasks) < num:
            state_dict = {
                self._robot: {
                    "x": 0.5,
                    "y": 0.5
                },
                self._light: {
                    "x": CFG.grid_row_num_cells - 0.5,
                    "y": CFG.grid_row_num_cells - 0.5,
                    "level": 0.0,
                    "target": 0.75,
                },
            }
            door = Object("door", self._door_type)
            state_dict[door] =     {
                     "x": 0.5,
                     "y": 1,
                    "move_key": 0.0,
                    "move_target": 0.5,
                    "turn_key": 0.0,
                    "turn_target": 0.75
                }
            door1 = Object("door1", self._door_type)
            state_dict[door1] =     {
                     "x": 1,
                     "y": 0.5,
                    "move_key": 0.0,
                    "move_target": 0.5,
                    "turn_key": 0.0,
                    "turn_target": 0.75
                }
            for i in range(CFG.grid_row_num_cells):
                for j in range(CFG.grid_row_num_cells):
                    state_dict[self._cells[i*CFG.grid_row_num_cells+j]] = {"x": j + 0.5, "y": i + 0.5}
            state = utils.create_state_from_dict(state_dict)
            tasks.append(EnvironmentTask(state, goal))
        # import ipdb;ipdb.set_trace()
        return tasks
    
    def simulate(self, state: State, action: Action) -> State:
        assert self.action_space.contains(action.arr)
        next_state = state.copy()
        dx, dy, dlight, dmove, dturn = action.arr
        # if dy > 0:
        #     import ipdb;ipdb.set_trace()
        robot_cells = [
        c for c in self._cells if self._In_holds(state, [self._robot, c])
        ]
        robot_cell = robot_cells[0]
        robot_x = state.get(self._robot, "x")
        robot_y = state.get(self._robot, "y")
        robot_cell = robot_cells[0]
        doors_next_to_robot = []
        for door in state:
            if not door.is_instance(self._door_type):
                continue
            door_x = state.get(door, "x")
            door_y = state.get(door, "y")
            if door_x == robot_x and abs(door_y-robot_y)==0.5 or door_y == robot_y and abs(door_x-robot_x)==0.5:
                doors_next_to_robot.append(door)
            door_move_key = state.get(door, "move_key")
            door_move_target = state.get(door, "move_target")
            door_turn_key = state.get(door, "turn_key")
            door_turn_target = state.get(door, "turn_target")
            door_cells = [
            c for c in self._cells if door_x == state.get(c, "x") and abs(door_y-state.get(c, "y"))==0.5 or door_y == state.get(c, "y") and abs(door_x-state.get(c, "x"))==0.5
            ]
            # Apply ddoor if we're in same cell as door
            # Can only open door, not close
            if robot_cell in door_cells and not (door_move_target - 0.1 \
                            <= door_move_key <= door_move_target + 0.1 \
    and door_turn_target - 0.1 <= door_turn_key <= door_turn_target + 0.1):
                new_door_level = np.clip(
                state.get(door, "move_key") + dmove, 0.0, 1.0)
                next_state.set(door, "move_key", new_door_level)
                new_door1_level = np.clip(
                state.get(door, "turn_key") + dturn, 0.0, 1.0)
                next_state.set(door, "turn_key", new_door1_level)

        # Apply dlight if we're in the same cell as the light.
        assert len(robot_cells) == 1
        light_cells = [
            c for c in self._cells if self._In_holds(state, [self._light, c])
        ]
        assert len(light_cells) == 1
        light_cell = light_cells[0]
        if robot_cell == light_cell and dlight == 0.75 and self._LightOff_holds(
                state, [self._light]):
            next_state.set(self._light, "level", 0.75)


        #if we're not in the same cell as a door, we can move forward
        #elif the door is open we can move forward
        can_move = True
        for door in doors_next_to_robot:
            door_move_key = state.get(door, "move_key")
            door_move_target = state.get(door, "move_target")
            door_turn_key = state.get(door, "turn_key")
            door_turn_target = state.get(door, "turn_target")
            door_open = (door_move_target - 0.1 <= door_move_key <= door_move_target + 0.1 \
    and door_turn_target - 0.1 <= door_turn_key <= door_turn_target + 0.1)
            if dx > 0 and state.get(door, "y") == robot_y and state.get(door, "x")-robot_x == 0.5 and not door_open:
                can_move = False
                break
            if dx < 0 and state.get(door, "y") == robot_y and state.get(door, "x")-robot_x == -0.5 and not door_open:
                can_move = False
                break
            if dy < 0 and state.get(door, "x") == robot_x and state.get(door, "y")-robot_y == -0.5 and not door_open:
                can_move = False
                break
            if dy > 0 and state.get(door, "x") == robot_x and state.get(door, "y")-robot_y == 0.5 and not door_open:
                # import ipdb;ipdb.set_trace()
                can_move = False
                break

            # Apply dx to robot.
        if can_move:
            new_x = np.clip(
                state.get(self._robot, "x") + dx, 0.0, len(self._cells))
            next_state.set(self._robot, "x", new_x)
            new_y = np.clip(
                state.get(self._robot, "y") + dy, 0.0, len(self._cells))
            next_state.set(self._robot, "y", new_y)
        return next_state  






    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> \
                matplotlib.figure.Figure:
        fig, ax = plt.subplots(1, 1)
        plt.xlim([0, len(self._cells)])
        plt.ylim([0, 1])

        # Draw the cells.
        for i in range(len(CFG.grid_row_num_cells)):
            for j in range(len(CFG.grid_row_num_cells)):
                rect = plt.Rectangle((i, j),
                                    self.cell_width,
                                    self.cell_width,
                                    edgecolor="gray",
                                    facecolor="gray")
                ax.add_patch(rect)

# OK FINISH CODING THE REST OF THIS !!!
        # Draw light, door, and robot.
        if self._LightOn_holds(state, (self._light, )):
            light = plt.Rectangle(
                (len(self._cells) - (self.cell_width + \
                                     self.light_width) / 2.0,
                 1 - self.light_width),
                self.light_width,
                self.light_width,
                edgecolor="yellow",
                facecolor="yellow")
        else:
            light = plt.Rectangle(
                (len(self._cells) - (self.cell_width + \
                                     self.light_width) / 2.0,
                 1 - self.light_width),
                self.light_width,
                self.light_width,
                edgecolor="white",
                facecolor="white")
        ax.add_patch(light)
        for door in state:
            if not door.is_instance(self._door_type):
                continue
            door_pos = state.get(door, "x")
            door_move_key = state.get(door, "move_key")
            door_move_key_target = state.get(door, "move_target")
            door_turn_key = state.get(door, "turn_key")
            door_turn_key_target = state.get(door, "turn_target")
            if (door_move_key_target - 0.1 <= door_move_key <= \
                door_move_key_target + 0.1 and door_turn_key_target - 0.1 \
                    <= door_turn_key <= door_turn_key_target + 0.1):
                draw_door = plt.Rectangle((door_pos - self.cell_width / 2.0, 0),
                                    self.cell_width,
                                    self.cell_width,
                                    edgecolor="gray",
                                    facecolor="gray")
            else:
                draw_door = plt.Rectangle((door_pos - self.cell_width / 2.0, 0),
                                    self.cell_width,
                                    self.cell_width,
                                    edgecolor="darkgoldenrod",
                                    facecolor="darkgoldenrod")
            ax.add_patch(draw_door)
        robot_pos = state.get(self._robot, "x")
        robot = plt.Rectangle((robot_pos - self.robot_width / 2.0, 0),
                              self.robot_width,
                              self.robot_height,
                              edgecolor="red",
                              facecolor="red")
        ax.add_patch(robot)
        return fig