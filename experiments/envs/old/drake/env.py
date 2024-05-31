from typing import ClassVar, List, Optional, Set, Tuple, Type, cast
from predicators.envs.base_env import BaseEnv
from predicators.structs import Action, EnvironmentTask, Predicate, State
import gym
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from pydrake.all import Simulator, System, Context
from manipulation.station import MakeHardwareStation, load_scenario
import pydot
import webbrowser

class DrakeEnv(BaseEnv):
    # Settings
    world_range_x: ClassVar[Tuple[float, float]] = [-2, 2]
    world_range_y: ClassVar[Tuple[float, float]] = [-2, 2]
    world_range_z: ClassVar[Tuple[float, float]] = [-2, 2]

    def __init__(self, use_gui: bool=True):
        super().__init__(use_gui)
        self._system: System = MakeHardwareStation(load_scenario(filename="experiments/envs/drake/scenario.yaml"))
        self._simulator = Simulator(self._system)
        svg_data = pydot.graph_from_dot_data(self._system.GetGraphvizString())[0].create_svg()
        open('drake_diagram.svg', 'wb').write(svg_data)
        webbrowser.get("wslview %s").open_new_tab("/home/barcisz/superurop/predicators/drake_diagram.svg")
        context = self._system.CreateDefaultContext()
        self._system.GetInputPort("wsg.position")

        image = self._system.GetOutputPort(f"plt_render_camera.rgb_image").Eval(context).data
        self._system.ForcedPublish(context)
        plt.imshow(image)
        plt.show()

    @classmethod
    def get_name(cls) -> str:
        """Get the unique name of this environment, used as the argument to
        `--env`."""
        return "drake"

    def simulate(self, state: State, action: Action) -> State:
        """Get the next state, given a state and an action.

        Note that this action is a low-level action (i.e., its array
        representation is a member of self.action_space), NOT an option.

        This function is primarily used in the default option model, and
        for implementing the default self.step(action). It is not meant to
        be part of the "final system", where the environment is the real world.
        """
        raise NotImplementedError("Override me!")

    def _generate_train_tasks(self) -> List[EnvironmentTask]:
        """Create an ordered list of tasks for training."""
        raise NotImplementedError("Override me!")

    def _generate_test_tasks(self) -> List[EnvironmentTask]:
        """Create an ordered list of tasks for testing / evaluation."""
        raise NotImplementedError("Override me!")

    @property
    def predicates(self) -> Set[Predicate]:
        """Get the set of predicates that are given with this environment."""
        raise NotImplementedError("Override me!")

    @property
    def goal_predicates(self) -> Set[Predicate]:
        """Get the subset of self.predicates that are used in goals."""
        raise NotImplementedError("Override me!")

    @property
    def types(self) -> Set[Type]:
        """Get the set of types that are given with this environment."""
        raise NotImplementedError("Override me!")

    @property
    def action_space(self) -> gym.spaces.Box:
        """(iiwa_xyz, iiwa_quaternion_rotation, wsg_position)"""
        lower_bound = np.array([self.world_range_x[0], self.world_range_y[0], self.world_range_z[0], 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        upper_bound = np.array([self.world_range_x[1], self.world_range_y[1], self.world_range_z[1], 1.0, 1.0, 1.0, 1.0], dtype=np.float32)
        return gym.spaces.Box(lower_bound, upper_bound)

    def render_state_plt(
            self,
            state: State,
            task: EnvironmentTask,
            action: Optional[Action] = None,
            caption: Optional[str] = None) -> matplotlib.figure.Figure:
        """Render a state and action into a Matplotlib figure.

        Like simulate, this function is not meant to be part of the
        "final system", where the environment is the real world. It is
        just for convenience, e.g., in test coverage.

        For environments which don't use Matplotlib for rendering, this
        function should be overriden to simply crash.

        NOTE: Users of this method must remember to call `plt.close()`,
        because this method returns an active figure object!
        """
        fig, ax = plt.subplot()
        context = cast(Context, state.simulator_state)
        image = self._system.GetOutputPort(f"plt_render_camera.rgb_image").Eval(context).data
        self._system.ForcedPublish(context)
        ax.imshow(image)
        return fig
