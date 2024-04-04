"""Test cases for the tools environment."""

import numpy as np
import pytest

from predicators.src import utils
from predicators.src.envs.tools import ToolsEnv
from predicators.src.structs import Action


class DummyToolsEnv(ToolsEnv):
    """Dummy tools environment that exposes get_best_screwdriver_or_none."""

    def get_best_screwdriver_or_none(self, state, screw):
        """Expose parent class method."""
        return self._get_best_screwdriver_or_none(state, screw)


def test_tools():
    """Tests for ToolsEnv class properties."""
    utils.reset_config({"env": "tools"})
    env = ToolsEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    assert len(env.predicates) == 15
    assert {pred.name for pred in env.goal_predicates} == \
        {"ScrewPlaced", "ScrewFastened", "NailPlaced",
         "NailFastened", "BoltPlaced", "BoltFastened"}
    assert len(env.options) == 11
    assert len(env.types) == 8
    assert env.action_space.shape == (4, )
    task = env.get_train_tasks()[0]
    with pytest.raises(NotImplementedError):
        env.render_state(task.init, task)


def test_tools_fasten_option():
    """Tests for the ToolsEnv option policy for fastening, which checks that
    the correct tool and contraption are used."""
    utils.reset_config({"env": "tools", "tools_num_items_train": [25]})
    env = DummyToolsEnv()
    task = env.get_train_tasks()[0]
    state = task.init
    robot = None
    screw = None
    contraption = None
    for obj in state:
        if obj.type.name == "robot":
            robot = obj
        elif obj.type.name == "contraption":
            contraption = obj
        elif obj.type.name == "screw":
            assert screw is None  # only 1 screw possible
            screw = obj
    assert robot is not None
    assert screw is not None
    assert contraption is not None
    FastenScrewByHand = [
        o for o in env.options if o.name == "FastenScrewByHand"
    ][0]
    option = FastenScrewByHand.ground([robot, screw, contraption], [])
    act = option.policy(state)
    exp = np.array([DummyToolsEnv.table_ux, DummyToolsEnv.table_uy, 0.0, 0.0])
    assert np.allclose(act.arr, exp)


def test_tools_failure_cases():
    """Tests for the cases where simulate() is a no-op."""
    utils.reset_config({"env": "tools", "tools_num_items_train": [25]})
    env = DummyToolsEnv()
    HandEmpty = [o for o in env.predicates if o.name == "HandEmpty"][0]
    task = env.get_train_tasks()[0]
    state = task.init
    atoms = utils.abstract(state, env.predicates)
    robot = None
    screw = None
    nail = None
    bolt = None
    wrench = None
    big_hammer = None
    small_hammer = None
    contraption = None
    for obj in state:
        if obj.type.name == "robot":
            robot = obj
        elif obj.type.name == "contraption":
            contraption = obj
        elif obj.type.name == "screw":
            assert screw is None  # only 1 screw possible
            screw = obj
        elif obj.type.name == "nail":
            nail = obj
        elif obj.type.name == "bolt":
            bolt = obj
        elif obj.type.name == "hammer" and \
             state.get(obj, "size") > 0.5:
            big_hammer = obj
        elif obj.type.name == "hammer" and \
             state.get(obj, "size") < 0.5:
            small_hammer = obj
        elif obj.type.name == "wrench":
            wrench = obj
    assert robot is not None
    assert screw is not None
    assert nail is not None
    assert bolt is not None
    assert big_hammer is not None
    assert small_hammer is not None
    assert wrench is not None
    assert contraption is not None
    assert HandEmpty([robot]) in atoms
    assert env.get_best_screwdriver_or_none(state, screw) is not None
    orig_val = env.screw_shape_hand_thresh
    env.screw_shape_hand_thresh = 0.01
    assert env.get_best_screwdriver_or_none(state, screw) is None
    env.screw_shape_hand_thresh = orig_val
    # Can't both pick and place
    arr = np.array([0.0, 0.0, 1.0, 1.0], dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert state.allclose(next_state)
    # Can't place while holding nothing
    arr = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert state.allclose(next_state)
    # Can't pick air
    arr = np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert state.allclose(next_state)
    # Can't pick big ungraspable object
    arr = np.array([
        state.get(big_hammer, "pose_x"),
        state.get(big_hammer, "pose_y"), 1.0, 0.0
    ],
                   dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert state.allclose(next_state)
    # Pick nail
    arr = np.array(
        [state.get(nail, "pose_x"),
         state.get(nail, "pose_y"), 1.0, 0.0],
        dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Can't pick while holding something
    arr = np.array(
        [state.get(nail, "pose_x"),
         state.get(nail, "pose_y"), 1.0, 0.0],
        dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert state.allclose(next_state)
    # Place nail on contraption
    arr = np.array([
        state.get(contraption, "pose_lx") + 0.1,
        state.get(contraption, "pose_ly") + 0.1, 0.0, 1.0
    ],
                   dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Can't pick off of contraption
    arr = np.array(
        [state.get(nail, "pose_x"),
         state.get(nail, "pose_y"), 1.0, 0.0],
        dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert state.allclose(next_state)
    # Can't fasten off a contraption
    arr = np.array(
        [state.get(wrench, "pose_x"),
         state.get(wrench, "pose_y"), 0.0, 0.0],
        dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert state.allclose(next_state)
    # Pick wrench
    arr = np.array(
        [state.get(wrench, "pose_x"),
         state.get(wrench, "pose_y"), 1.0, 0.0],
        dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Can't place wrench onto contraption
    arr = np.array([
        state.get(contraption, "pose_lx") + 0.1,
        state.get(contraption, "pose_ly") + 0.1, 0.0, 1.0
    ],
                   dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert state.allclose(next_state)
    # Can't fasten nail with wrench
    arr = np.array(
        [state.get(nail, "pose_x"),
         state.get(nail, "pose_y"), 0.0, 0.0],
        dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert state.allclose(next_state)
    # Reset to initial state
    state = task.init
    # Pick bolt
    arr = np.array(
        [state.get(bolt, "pose_x"),
         state.get(bolt, "pose_y"), 1.0, 0.0],
        dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Place bolt on contraption
    arr = np.array([
        state.get(contraption, "pose_lx") + 0.1,
        state.get(contraption, "pose_ly") + 0.1, 0.0, 1.0
    ],
                   dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Pick hammer
    arr = np.array([
        state.get(small_hammer, "pose_x"),
        state.get(small_hammer, "pose_y"), 1.0, 0.0
    ],
                   dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert not state.allclose(next_state)
    # Change the state
    state = next_state
    # Can't fasten bolt with hammer
    arr = np.array(
        [state.get(bolt, "pose_x"),
         state.get(bolt, "pose_y"), 0.0, 0.0],
        dtype=np.float32)
    next_state = env.simulate(state, Action(arr))
    assert state.allclose(next_state)
