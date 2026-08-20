"""Test cases for PyBulletDominoComposedEnv and its components."""

import numpy as np
import pytest

from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.components.grid_component import \
    GridComponent
from predicators.envs.pybullet_domino.task_generators import \
    domino_task_generator as dtg
from predicators.settings import CFG
from predicators.structs import Object, State, Type


class TestDominoComponent:
    """Tests for DominoComponent."""

    def __init__(self) -> None:
        self.comp: DominoComponent = None  # type: ignore

    def setup_method(self) -> None:
        """setup method."""
        workspace_bounds = {
            "x_lb": 0.4,
            "x_ub": 1.1,
            "y_lb": 1.1,
            "y_ub": 1.6,
            "z_lb": 0.4,
            "z_ub": 0.95,
        }
        CFG.domino_use_domino_blocks_as_target = True
        CFG.domino_has_glued_dominos = False
        self.comp = DominoComponent(num_dominos_max=5,
                                    num_targets_max=2,
                                    num_pivots_max=1,
                                    workspace_bounds=workspace_bounds)

    def test_get_types(self) -> None:
        """Test get types."""
        types = self.comp.get_types()
        type_names = {t.name for t in types}
        assert "domino" in type_names

    def test_get_predicates(self) -> None:
        """Test get predicates."""
        preds = self.comp.get_predicates()
        pred_names = {p.name for p in preds}
        assert "Toppled" in pred_names
        assert "Upright" in pred_names
        assert "Tilting" in pred_names

    def test_get_goal_predicates(self) -> None:
        """Test get goal predicates."""
        goal_preds = self.comp.get_goal_predicates()
        assert len(goal_preds) == 1
        assert "Toppled" in {p.name for p in goal_preds}

    def test_get_objects(self) -> None:
        """Test get objects."""
        objects = self.comp.get_objects()
        # With domino_use_domino_blocks_as_target=True,
        # dominos = 5 + 2 = 7, targets = 0, pivots = 1
        assert len(objects) == 8

    def test_place_domino(self) -> None:
        """Test place domino."""
        d = self.comp.place_domino(0, 0.5, 1.3, 0.0, is_start_block=True)
        assert d["x"] == 0.5
        assert d["y"] == 1.3
        assert d["is_held"] == 0.0
        # Start block should have green color
        assert d["r"] == pytest.approx(0.56, abs=0.01)

    def test_place_target_domino(self) -> None:
        """Test place target domino."""
        d = self.comp.place_domino(1, 0.6, 1.3, 0.0, is_target_block=True)
        # Target should have purple/pink color
        assert d["r"] == pytest.approx(0.85, abs=0.01)


def test_unfinished_state_avoids_staging_collisions() -> None:
    """Test unfinished movable blocks avoid start/target blocks."""
    workspace_bounds = {
        "x_lb": 0.4,
        "x_ub": 1.1,
        "y_lb": 1.1,
        "y_ub": 1.6,
        "z_lb": 0.4,
        "z_ub": 0.95,
    }
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_has_glued_dominos = False
    comp = DominoComponent(num_dominos_max=5,
                           num_targets_max=2,
                           num_pivots_max=1,
                           workspace_bounds=workspace_bounds)
    robot = Object("robot", Type("robot", ["x", "y", "z"]))
    generator = dtg.DominoTaskGenerator(comp, robot, {})

    first_staging_x = comp.domino_x_lb + comp.domino_width
    first_staging_y = comp.domino_y_lb + comp.domino_width
    obj_dict = {
        comp.dominos[0]:
        comp.place_domino(0,
                          first_staging_x,
                          first_staging_y,
                          0.0,
                          is_start_block=True),
        comp.dominos[1]:
        comp.place_domino(1,
                          first_staging_x + 0.25,
                          first_staging_y,
                          0.0,
                          is_target_block=True),
        comp.dominos[2]:
        comp.place_domino(2, 0.9, 1.35, 0.0),
    }

    # pylint: disable=protected-access
    moved = generator.stage_movable_blocks(obj_dict)

    assert moved is not None
    movable = comp.dominos[2]
    assert not generator._placement_collides(
        movable, moved[movable], {
            comp.dominos[0]: moved[comp.dominos[0]],
            comp.dominos[1]: moved[comp.dominos[1]],
        })
    assert moved[movable]["x"] != pytest.approx(first_staging_x)


def test_plain_task_attaches_domino_evaluator() -> None:
    """The plain chain generator attaches a DominoEvaluator (cascade
    certificate + per-toppled-blue reward cost) exactly when targets are domino
    blocks and dominoes are the only dynamic component; composed variants with
    extra components keep evaluator=None."""
    workspace_bounds = {
        "x_lb": 0.4,
        "x_ub": 1.1,
        "y_lb": 1.1,
        "y_ub": 1.6,
        "z_lb": 0.4,
        "z_ub": 0.95,
    }
    CFG.domino_use_domino_blocks_as_target = True
    CFG.domino_has_glued_dominos = False
    CFG.domino_min_block_tasks = False
    CFG.domino_initialize_at_finished_state = True
    comp = DominoComponent(num_dominos_max=5,
                           num_targets_max=2,
                           num_pivots_max=1,
                           workspace_bounds=workspace_bounds)
    robot = Object("robot", Type("robot", ["x", "y", "z"]))
    robot_init = {"x": 0.75, "y": 0.72, "z": 0.9}
    generator = dtg.DominoTaskGenerator(comp, robot, robot_init)

    # pylint: disable=protected-access
    task = generator._generate_single_task(0, np.random.default_rng(0), [3],
                                           [1], [0])
    assert task is not None
    from predicators.envs.pybullet_domino.env import \
        DominoEvaluator  # pylint: disable=import-outside-toplevel
    assert isinstance(task.evaluator, DominoEvaluator)
    assert task.goal_nl is not None
    assert "Only the green domino may ever be pushed." in task.goal_nl
    # Every enforced rule is stated in the goal text, including the
    # counterfactual fingertip verification - without it an arm-assisted
    # layout fails with verdicts the agent cannot explain
    # (run_20260718_141716).
    assert "every robot link except the fingertips" in task.goal_nl

    # A ball/fan-style extra component can topple dominoes without a
    # robot Push, which the certificate would falsely reject - so its
    # presence must disable the evaluator.
    composed = dtg.DominoTaskGenerator(comp,
                                       robot,
                                       robot_init,
                                       additional_components=[object()])
    task = composed._generate_single_task(0, np.random.default_rng(0), [3],
                                          [1], [0])
    assert task is not None
    assert task.evaluator is None


def test_counterfactual_cascade_probe() -> None:
    """The counterfactual push probe certifies the generator's own finished
    chain (guaranteed cascade geometry) and rejects the same scene with its
    blues removed from the chain."""
    # pylint: disable=protected-access
    from predicators.envs.pybullet_domino.env import \
        PyBulletDominoEnv  # pylint: disable=import-outside-toplevel
    from predicators.utils import \
        reset_config  # pylint: disable=import-outside-toplevel
    reset_config({
        "env": "pybullet_domino",
        "approach": "oracle",
        "seed": 0,
        "num_train_tasks": 0,
        "num_test_tasks": 1,
        "domino_test_turn_ratio": 1.0,
        "domino_initialize_at_finished_state": True,
        "domino_use_domino_blocks_as_target": True,
        "domino_use_continuous_place": True,
        "domino_has_glued_dominos": False,
    })
    env = PyBulletDominoEnv(use_gui=False)
    task = env.get_test_tasks()[0].task
    init, goal = task.init, task.goal
    domino_type = next(t for t in env.types if t.name == "domino")
    greens = [
        d for d in init.get_objects(domino_type)
        if DominoComponent._StartBlock_holds(init, [d])
    ]
    blues = [
        d for d in init.get_objects(domino_type)
        if DominoComponent._MovableBlock_holds(init, [d])
    ]
    assert greens and blues
    ok, detail = env.run_counterfactual_cascade_probe(init.copy(), greens,
                                                      goal)
    assert ok, detail
    # Remove the chain's blues: the same push cannot reach the target.
    broken = init.copy()
    for i, blue in enumerate(blues):
        broken.set(blue, "x", 0.45)
        broken.set(blue, "y", 1.55 - 0.1 * i)
    ok, detail = env.run_counterfactual_cascade_probe(broken, greens, goal)
    assert not ok
    assert "reaches the goal at none of" in detail

    # Combined substrate: with a probe_process_model_factory stamped (a
    # belief env whose approach learned residual rules), the same broken
    # scene certifies when the rules model the propagation the base sim
    # lacks - and the detail carries the load-bearing diagnostic. The
    # real env never stamps a factory, so clearing it restores the
    # base-only rejection.
    targets = sorted(
        {a.objects[0]
         for a in goal if a.predicate.name == "Toppled"}, key=str)
    assert targets

    def make_stepper():
        """Force-topple every target once a green has toppled."""

        def step(state, action):
            del action
            if all(abs(state.get(g, "roll")) < 1.2 for g in greens):
                return state
            new_state = state.copy()
            for t in targets:
                new_state.set(t, "roll", 1.57)
            return new_state

        return step

    env.probe_process_model_factory = make_stepper
    ok, detail = env.run_counterfactual_cascade_probe(broken, greens, goal)
    assert ok, detail
    assert "residual rules riding on the base sim" in detail
    assert "load-bearing" in detail

    # Physically no-op rules must NOT be flagged load-bearing: the
    # base-only diagnostic replay runs the same attempt count as the
    # combined probe, so identical physics yields identical verdicts
    # (run_20260728_111805 logged 14 spurious notes off a 3-vs-1
    # attempt asymmetry on a knife-edge layout).
    env.probe_process_model_factory = lambda: (lambda state, action: state)
    ok, detail = env.run_counterfactual_cascade_probe(init.copy(), greens,
                                                      goal)
    assert ok, detail
    assert "load-bearing" not in detail

    env.probe_process_model_factory = None
    ok, _ = env.run_counterfactual_cascade_probe(broken, greens, goal)
    assert not ok


class TestGridComponent:
    """Tests for GridComponent."""

    def __init__(self) -> None:
        self.workspace_bounds: dict = {}
        self.domino_type: Type = None  # type: ignore
        self.comp: GridComponent = None  # type: ignore

    def setup_method(self) -> None:
        """setup method."""
        self.workspace_bounds = {
            "x_lb": 0.4,
            "x_ub": 1.1,
            "y_lb": 1.1,
            "y_ub": 1.6,
            "z_lb": 0.4,
            "z_ub": 0.95,
        }
        self.domino_type = Type(
            "domino", ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"])
        CFG.domino_include_connected_predicate = True
        self.comp = GridComponent(workspace_bounds=self.workspace_bounds,
                                  pos_gap=0.098,
                                  domino_type=self.domino_type)

    def test_get_types(self) -> None:
        """Test get types."""
        types = self.comp.get_types()
        type_names = {t.name for t in types}
        assert "loc" in type_names
        assert "angle" in type_names

    def test_get_predicates(self) -> None:
        """Test get predicates."""
        preds = self.comp.get_predicates()
        pred_names = {p.name for p in preds}
        assert "DominoAtPos" in pred_names
        assert "DominoAtRot" in pred_names
        assert "PosClear" in pred_names
        assert "Connected" in pred_names

    def test_get_predicates_with_adjacent(self) -> None:
        """Test get predicates with adjacent."""
        CFG.domino_include_connected_predicate = False
        comp = GridComponent(workspace_bounds=self.workspace_bounds,
                             pos_gap=0.098,
                             domino_type=self.domino_type)
        preds = comp.get_predicates()
        pred_names = {p.name for p in preds}
        assert "AdjacentTo" in pred_names
        assert "Connected" not in pred_names

    def test_rotations(self) -> None:
        """Test rotations."""
        assert len(self.comp.rotations) == 8

    def test_generate_grid_coordinates(self) -> None:
        """Test generate grid coordinates."""
        x_coords, y_coords = self.comp.generate_grid_coordinates(3, 3)
        assert len(x_coords) == 3
        assert len(y_coords) == 3
        # Grid should be centered in workspace
        x_center = (self.workspace_bounds["x_lb"] +
                    self.workspace_bounds["x_ub"]) / 2
        assert x_coords[1] == pytest.approx(x_center, abs=0.01)

    def test_create_position_objects(self) -> None:
        """Test create position objects."""
        positions, pos_dict = self.comp.create_position_objects(3, 2)
        assert len(positions) == 6
        assert len(pos_dict) == 6
        # Check naming convention
        assert positions[0].name == "loc_y0_x0"
        assert positions[0].type.name == "loc"

    def test_extract_feature_position(self) -> None:
        """Test extract feature position."""
        self.comp.create_position_objects(2, 2)
        pos = self.comp.positions[0]
        assert self.comp.extract_feature(pos, "xx") is not None
        assert self.comp.extract_feature(pos, "yy") is not None

    def test_extract_feature_angle(self) -> None:
        """Test extract feature angle."""
        rot = self.comp.rotations[3]  # ang_0
        val = self.comp.extract_feature(rot, "angle")
        assert val == 0.0

    def test_extract_feature_unknown(self) -> None:
        """Test extract feature unknown."""
        domino = Object("d0", self.domino_type)
        assert self.comp.extract_feature(domino, "x") is None

    def test_get_init_dict_entries(self) -> None:
        """Test get init dict entries."""
        rng = np.random.default_rng(0)
        entries = self.comp.get_init_dict_entries(rng)
        assert len(entries) == 8  # 8 rotation objects
        # Check that angle values are present
        for _obj, d in entries.items():
            assert "angle" in d

    def test_connected_predicate(self) -> None:
        """Test Connected predicate with a simple grid."""
        # pylint: disable=protected-access
        positions, pos_dict = self.comp.create_position_objects(3, 3)

        # Build a state with position objects
        data = {}
        for pos, d in pos_dict.items():
            data[pos] = np.array([d["xx"], d["yy"]], dtype=np.float32)

        state = State(data)

        # pos (0,0) and (1,0) should be connected (adjacent in x)
        assert self.comp._Connected_holds(state, [positions[0], positions[1]])

        # pos (0,0) and (0,1) should be connected (adjacent in y)
        assert self.comp._Connected_holds(state, [positions[0], positions[3]])

        # pos (0,0) and (1,1) should NOT be connected (diagonal)
        assert not self.comp._Connected_holds(state,
                                              [positions[0], positions[4]])

        # Same position should not be connected
        assert not self.comp._Connected_holds(state,
                                              [positions[0], positions[0]])

    def test_pos_clear_predicate(self) -> None:
        """Test PosClear predicate."""
        # pylint: disable=protected-access
        positions, pos_dict = self.comp.create_position_objects(3, 3)
        x0 = pos_dict[positions[0]]["xx"]
        y0 = pos_dict[positions[0]]["yy"]

        # Create state with one domino at position (0,0) and position objects
        domino = Object("d0", self.domino_type)
        data = {
            domino:
            np.array([x0, y0, 0.5, 0.0, 0.0, 0.6, 0.8, 1.0, 0.0],
                     dtype=np.float32),
        }
        for pos, d in pos_dict.items():
            data[pos] = np.array([d["xx"], d["yy"]], dtype=np.float32)
        state = State(data)

        # Position (0,0) should not be clear
        assert not self.comp._PosClear_holds(state, [positions[0]])

        # Other positions should be clear
        assert self.comp._PosClear_holds(state, [positions[1]])

    def test_domino_at_pos_predicate(self) -> None:
        """Test DominoAtPos predicate."""
        # pylint: disable=protected-access
        positions, pos_dict = self.comp.create_position_objects(3, 3)
        x0 = pos_dict[positions[0]]["xx"]
        y0 = pos_dict[positions[0]]["yy"]

        domino = Object("d0", self.domino_type)
        data = {
            domino:
            np.array([x0, y0, 0.5, 0.0, 0.0, 0.6, 0.8, 1.0, 0.0],
                     dtype=np.float32),
        }
        for pos, d in pos_dict.items():
            data[pos] = np.array([d["xx"], d["yy"]], dtype=np.float32)
        state = State(data)

        # Domino should be at position 0
        assert self.comp._DominoAtPos_holds(state, [domino, positions[0]])
        # Not at position 1
        assert not self.comp._DominoAtPos_holds(state, [domino, positions[1]])

    def test_domino_at_rot_predicate(self) -> None:
        """Test DominoAtRot predicate."""
        # pylint: disable=protected-access
        rot_0 = self.comp.rotations[3]  # ang_0
        rot_90 = self.comp.rotations[5]  # ang_90

        domino = Object("d0", self.domino_type)
        # yaw = 0 radians
        data = {
            domino:
            np.array([0.5, 1.3, 0.5, 0.0, 0.0, 0.6, 0.8, 1.0, 0.0],
                     dtype=np.float32),
            rot_0:
            np.array([0.0], dtype=np.float32),
            rot_90:
            np.array([90.0], dtype=np.float32),
        }
        state = State(data)

        assert self.comp._DominoAtRot_holds(state, [domino, rot_0])
        assert not self.comp._DominoAtRot_holds(state, [domino, rot_90])
