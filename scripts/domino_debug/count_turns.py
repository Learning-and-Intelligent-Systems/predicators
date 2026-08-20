"""Count % of generated domino test tasks that contain a turn.

Uses whatever DominoTaskGenerator is currently installed on disk, so it
can be run across git versions of the generator by swapping the file in
place.
"""
import numpy as np

from predicators import utils
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.envs.pybullet_domino.env import PyBulletDominoEnv
from predicators.envs.pybullet_domino.task_generators import \
    domino_task_generator as dtg
from predicators.settings import CFG
from predicators.structs import Task

N_TASKS = 40
SEED = 0


def ang_diff(a: float, b: float) -> float:
    """Return the smallest unsigned angle between a and b (mod pi)."""
    d = (a - b) % np.pi
    return min(d, np.pi - d)


def is_turn(task: Task, comp: DominoComponent) -> bool:
    """Return True if the task's start/target dominoes differ in yaw."""
    st = task.init
    sy = ty = None
    for d in st.get_objects(comp.domino_type):
        r, g, b = st.get(d, "r"), st.get(d, "g"), st.get(d, "b")
        if abs(r - comp.start_domino_color[0]) < 1e-2 and \
           abs(g - comp.start_domino_color[1]) < 1e-2:
            sy = st.get(d, "yaw")
        elif abs(r - comp.target_domino_color[0]) < 1e-2 and \
             abs(b - comp.target_domino_color[2]) < 1e-2:
            ty = st.get(d, "yaw")
    return sy is not None and ty is not None and \
        ang_diff(sy, ty) > np.deg2rad(30)


def main() -> None:
    """Print the percentage of generated test tasks with a turn."""
    utils.reset_config({
        "env": "pybullet_domino",
        "seed": SEED,
        "num_train_tasks": 0,
        "num_test_tasks": N_TASKS,
        "domino_initialize_at_finished_state": False,
        "domino_use_domino_blocks_as_target": True,
        "domino_has_glued_dominos": False,
        "domino_test_num_dominos": [3],
        "domino_test_num_targets": [1, 2],
        "domino_test_num_pivots": [0],
    })
    env = PyBulletDominoEnv(use_gui=False)
    comp = env._domino_component  # pylint: disable=protected-access
    assert comp is not None
    robot_init_state = {
        "x": env.robot_init_x,
        "y": env.robot_init_y,
        "z": env.robot_init_z,
        "fingers": env.open_fingers,
        "roll": env.robot_init_roll,
        "tilt": env.robot_init_tilt,
        "wrist": env.robot_init_wrist,
    }
    gen = dtg.DominoTaskGenerator(
        domino_component=comp,
        robot=env._robot,  # pylint: disable=protected-access
        robot_init_state=robot_init_state,
        additional_components=[])
    # Reproduce the env's per-seed test path: seeds 0-4, 5 tasks each.
    turns = total = 0
    for seed in range(5):
        rng = np.random.default_rng(seed + 10000)
        tasks = gen.generate_tasks(
            num_tasks=5,
            rng=rng,
            possible_num_dominos=CFG.domino_test_num_dominos,
            possible_num_targets=CFG.domino_test_num_targets,
            possible_num_pivots=CFG.domino_test_num_pivots)
        turns += sum(is_turn(t.task, comp) for t in tasks)
        total += len(tasks)
    print(f"RESULT turns={turns}/{total} = {100.0 * turns / total:.1f}%")


if __name__ == "__main__":
    main()
