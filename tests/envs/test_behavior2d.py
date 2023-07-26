import matplotlib
import matplotlib.pyplot as plt

from predicators.envs.behavior2d import Behavior2DEnv
from predicators import utils

def test_behavior2d():
    utils.reset_config({
        "env": "behavior2d",
        "behavior_task_name": "defrosting_freezer",
        "num_train_tasks": 2,
        "num_test_tasks": 2
    })

    env = Behavior2DEnv()
    for task in env.get_train_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    for task in env.get_test_tasks():
        for obj in task.init:
            assert len(obj.type.feature_names) == len(task.init[obj])
    print(f"Predicates: {len(env.predicates)}")
    print(f"Types: {len(env.types)}")
    task = env.get_train_tasks()[0]
    print(f"Init: {len(utils.abstract(task.init, env.predicates))}")

    matplotlib.use('TkAgg')
    env.render_state_plt(task.init, task)
    plt.show()

if __name__ == '__main__':
    test_behavior2d()