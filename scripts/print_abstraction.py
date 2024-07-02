"""Print the learned/GT predicates and action operators of an environment.

python scripts/print_abstraction.py --approach oracle --seed 0 --env
cover_multistep_options
"""
from pprint import pprint

from predicators import utils
from predicators.approaches import create_approach
from predicators.envs import create_new_env
from predicators.ground_truth_models import get_gt_options
from predicators.perception import create_perceiver
from predicators.settings import CFG

# ENVS = [
#     "cover_multistep_options",
#     "doors",
#     "stick_button",
#     "coffee"
# ]


def _main():

    env_name = CFG.env
    env = create_new_env(env_name, do_cache=True)
    preds = env.predicates
    approach_name = CFG.approach
    options = get_gt_options(env.get_name())
    env_train_tasks = env.get_train_tasks()
    perceiver = create_perceiver(CFG.perceiver)
    train_tasks = [perceiver.reset(t) for t in env_train_tasks]
    stripped_train_tasks = [
        utils.strip_task(task, preds) for task in train_tasks
    ]

    # Print example environment states
    print("# Example Environment States:")
    for i in range(1):
        print(f"## Task {i}")
        print("### Init State:")
        print(train_tasks[i].init.pretty_str())
        print("### Goal:")
        print(train_tasks[i].goal)
        print("### Simulator State:")
        pprint(train_tasks[i].init.simulator_state)
    print()

    # Print the GT predicates in PDDL format
    print("# Goal Predicates:")
    for pred in env.goal_predicates:
        print(pred.pddl_str())

    print("# All Predicates:")
    for pred in preds:
        print(pred.pddl_str())
    print()

    # Print the GT NSRTs
    approach = create_approach(approach_name, preds, options, env.types,
                               env.action_space, stripped_train_tasks)
    print("# NSRTs:")
    nsrts = approach._nsrts
    for nsrt in nsrts:
        print(nsrt)


if __name__ == "__main__":
    args = utils.parse_args()
    utils.update_config(args)

    print("# Environment: ", CFG.env)
    _main()
