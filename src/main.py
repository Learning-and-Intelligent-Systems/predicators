"""Main entry point for running approaches in environments.

Example usage:
    python src/main.py --env cover --approach oracle --seed 0
"""

from predicators.src.args import parse_args
from predicators.src.settings import CFG
from predicators.src.envs import create_env
from predicators.src.approaches import create_approach, ApproachTimeout, \
    ApproachFailure
from predicators.src.datasets import create_dataset
from predicators.src import utils


def main() -> None:
    """Main entry point for running approaches in environments.
    """
    # Parse & validate args
    args = parse_args()
    utils.update_config(args)
    # Create & seed classes
    env = create_env(CFG.env)
    approach = create_approach(CFG.approach, env.simulate, env.predicates,
                               env.options, env.types, env.action_space,
                               env.get_train_tasks())
    env.seed(CFG.seed)
    approach.seed(CFG.seed)
    # If approach is learning-based, get training datasets
    if approach.is_learning_based:
        dataset = create_dataset(env)
        approach.learn_from_offline_dataset(dataset)
    # Run approach
    for i, task in enumerate(env.get_test_tasks()):
        try:
            policy = approach.solve(task, timeout=500)
        except (ApproachTimeout, ApproachFailure) as e:
            print(f"Approach failed to solve task {i} with error: {e}")
            continue
        if utils.policy_solves_task(policy, task, env.simulate, env.predicates):
            print(f"Task {i} solved")
        else:
            print(f"Policy failed on task {i}")


if __name__ == "__main__":  # pragma: no cover
    main()
