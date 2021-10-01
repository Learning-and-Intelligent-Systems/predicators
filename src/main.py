"""Main entry point for running approaches in environments.

Example usage:
    python src/main.py --env cover --approach oracle --seed 0
"""

from predicators.src.args import parse_args
from predicators.src.settings import CFG
from predicators.src.envs import create_env
from predicators.src.approaches import create_approach
from predicators.src.utils import update_config


def main() -> None:
    """Main entry point for running approaches in environments.
    """
    # Parse & validate args
    args = parse_args()
    update_config(args)
    # Create & seed classes
    env = create_env(CFG.env)
    approach = create_approach(CFG.approach, env.simulate, env.predicates,
                               env.options, env.types, env.action_space)
    env.seed(CFG.seed)
    approach.seed(CFG.seed)
    # Run approach
    for task in env.get_test_tasks():
        approach.solve(task, timeout=500)


if __name__ == "__main__":  # pragma: no cover
    main()
