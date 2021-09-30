"""Main entry point for running approaches in environments.

Example usage:
    python main.py --env=configs/envs/cover_config.py \
        --approach=configs/approaches/random_actions_config.py
"""

from absl import app
from absl import flags
from ml_collections.config_flags import config_flags

# TODO: unify flags/FLAGS/CONFIG, then cover this file with a test

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("env")
config_flags.DEFINE_config_file("approach")


def main(_):
    """Main entry point for running approaches in environments.
    """
    print(f"Starting run with env {FLAGS.env.name}, "
          f"approach {FLAGS.approach.name}")


if __name__ == "__main__":  # pragma: no cover
    app.run(main)
