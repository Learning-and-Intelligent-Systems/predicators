"""Run experiments from a YAML config, sequentially in the current terminal.

    python scripts/local/launch_simp.py -c example_basic.yaml

Pass ``--parallel`` to launch each experiment in its own macOS
Terminal window concurrently. See ``launch.py`` for the featureful
variant (branch checkout, logfile redirect).
"""
import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

# Add project root to sys.path so `scripts` is importable without PYTHONPATH=.
# parents[0] = scripts/local, parents[1] = scripts, parents[2] = project root
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position
from scripts.cluster_utils import config_to_cmd_flags, generate_run_configs

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _launch_in_new_terminal(cmd: str) -> None:
    """Open a new macOS Terminal window and run ``cmd`` in it.

    Writes the command to a temp ``.command`` script and ``open``s it,
    which macOS routes to Terminal.app as a fresh window. Using a temp
    file sidesteps quoting headaches from embedding ``cmd`` in
    ``osascript``.
    """
    if sys.platform != "darwin":
        raise RuntimeError(
            "--parallel currently only supports macOS Terminal.app; "
            f"detected platform: {sys.platform}")
    with tempfile.NamedTemporaryFile(mode="w",
                                     suffix=".command",
                                     prefix="predicators_run_",
                                     delete=False) as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {shlex.quote(str(_REPO_ROOT))}\n")
        f.write("export PYTHONHASHSEED=0\n")
        f.write(f"{cmd}\n")
        f.write("echo\n")
        f.write("echo '=== Command finished. Press enter to close. ==='\n")
        f.write("read\n")
        script_path = f.name
    os.chmod(script_path, 0o755)
    subprocess.Popen(["open", script_path])


def _main() -> None:
    # Set up argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str)
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Launch each run in its own macOS Terminal window "
        "(concurrent). Default is sequential in the current terminal.")
    args = parser.parse_args()

    cmds = []
    # Loop through all experiments
    for cfg in generate_run_configs(args.config):
        cmd_str = config_to_cmd_flags(cfg)
        if "use_classification_problem_setting" in cfg.flags:
            use_classification_problem_setting = cfg.flags[
                'use_classification_problem_setting']
        else:
            use_classification_problem_setting = False

        if use_classification_problem_setting:
            entry_point = "main_classification.py"
        else:
            entry_point = "main.py"
        # Use the absolute path to our Python interpreter so that
        # --parallel works regardless of which conda env the new
        # Terminal window's shell activates by default. Has no effect
        # on sequential mode (same interpreter either way).
        python_exe = shlex.quote(sys.executable)
        cmd = f"{python_exe} predicators/{entry_point} {cmd_str}"
        cmds.append(cmd)

    # run the command
    num_cmds = len(cmds)
    for i, cmd in enumerate(cmds):
        if args.parallel:
            print(f"********* LAUNCHING COMMAND {i+1} of {num_cmds} "
                  "in new Terminal window *********")
            _launch_in_new_terminal(cmd)
        else:
            print(f"********* RUNNING COMMAND {i+1} of {num_cmds} *********")
            subprocess.run(cmd, shell=True, check=False)


if __name__ == "__main__":
    _main()
