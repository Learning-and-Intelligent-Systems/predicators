"""Launch experiments defined in config files locally.

Reads a YAML config from ``scripts/configs/``, expands it into one
shell command per experiment via ``scripts.cluster_utils``, preps the
repo (git checkout / pull on the chosen branch), then dispatches the
commands either sequentially or in parallel.

Run from the project root — no ``PYTHONPATH=.`` prefix needed, the
script bootstraps ``sys.path`` itself.

Flags
-----
``--config <name.yaml>`` (required)
    Config file name under ``scripts/configs/``. Each entry becomes
    one experiment command.

``--branch <name>`` (default ``DEFAULT_BRANCH``)
    Branch to check out / pull before running. Passed to
    ``get_cmds_to_prep_repo``.

``--parallel``
    Launch each experiment in its own macOS Terminal window for
    concurrent execution. Default is sequential in the current
    terminal. Requires macOS (uses Terminal.app).

Behavior
--------
* Logs go to ``logs/<config_to_logfile(cfg)>``. In sequential mode
  output is redirected with ``>``; in parallel mode it's ``tee``'d
  so each Terminal shows output live AND the logfile is written.
* Parallel-mode Terminals pause on ``read`` after the run finishes,
  so you can inspect the final state before closing the window.
* Each parallel-mode Terminal exports ``PYTHONHASHSEED=0`` (required
  by the codebase per ``README.md``); sequential mode inherits it
  from the parent shell.
* Entry point is ``predicators/main.py`` by default, or
  ``predicators/train_refinement_estimator.py`` when
  ``cfg.train_refinement_estimator`` is truthy.

Examples
--------
Sequential, current terminal::

    python scripts/local/launch.py --config example_basic.yaml

Sequential on a specific branch::

    python scripts/local/launch.py --config example_basic.yaml \\
        --branch my-feature-branch

Parallel, one Terminal window per experiment::

    python scripts/local/launch.py --config example_basic.yaml --parallel

See ``scripts/local/launch_simp.py`` for a simpler variant that skips
the git prep and always runs in the current terminal.
"""

import argparse
import os
import shlex
import subprocess
import sys
import tempfile
from pathlib import Path

# Bootstrap sys.path so ``scripts.cluster_utils`` is importable without
# the caller having to set PYTHONPATH=. — parents[0] = scripts/local,
# parents[1] = scripts, parents[2] = project root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

# pylint: disable=wrong-import-position
from scripts.cluster_utils import DEFAULT_BRANCH, config_to_cmd_flags, \
    config_to_logfile, generate_run_configs, get_cmds_to_prep_repo

_REPO_ROOT = Path(__file__).resolve().parents[2]


def _launch_in_new_terminal(cmd: str) -> None:
    """Open a new macOS Terminal window and run ``cmd`` in it.

    Writes the command to a temp ``.command`` script and ``open``s it,
    which macOS routes to Terminal.app as a fresh window. Using a temp
    file sidesteps the quoting headaches of embedding ``cmd`` directly
    into ``osascript``.
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
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--branch", type=str, default=DEFAULT_BRANCH)
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="Launch each run in its own macOS Terminal window "
        "(concurrent). Default is sequential in the current terminal.")
    args = parser.parse_args()
    # Prepare the repo.
    for cmd in get_cmds_to_prep_repo(args.branch):
        subprocess.run(cmd, shell=True, check=False)
    # Create the run commands.
    cmds = []
    for cfg in generate_run_configs(args.config):
        cmd_flags = config_to_cmd_flags(cfg)
        logfile = os.path.join("logs", config_to_logfile(cfg))
        if cfg.train_refinement_estimator:
            entry_point = "train_refinement_estimator.py"
        else:
            entry_point = "main.py"
        # Use the absolute path to our Python interpreter so that
        # --parallel works regardless of which conda env the new
        # Terminal window's shell activates by default. Has no effect
        # on sequential mode (same interpreter either way).
        python_exe = shlex.quote(sys.executable)
        if args.parallel:
            # ``tee`` so the new Terminal shows output live AND the
            # logfile is still written for later review.
            cmd = (f"{python_exe} predicators/{entry_point} {cmd_flags} "
                   f"2>&1 | tee {logfile}")
        else:
            cmd = (f"{python_exe} predicators/{entry_point} {cmd_flags} "
                   f"> {logfile}")
        cmds.append(cmd)
    # Run the commands.
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
