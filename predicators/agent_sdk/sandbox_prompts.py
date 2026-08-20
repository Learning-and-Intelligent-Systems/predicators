"""Shared prompt text for agent session managers.

Both ``LocalSandboxSessionManager`` and ``DockerSessionManager`` use
these constants and builder functions so that prompt text stays in sync.
Sandbox directory scaffolding lives in :mod:`sandbox_setup`.
"""
from typing import Optional

from predicators.agent_sdk.tools import BUILTIN_TOOLS

# ---------------------------------------------------------------------------
# Prompt template builders
# ---------------------------------------------------------------------------

_BUILTIN_TOOLS_STR = ", ".join(BUILTIN_TOOLS)

_CLAUDE_MD_HEADER = """\
# Predicators Agent Sandbox

## Working Directory
Your working directory is the sandbox. All files you create MUST stay here.
Always use relative paths (e.g., `./my_script.py`).

## Python
The Python interpreter is `python3`. The predicators package is available
for import in your scripts:

    python3 -c "from predicators.structs import State, Type; print('OK')"

You can write and run test scripts in the sandbox:

    python3 my_experiment.py

## Reference Files
Curated source files are available in ./reference/ for you to read.
Read these to understand the APIs before writing code.

## Session Logs
Your past session queries and tool results are in ./session_logs/. Files are
named `<NNN>_<kind>_<timestamp>.md` where `<NNN>` is a run-wide counter and
`<kind>` is the query phase (e.g. `learn`, `test`, `explore`). The counter
comes first so alphabetical sort matches chronological order. Use Glob and
Read to review your earlier attempts when debugging:

    Glob ./session_logs/*.md
    Read ./session_logs/001_learn_*.md

## Scene Images
`evaluate_option_plan` automatically saves scene images to ./test_images/
after each step. You can Read them to inspect the spatial state of
the environment.

## Proposed Code
All proposal code and option source code is saved to ./proposed_code/.
Proposals are numbered (e.g. `001_propose_options_Pick.py`); saved
option source uses the option name (e.g. `Pick.py`):

    Glob ./proposed_code/*.py
    Read ./proposed_code/001_propose_options_Pick.py
"""

_CLAUDE_MD_RULES = """\

## Rules
- Do NOT attempt to read or browse files outside the sandbox directory.
  This is enforced for the file tools AND for Bash and the Python
  execution tools (run_python / explore_python): commands or code
  containing absolute or `../` paths that leave the sandbox (or source
  introspection) are blocked. Use relative paths inside the sandbox.
- Do NOT modify files in ./reference/ — they are for reading only
- Write all your code, experiments, and tests in the sandbox
- Do NOT inspect predicators source code (e.g. via `inspect.getsource()`,
  `inspect.getfile()`, reading `.py` files from site-packages, or any other
  method). Use the MCP tools and reference files instead.
"""

_CLAUDE_MD_SOLVE_STRATEGY = """\

## Debugging Strategy
- **Visualize liberally** — {visualize_hint} It's free (no physics, no
  failure modes). When stuck on a step, STOP testing and visualize the
  object at several candidate positions and orientations to find the
  right region before spending more evaluate_option_plan calls.
- **Vary all parameters** — orientation and other non-position params
  affect both the outcome and whether the action succeeds.
- **Search coarse-to-fine** — spread initial attempts across the full
  parameter range. After 3 failures in a small neighborhood, jump to a
  different region.
"""

# The solve strategy's visualization pointer depends on whether the
# session has the probe: explore_python's sim.reset staging +
# sim.render overlays are the only visualization surface.
_VISUALIZE_HINT_PROBE = ("use explore_python (`sim.reset(mods={...})`, "
                         "then `sim.render(...)`).")
_VISUALIZE_HINT_GENERIC = ("render candidate layouts with whatever "
                           "visualization your tools provide.")

_CLAUDE_MD_SYNTHESIS_STRATEGY = """\

## Model-Learning Strategy

Trajectory numbers are evidence, not ground truth. Two states with nearly
identical recorded coordinates can be geometrically very different — an
object's recorded pose origin often does not coincide with the part that
actually drives the rule (a body center vs. an outlet on its side, a
joint base vs. an end-effector tip, a container origin vs. its opening,
a switch housing vs. its handle). Before encoding any geometric
threshold, render the scene and check what's actually where.

**Threshold-fitting protocol** — follow this whenever a predicate or rule
condition compares a recorded feature against a learned cutoff:

1. Bucket trajectory steps by whether the downstream effect actually
   occurred (the rule-relevant feature advanced, the goal-relevant
   quantity changed, etc.). Compute your candidate quantity at each step.
2. Inspect the two buckets' value ranges. They must separate by a clear
   margin. If they overlap, or the gap is narrower than roughly 5% of
   the value range, STOP — a knife-edge separator is a symptom, not a
   fit, and a threshold flush against the data boundary is rejected.
   The candidate quantity is measuring against the wrong reference
   point; do not widen the threshold to absorb the gap.
3. For any two-body geometric gate, default to a learned anchor offset
   in the fixture's LOCAL frame, rotated into the world frame by the
   fixture's `rot` (origin + R(rot) @ (local_dx, local_dy)), with
   local_dx/local_dy declared as ParamSpecs and shared between the rule
   and its gating predicate — not a raw origin-distance threshold. To
   find the offset, stage one representative state from each bucket
   with `sim.reset(task_idx=..., mods={...})` and use
   `sim.render(label, annotations=[...])` to overlay, on one render,
   the recorded object origin and the positions where the effect did
   vs. did not fire. The gap between the origin and the effect-firing
   cluster is the offset.
4. Re-derive the candidate quantity using the anchored reference and
   refit. Only commit once the buckets separate by a comfortable margin
   (well past the 5% knife-edge). If the fit drives local_dx/local_dy to
   ~0, the origin was the functional point after all — fine, keep them.

**Other times to render the scene:**
- A new predicate is proposed: render a state where it should be true
  and one where it should be false to sanity-check the definition.
- A predicate's classifier looks right numerically but downstream signal
  (refinement success, residual reduction, plan completion) doesn't
  follow — the predicate is firing in the wrong places.
- You're choosing between candidate reference points (body center vs.
  contact surface, frame origin vs. tool tip, etc.).

`sim.reset` staging and `sim.render` overlays are free (no physics, no
failure modes). Reach for them before, not after, you commit a numeric
fit.
"""


def build_claude_md(phase: Optional[str] = None) -> str:
    """Build the CLAUDE.md content written into the sandbox directory.

    Args:
        phase: ``"synthesis"`` selects the model-learning strategy block;
            anything else (including ``None`` and ``"solve"``) selects the
            solve-time debugging block. The choice is reflected in the file
            written into the sandbox so the agent reads phase-appropriate
            guidance every turn.
    """
    # pylint: disable-next=import-outside-toplevel
    from predicators.agent_sdk.config import ToolSurfaceConfig
    if phase == "synthesis":
        strategy = _CLAUDE_MD_SYNTHESIS_STRATEGY
    else:
        if ToolSurfaceConfig.from_cfg().use_explore_python:
            hint = _VISUALIZE_HINT_PROBE
        else:
            hint = _VISUALIZE_HINT_GENERIC
        strategy = _CLAUDE_MD_SOLVE_STRATEGY.format(visualize_hint=hint)
    return _CLAUDE_MD_HEADER + strategy + _CLAUDE_MD_RULES


def build_sandbox_system_prompt(
    env_description: str = "a local sandbox environment",
    workspace_description: str = "the current directory",
    ref_path: str = "./reference/",
) -> str:
    """Build the system prompt suffix appended for sandbox sessions.

    Args:
        env_description: Short description of the sandbox environment.
        workspace_description: How the workspace directory is described.
        ref_path: Path to reference files shown in examples.
    """
    return f"""

## Sandbox Environment
You are running in {env_description}. You have the following
built-in tools available: {_BUILTIN_TOOLS_STR}.

Your workspace is {workspace_description}. All file operations (Read, Write,
Edit, Glob, Grep) are restricted to this directory.

### Writing and Running Code
You can write Python scripts and execute them with `python3`:
```
python3 my_script.py
```
The predicators package is importable in your scripts:
```python
from predicators.structs import State, Type, Object, Predicate
from predicators.structs import ParameterizedOption, Action
```

### Reference Files
Curated API reference files are in {ref_path}.
Read these files to understand the system APIs before writing code.

### Session Logs
Your past queries and tool results are saved in ./session_logs/ as markdown
files named `<NNN>_<kind>_<timestamp>.md` (e.g. `001_learn_...md`,
`002_test_...md`). The counter comes first so alphabetical sort matches
chronological order. Use Glob and Read to review previous attempts:
```
Glob ./session_logs/*.md
Read ./session_logs/001_learn_*.md
```

### Scene Images
`evaluate_option_plan` automatically saves scene images to ./test_images/
after each plan step for later review.

### Proposed Code
All proposal code and option source code is saved to ./proposed_code/.
Proposals are numbered (e.g. `001_propose_options_Pick.py`); saved
option source uses the option name (e.g. `Pick.py`):
```
Glob ./proposed_code/*.py
Read ./proposed_code/001_propose_options_Pick.py
```

### Rules
- Do NOT try to read or browse files outside the sandbox directory
- Do NOT modify files in ./reference/
- Do NOT inspect predicators source code via `inspect.getsource()`,
  `inspect.getfile()`, or by reading `.py` files from site-packages.
  Use MCP tools and reference files instead.
"""
