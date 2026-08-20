"""Read-only inspection tools (views over ToolContext state).

The digest renderers at module level are the single formatting source
for this information: the tools below delegate to them, and the solve /
synthesis prompt builders inject the same digests directly into their
prompts (so sessions that drop the corresponding tools lose no
information and the wording cannot drift between surfaces).
"""
import json
import os
from typing import Any, Callable, Collection, Dict, Iterable, List, Optional, \
    Union

from predicators import utils
from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.results import _error_result, \
    _save_option_to_sandbox
from predicators.agent_sdk.tools.scene import render_pybullet_image


def render_types_digest(types: Iterable[Any]) -> str:
    """One line per type: name, parent, feature names."""
    lines = []
    for t in sorted(types, key=lambda t: t.name):
        features = ", ".join(
            t.feature_names) if t.feature_names else "(no features)"
        parent_str = f" (parent: {t.parent.name})" if t.parent else ""
        lines.append(f"- {t.name}{parent_str}: [{features}]")
    if not lines:
        return "No types defined."
    return "\n".join(lines)


def render_options_digest(options: Iterable[Any],
                          gt_options_ref_path: Optional[str] = None) -> str:
    """One line per option: typed signature plus parameter box/descriptions.

    The ``obj:type`` signature and parameter listing here are what the
    strict plan parser expects plans to match, so this exact rendering
    is shared by the prompts and the ``inspect_options`` tool.
    """
    lines = []
    for opt in sorted(options, key=lambda o: o.name):
        type_sig = ", ".join(t.name for t in opt.types)
        params_dim = opt.params_space.shape[0] if opt.params_space.shape else 0
        if params_dim > 0:
            low = opt.params_space.low.tolist()
            high = opt.params_space.high.tolist()
            if opt.params_description:
                desc = ", ".join(opt.params_description)
                param_info = f", params=[{desc}], low={low}, high={high}"
            else:
                param_info = (f", params_dim={params_dim}, "
                              f"low={low}, high={high}")
        else:
            param_info = ""
        lines.append(f"  {opt.name}({type_sig}{param_info})")
    if not lines:
        return "No options defined."
    if gt_options_ref_path:
        lines.append(f"\nOption definition source code: "
                     f"`{gt_options_ref_path}` (search for the option name).")
    return "\n".join(lines)


def render_task_digest(task: Any,
                       task_idx: Union[int, str],
                       predicates: Collection[Any],
                       include_goal_query_hint: bool = False) -> str:
    """Goal (NL preferred), initial atoms, objects, and init-state details for
    one task.

    ``task_idx`` is the header label; a string (e.g. ``"(current solve
    task)"``) is allowed for tasks outside the train list.
    ``include_goal_query_hint`` adds the ``is_goal_state`` /
    ``goal_holds`` pointer, which only makes sense (and is only
    numerically valid) in sessions whose exec namespace binds those
    names (synthesis ``run_python``), so pass it only with an integer
    ``task_idx``.
    """
    if task.goal_nl:
        goal_line = f"  Goal (natural language): {task.goal_nl}"
    else:
        goal_str = ", ".join(str(g) for g in sorted(task.goal))
        goal_line = f"  Goal: {{{goal_str}}}"
    init_atoms = utils.abstract(task.init, predicates)
    atoms_str = ", ".join(str(a) for a in sorted(init_atoms))
    objects = sorted(task.init, key=str)
    obj_str = ", ".join(f"{o.name}:{o.type.name}" for o in objects)
    state_str = task.init.pretty_str()
    hint_line = ""
    if include_goal_query_hint:
        hint_line = (f"  Goal achievement: query "
                     f"`is_goal_state(state, {task_idx})` or "
                     f"`train_tasks[{task_idx}].goal_holds(state)`.\n")
    return (f"Task {task_idx}:\n"
            f"{goal_line}\n"
            f"{hint_line}"
            f"  Initial atoms: {{{atoms_str}}}\n"
            f"  Objects: [{obj_str}]\n\n"
            f"Initial state details:\n{state_str}")


def render_trajectory_digest(trajectories: List[Any],
                             train_tasks: List[Any],
                             predicates: Collection[Any],
                             traj_idx: int,
                             include_states: bool = True,
                             include_atoms: bool = False,
                             max_timesteps: int = 10) -> str:
    """Header (provenance, goal, reached_goal) plus per-timestep state / atoms.

    / action for one trajectory.

    Raises ``ValueError`` on an out-of-range ``traj_idx``.
    """
    if not trajectories:
        raise ValueError("No trajectories available yet.")
    if traj_idx < 0 or traj_idx >= len(trajectories):
        raise ValueError(f"Invalid traj_idx {traj_idx}. "
                         f"Available: 0-{len(trajectories) - 1}")
    traj = trajectories[traj_idx]
    provenance = "demo" if traj.is_demo else "interaction"
    task_idx = traj._train_task_idx  # pylint: disable=protected-access
    header = (f"Trajectory {traj_idx}: {len(traj.states)} states, "
              f"{len(traj.actions)} actions  "
              f"[provenance={provenance}, task={task_idx}")
    if task_idx is not None and 0 <= task_idx < len(train_tasks):
        task = train_tasks[task_idx]
        reached = task.goal_holds(traj.states[-1])
        goal_str = ", ".join(str(g) for g in sorted(task.goal))
        header += f", reached_goal={reached}]"
        lines = [header, f"Goal: {{{goal_str}}}"]
    else:
        header += "]"
        lines = [header]

    for t_step, state in enumerate(traj.states[:max_timesteps]):
        lines.append(f"\n--- Timestep {t_step} ---")
        if include_states:
            lines.append("State:")
            lines.append(state.dict_str(indent=2, num_decimal_points=4))
        if include_atoms:
            atoms = utils.abstract(state, predicates)
            atoms_str = ", ".join(str(a) for a in sorted(atoms))
            lines.append(f"Atoms: {{{atoms_str}}}")
        if t_step < len(traj.actions):
            act = traj.actions[t_step]
            opt = act.get_option()
            lines.append(f"Action: {opt.name}({opt.objects})")

    if len(traj.states) > max_timesteps:
        lines.append(
            f"\n... ({len(traj.states) - max_timesteps} more timesteps)")
    return "\n".join(lines)


def _build_inspection_tools(ctx: ToolContext, _text_result: Callable,
                            tool: Callable) -> Dict[str, Any]:
    """Read-only inspection tools (views over ToolContext state)."""

    @tool("inspect_types", "List all object types and their features", {})
    async def inspect_types(_args: Dict[str, Any]) -> Dict[str, Any]:
        digest = render_types_digest(ctx.types)
        if digest == "No types defined.":
            return _text_result(digest)
        return _text_result("Current types:\n" + digest)

    @tool("inspect_predicates",
          "List all predicates and their type signatures", {})
    async def inspect_predicates(_args: Dict[str, Any]) -> Dict[str, Any]:
        lines = []
        for p in sorted(ctx.predicates, key=lambda p: p.name):
            type_sig = ", ".join(t.name for t in p.types)
            lines.append(f"- {p.name}({type_sig})")
        if not lines:
            return _text_result("No predicates defined.")
        return _text_result("Current predicates:\n" + "\n".join(lines))

    @tool("inspect_processes",
          "List all processes with conditions, effects, and delays", {})
    async def inspect_processes(_args: Dict[str, Any]) -> Dict[str, Any]:
        lines = []
        for proc in sorted(ctx.processes, key=lambda p: p.name):
            conds = ", ".join(str(a) for a in sorted(proc.condition_at_start))
            adds = ", ".join(str(a) for a in sorted(proc.add_effects))
            dels = ", ".join(str(a) for a in sorted(proc.delete_effects))
            lines.append(f"- {proc.name}\n"
                         f"    Conditions: {{{conds}}}\n"
                         f"    Add effects: {{{adds}}}\n"
                         f"    Delete effects: {{{dels}}}\n"
                         f"    Delay: {proc.delay_distribution}")
        if not lines:
            return _text_result("No processes defined.")
        return _text_result("Current processes:\n" + "\n".join(lines))

    @tool(
        "inspect_options",
        "List all options, or inspect a specific option in detail. "
        "When given an option_name, saves source code to "
        "./proposed_code/<name>.py in the sandbox for you to Read.",
        {
            "type": "object",
            "properties": {
                "option_name": {
                    "type":
                    "string",
                    "description":
                    "Name of a specific option to inspect. Saves its "
                    "source code to ./proposed_code/<name>.py. "
                    "Omit to list all options.",
                },
            },
        },
    )
    async def inspect_options(args: Dict[str, Any]) -> Dict[str, Any]:
        option_name = args.get("option_name")

        if option_name is None:
            # List all options (same digest the prompts inject).
            digest = render_options_digest(ctx.options)
            if digest == "No options defined.":
                return _text_result(digest)
            return _text_result("Current options:\n" + digest)

        # Detailed inspection of a specific option
        opt_map = {o.name: o for o in ctx.options}
        if option_name not in opt_map:
            return _error_result(f"Unknown option '{option_name}'. "
                                 f"Available: {sorted(opt_map.keys())}")

        opt = opt_map[option_name]
        type_sig = ", ".join(t.name for t in opt.types)
        dim = opt.params_space.shape[0] if opt.params_space.shape else 0

        lines = [f"## {opt.name}({type_sig})", ""]

        # Params space
        if dim > 0:
            lines.append(f"params_dim: {dim}")
            lines.append(f"params_low:  {opt.params_space.low.tolist()}")
            lines.append(f"params_high: {opt.params_space.high.tolist()}")
            if opt.params_description:
                lines.append(f"params_desc: {list(opt.params_description)}")
        else:
            lines.append("params_dim: 0 (no continuous parameters)")

        # Source code — point to existing file or reference
        if ctx.show_option_source:
            lines.append("")
            code_path = os.path.join(ctx.sandbox_dir, "proposed_code",
                                     f"{option_name}.py") \
                if ctx.sandbox_dir else None

            if code_path and os.path.exists(code_path):
                # Already saved (e.g. from propose_options)
                lines.append(
                    f"Source code: `./proposed_code/{option_name}.py`")
            elif ctx.gt_options_ref_path:
                # GT options — point to reference file instead of extracting
                lines.append(f"Definition code: `{ctx.gt_options_ref_path}` "
                             f"(search for \"{option_name}\")")
            else:
                # Extract source and save it
                # pylint: disable-next=import-outside-toplevel
                import inspect as _inspect
                code_parts = []
                for attr_name in ("policy", "initiable", "terminal"):
                    fn = getattr(opt, attr_name, None)
                    if fn is None:
                        continue
                    try:
                        src = _inspect.getsource(fn)
                        code_parts.append(f"# {attr_name}\n{src.rstrip()}")
                    except (TypeError, OSError):
                        code_parts.append(
                            f"# {attr_name}: (source not available)")

                if code_parts:
                    full_code = "\n\n".join(code_parts)
                    rel_path = _save_option_to_sandbox(ctx, option_name,
                                                       full_code)
                    if rel_path:
                        lines.append(f"Source code: `{rel_path}`")
                    else:
                        # No sandbox — inline as fallback
                        lines.append("### Source Code")
                        lines.append("```python")
                        lines.append(full_code)
                        lines.append("```")

        return _text_result("\n".join(lines))

    @tool(
        "inspect_trajectories",
        "Inspect trajectory data. Returns state features and/or atoms.",
        {
            "type": "object",
            "properties": {
                "traj_idx": {
                    "type": "integer",
                    "description": "Trajectory index (0-based)"
                },
                "include_states": {
                    "type": "boolean",
                    "description": "Include state feature dicts",
                    "default": True
                },
                "include_atoms": {
                    "type": "boolean",
                    "description": "Include abstract atoms",
                    "default": False
                },
                "max_timesteps": {
                    "type": "integer",
                    "description": "Max timesteps to show",
                    "default": 10
                },
            },
            "required": ["traj_idx"],
        },
    )
    async def inspect_trajectories(args: Dict[str, Any]) -> Dict[str, Any]:
        traj_idx = args["traj_idx"]
        include_states = args.get("include_states", True)
        include_atoms = args.get("include_atoms", False)
        max_timesteps = args.get("max_timesteps", 10)

        all_trajs = ctx.offline_trajectories + ctx.online_trajectories
        try:
            digest = render_trajectory_digest(all_trajs,
                                              ctx.train_tasks,
                                              ctx.predicates,
                                              traj_idx,
                                              include_states=include_states,
                                              include_atoms=include_atoms,
                                              max_timesteps=max_timesteps)
        except ValueError as e:
            return _error_result(str(e))
        return _text_result(digest)

    @tool(
        "inspect_train_tasks",
        "Inspect training tasks (goals, initial atoms, objects, state "
        "details, and optionally an image of the initial scene)",
        {
            "type": "object",
            "properties": {
                "task_idx": {
                    "type":
                    "integer",
                    "description":
                    "Task index (0-based). Omit to see summary of all.",
                },
                "include_image": {
                    "type":
                    "boolean",
                    "description":
                    "If true and task_idx is given, render and return an "
                    "image of the initial state (PyBullet envs only).",
                },
            },
        },
    )
    async def inspect_train_tasks(args: Dict[str, Any]) -> Dict[str, Any]:
        task_idx = args.get("task_idx")
        include_image = args.get("include_image", False)

        if task_idx is not None:
            if task_idx < 0 or task_idx >= len(ctx.train_tasks):
                return _error_result(f"Invalid task_idx {task_idx}. "
                                     f"Available: 0-{len(ctx.train_tasks)-1}")
            task = ctx.train_tasks[task_idx]
            text = render_task_digest(task,
                                      task_idx,
                                      ctx.predicates,
                                      include_goal_query_hint=True)

            content: List[Dict[str, Any]] = [{"type": "text", "text": text}]

            if include_image:
                img_block = render_pybullet_image(ctx,
                                                  f"task_{task_idx}_init",
                                                  state=task.init)
                if img_block is not None:
                    content.append(img_block)

            return {"content": content}

        lines = [f"Total tasks: {len(ctx.train_tasks)}"]
        for i, task in enumerate(ctx.train_tasks[:10]):
            if task.goal_nl:
                lines.append(f"  Task {i}: {task.goal_nl}")
            else:
                goal_str = ", ".join(str(g) for g in sorted(task.goal))
                lines.append(f"  Task {i}: goal={{{goal_str}}}")
        if len(ctx.train_tasks) > 10:
            lines.append(f"  ... ({len(ctx.train_tasks) - 10} more tasks)")
        return _text_result("\n".join(lines))

    @tool("inspect_planning_results",
          "Get latest planning performance metrics", {})
    async def inspect_planning_results(
            _args: Dict[str, Any]) -> Dict[str, Any]:
        if not ctx.planning_results:
            return _text_result("No planning results available yet.")
        return _text_result(
            json.dumps(ctx.planning_results, indent=2, default=str))

    @tool("inspect_past_proposals",
          "Get summaries of proposals and retractions from all past "
          "iterations", {})
    async def inspect_past_proposals(_args: Dict[str, Any]) -> Dict[str, Any]:
        if not ctx.iteration_history:
            return _text_result("No past proposals available yet.")
        lines = []
        for entry in ctx.iteration_history:
            lines.append(json.dumps(entry, indent=2, default=str))
        return _text_result("\n---\n".join(lines))

    return {
        "inspect_types": inspect_types,
        "inspect_predicates": inspect_predicates,
        "inspect_processes": inspect_processes,
        "inspect_options": inspect_options,
        "inspect_trajectories": inspect_trajectories,
        "inspect_train_tasks": inspect_train_tasks,
        "inspect_planning_results": inspect_planning_results,
        "inspect_past_proposals": inspect_past_proposals,
    }
