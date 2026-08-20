"""Agent runner for Docker sandbox.

Executed inside the Docker container by DockerSessionManager.  Loads a
pickled ``QueryInput`` dict, creates a ``ClaudeSDKClient`` session with
both Claude built-in tools (Bash, Read, Write, Edit, Glob, Grep, Task*)
and custom predicator MCP tools, queries the agent, and pickles results
back to a shared directory.

The predicators source tree is mounted read-only at ``/opt/predicators``
(via ``PYTHONPATH``) for imports.  Curated reference files are available
at ``/sandbox/reference/``.  A writable sandbox is at ``/sandbox``.
PreToolUse hooks restrict the agent's built-in tools to ``/sandbox/``.

Usage (inside Docker)::

    PYTHONPATH=/opt/predicators python3 \
        /opt/predicators/predicators/agent_sdk/docker_agent_runner.py \
        /data/query_input.pkl /data/query_output.pkl
"""
import asyncio
import logging
import sys
import traceback
from typing import Any, Dict, List, Optional

import dill as pkl

# Bootstrap: import predicators.utils before anything else so that Python
# resolves the circular import chain (structs → utils → image_patch_wrapper
# → structs) in the correct order.  Without this, importing predicators.structs
# first causes image_patch_wrapper to try "from predicators.structs import Mask"
# while structs is still being initialized, raising an ImportError.
import predicators.utils  # noqa: F401, E402  # pylint: disable=unused-import

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# pylint: disable=wrong-import-position
from predicators.agent_sdk.log_formatter import \
    format_conversation_markdown  # noqa: E402
from predicators.agent_sdk.session_base import build_agent_options, \
    build_sandbox_mcp, stream_agent_response  # noqa: E402


async def _run_query(query_input: Dict[str, Any]) -> Dict[str, Any]:
    """Create a ClaudeSDKClient, query the agent, and collect responses."""
    from claude_agent_sdk import \
        ClaudeSDKClient  # pylint: disable=import-outside-toplevel

    ctx = query_input["tool_context"]
    tool_names: Optional[List[str]] = query_input.get("tool_names")

    # MCP server and options come from the same helpers the host-side
    # managers use; every value is an explicit query_input entry (no CFG
    # reads in-container).  An invalid reasoning_effort raises here and
    # surfaces as an error response, matching host-side validation.
    mcp_server, allowed_tools = build_sandbox_mcp(ctx, tool_names)
    options = build_agent_options(
        system_prompt=query_input["system_prompt"],
        model_name=query_input["model_name"],
        allowed_tools=allowed_tools,
        mcp_server=mcp_server,
        max_turns=query_input.get("max_turns", 20),
        # Sent by DockerSessionManager from its SessionConfig; the 20MB
        # fallback only covers pickles from older hosts.
        max_buffer_size=query_input.get("max_buffer_size", 20 * 1024 * 1024),
        reasoning_effort=str(query_input.get("reasoning_effort", "")),
    )

    client = ClaudeSDKClient(options=options)
    await client.connect()

    # Incremental log file path (on shared /data or /log volume)
    log_path = query_input.get("log_path")
    log_meta = {"query": query_input.get("message", "")}

    def _flush_log(collected: List[Dict[str, Any]]) -> None:
        """Write current conversation state as markdown to the log file."""
        if not log_path:
            return
        try:
            content = format_conversation_markdown(collected,
                                                   title="Docker Query",
                                                   meta=log_meta)
            with open(log_path, "w", encoding="utf-8") as lf:
                lf.write(content)
        except Exception:  # pylint: disable=broad-except
            pass  # Don't let logging errors break the agent

    # Docker-specific stderr reporting for real-time host visibility
    # (the host streams container stderr into its own log).
    def _report_block(dt: float, preview: str) -> None:
        print(f"[+{dt:.2f}s] {preview}", file=sys.stderr, flush=True)

    def _report_result(entry: Dict[str, Any]) -> None:
        print(
            f"Agent iteration complete. "
            f"Turns: {entry.get('num_turns', '?')}, "
            f"Cost: ${entry.get('total_cost_usd', '?')}",
            file=sys.stderr,
            flush=True)

    try:
        collected = await stream_agent_response(
            client,
            query_input["message"],
            log_label="Docker runner",
            report_block=_report_block,
            on_result=_report_result,
            flush=_flush_log,
        )
    finally:
        try:
            await client.disconnect()
        except Exception:  # pylint: disable=broad-except
            pass

    return {
        "responses": collected,
        "iteration_proposals": ctx.iteration_proposals,
    }


def _rehash_objects_after_unpickle(ctx: Any) -> None:
    """Fix stale Object hash caches after cross-process unpickling.

    ``Object.__hash__`` returns a ``cached_property`` (``_hash``) that
    stores ``hash(str(self))``.  Python randomises string hashes across
    processes (PYTHONHASHSEED), so cached values from the *pickling*
    process are stale here.  When the option-model simulator later
    creates fresh Objects (e.g. ``self._robot`` in ``_get_state``),
    their hashes differ from the unpickled Objects, causing KeyError on
    ``State.data`` dict lookups.

    Fix: clear every Object's cached ``_hash`` (and ``_str``) so it is
    re-computed with the current process's hash seed, then rebuild every
    ``State.data`` dict so its internal hash-table is consistent.
    """
    from predicators.structs import \
        State  # pylint: disable=import-outside-toplevel

    seen: set = set()

    def _clear(obj: Any) -> None:
        oid = id(obj)
        if oid in seen:
            return
        seen.add(oid)
        obj.__dict__.pop("_hash", None)
        obj.__dict__.pop("_str", None)

    def _process_state(state: Any) -> None:
        if state is None or not isinstance(state, State):
            return
        for obj in list(state.data.keys()):
            _clear(obj)
        # Rebuild dict so Python re-hashes keys with current seed. A
        # comprehension (not ``dict(...)``) is load-bearing: ``dict(d)``
        # copies each entry's stored hash without calling ``__hash__``,
        # so it would preserve exactly the stale table this repairs.
        # pylint: disable-next=unnecessary-comprehension
        state.data = {obj: vals for obj, vals in state.data.items()}

    def _process_atoms(atoms: Any) -> None:
        for atom in atoms:
            for obj in atom.objects:
                _clear(obj)

    def _process_task(task: Any) -> None:
        # Task has .init (State) and .goal (Set[GroundAtom])
        # EnvironmentTask has .init_obs and .goal_description
        if hasattr(task, "init"):
            _process_state(task.init)
        if hasattr(task, "init_obs"):
            _process_state(task.init_obs)
        for attr in ("goal", "alt_goal", "goal_description", "alt_goal_desc"):
            atoms = getattr(task, attr, None)
            # goal_description may be a plain NL string on
            # EnvironmentTask; only atom collections carry Objects.
            if atoms and not isinstance(atoms, str):
                _process_atoms(atoms)

    # Train tasks
    for task in getattr(ctx, "train_tasks", []):
        _process_task(task)

    # Current task
    if ctx.current_task is not None:
        _process_task(ctx.current_task)

    # Example state
    _process_state(getattr(ctx, "example_state", None))

    # Trajectories
    for traj in (getattr(ctx, "offline_trajectories", []) +
                 getattr(ctx, "online_trajectories", [])):
        for state in traj.states:
            _process_state(state)


def main() -> None:
    """Entry point for Docker agent runner."""
    if len(sys.argv) != 3:
        print(f"Usage: {sys.argv[0]} <input.pkl> <output.pkl>",
              file=sys.stderr)
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    logger.info("Docker agent runner starting: input=%s output=%s", input_path,
                output_path)

    # Load query input
    with open(input_path, "rb") as f:
        query_input = pkl.load(f)

    # Restore host CFG settings (arg-specific settings like
    # max_num_steps_option_rollout are not set by default import)
    if "cfg_snapshot" in query_input:
        from predicators.settings import \
            CFG  # pylint: disable=import-outside-toplevel
        for k, v in query_input["cfg_snapshot"].items():
            setattr(CFG, k, v)

    # Fix stale Object hash caches from cross-process pickling.
    ctx = query_input.get("tool_context")
    if ctx is not None:
        _rehash_objects_after_unpickle(ctx)

    # Recreate option model — the simulator (e.g. PyBullet physics
    # server) is process-local and cannot survive pickling.
    if ctx is not None and ctx.option_model is not None:
        from predicators.option_model import \
            create_option_model  # pylint: disable=import-outside-toplevel
        from predicators.settings import \
            CFG as _cfg  # pylint: disable=import-outside-toplevel
        logger.info("Recreating option model (%s) inside Docker...",
                    _cfg.option_model_name)
        ctx.option_model = create_option_model(
            _cfg.option_model_name,
            skip_residual_dynamics=_cfg.agent_planner_use_base_simulator)
        # Sync with all options in context (GT + any previously proposed)
        # after the model has its physics server set up.
        ctx.option_model._name_to_parameterized_option = {  # pylint: disable=protected-access
            o.name: o
            for o in ctx.options
        }

    # Recreate SkillConfig in skill_factory_context — the robot's
    # physics_client_id is process-local and stale after pickling.
    if (ctx is not None
            and ctx.skill_factory_context.get("skill_config") is not None):
        from predicators.settings import \
            CFG as _cfg  # pylint: disable=import-outside-toplevel
        if _cfg.env.startswith("pybullet"):
            try:
                # pylint: disable=import-outside-toplevel,reimported
                from predicators import utils as _utils
                from predicators.envs.base_env import BaseEnv
                from predicators.envs.pybullet_env import PyBulletEnv
                from predicators.ground_truth_models.skill_factories import \
                    SkillConfig

                # Find the PyBulletEnv subclass (envs already imported above
                # by create_option_model → create_new_env).
                env_cls = None
                for cls in _utils.get_all_subclasses(BaseEnv):
                    if (not cls.__abstractmethods__
                            and issubclass(cls, PyBulletEnv)
                            and cls.get_name() == _cfg.env):
                        env_cls = cls
                        break

                if env_cls is None:
                    logger.warning(
                        "Could not find PyBulletEnv for %s; "
                        "skill_config NOT recreated", _cfg.env)
                else:
                    _, robot, _ = env_cls.initialize_pybullet(using_gui=False)
                    ctx.skill_factory_context["skill_config"] = SkillConfig(
                        robot=robot,
                        open_fingers_joint=robot.open_fingers,
                        closed_fingers_joint=robot.closed_fingers,
                        fingers_state_to_joint=(
                            env_cls._fingers_state_to_joint),  # pylint: disable=protected-access
                        max_vel_norm=_cfg.pybullet_max_vel_norm,
                        ik_validate=_cfg.pybullet_ik_validate,
                        robot_init_tilt=getattr(env_cls, 'robot_init_tilt',
                                                0.0),
                        robot_init_wrist=getattr(env_cls, 'robot_init_wrist',
                                                 0.0),
                    )
                    logger.info(
                        "Recreated SkillConfig inside Docker for %s "
                        "(physics_client_id=%d)", _cfg.env,
                        robot.physics_client_id)
            except Exception as e:  # pylint: disable=broad-except
                logger.error("Failed to recreate SkillConfig in Docker: %s",
                             e,
                             exc_info=True)

    logger.info("Loaded query input: message length=%d, model=%s",
                len(query_input.get("message", "")),
                query_input.get("model_name", "?"))

    # Run the query
    try:
        query_output = asyncio.run(_run_query(query_input))
    except Exception as e:  # pylint: disable=broad-except
        logger.error("Fatal error in agent runner: %s\n%s", e,
                     traceback.format_exc())
        query_output = {
            "responses": [{
                "type": "error",
                "error": str(e)
            }],
            "iteration_proposals": None,
        }

    # Save output
    with open(output_path, "wb") as f:
        pkl.dump(query_output, f)

    logger.info("Docker agent runner finished: %d responses",
                len(query_output.get("responses", [])))


if __name__ == "__main__":
    main()
