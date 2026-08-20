"""Predicate-invention synthesis tools (predicate invention loop)."""
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from predicators.agent_sdk.proposal_exec import build_exec_context, \
    exec_code_safely, validate_predicate
from predicators.agent_sdk.synthesis_backend import PredicateSynthesisBackend
from predicators.agent_sdk.tools.params_view import _ParamsView
from predicators.agent_sdk.tools.results import _make_coercing_tool, \
    _make_spilling_text_result
from predicators.agent_sdk.tools.sandbox_guard import _scrub_host_paths
from predicators.agent_sdk.tools.snapshots import _ArtifactSnapshotter
from predicators.structs import LowLevelTrajectory, Predicate, State, Type


def create_predicate_synthesis_tools(
    predicates_file: str,
    predicates_versions_dir: str,
    approach: PredicateSynthesisBackend,
    trajectories: List[LowLevelTrajectory],
    cycle_index_provider: Optional[Callable[[], int]] = None,
) -> list:
    """Create the predicate-invention synthesis tool.

    Returns ``[evaluate_predicate_quality]``. The tool loads
    ``predicates.py`` fresh on each call (snapshotting into
    ``predicates_versions_dir`` as
    ``cycle_XXX_vers_YYY_predicates.py``), validates each
    ``Predicate``, mutates ``approach._learned_predicates`` so
    subsequent refinement calls see the agent's draft, and reports
    milestone behaviour over the demo trajectories.

    Args:
        predicates_file: Host path to the canonical ``predicates.py``
            file the agent edits.
        predicates_versions_dir: Directory for per-call snapshots
            (created on first use).
        approach: The ``AgentSimPredicateInventionApproach`` instance.
            Must expose ``_types``, ``_kept_initial_predicates``,
            ``_get_all_options()``, and ``_learned_predicates``.
        trajectories: Demo trajectories used for milestone reporting.
        cycle_index_provider: Callable returning the current cycle
            (1-indexed) at snapshot time. Defaults to a constant 0.
    """
    # pylint: disable=import-outside-toplevel
    import traceback  # pylint: disable=redefined-outer-name,reimported

    from claude_agent_sdk import tool as _sdk_tool
    tool = _make_coercing_tool(_sdk_tool)

    from predicators.code_sim_learning.fit_space import ParamSpec

    # pylint: enable=import-outside-toplevel
    # ``predicates_file`` lives at ``<sandbox>/predicates.py``, so its
    # parent is the sandbox root — spill oversize output there rather than
    # letting the agent SDK dump it outside the sandbox.
    _text = _make_spilling_text_result(os.path.dirname(predicates_file))
    _snapshotter = _ArtifactSnapshotter(
        live_file=predicates_file,
        versions_dir=predicates_versions_dir,
        artifact_name="predicates",
        cycle_index_provider=cycle_index_provider,
        missing_file_hint=("Use Write to create it with "
                           "LEARNED_PREDICATES = [...]."),
    )

    params_view = _ParamsView(approach._fitted_params)  # pylint: disable=protected-access

    def _snapshot_and_load_predicates(
        path: str,
    ) -> Tuple[List[Predicate], Optional[str], Optional[str], List[str]]:
        """Snapshot ``path`` then exec it into a fresh namespace.

        Returns ``(predicates, version_tag, error_msg, warnings)``.
        ``error_msg`` is ``None`` on success. Predicates that failed
        validation are excluded; ``warnings`` describes them.
        """
        raw, version_tag, err = _snapshotter.snapshot(path)
        if err is not None:
            return [], None, err, []
        assert raw is not None and version_tag is not None

        ctx = build_exec_context(
            types=approach._types,  # pylint: disable=protected-access
            predicates=approach._kept_initial_predicates,  # pylint: disable=protected-access
            options=approach._get_all_options(),  # pylint: disable=protected-access
            extra_context={
                "params": params_view,
                "ParamSpec": ParamSpec,
            })
        result, err = exec_code_safely(raw.decode("utf-8"), ctx,
                                       "LEARNED_PREDICATES")
        if err is not None:
            return [], version_tag, (f"[{version_tag}] Error executing "
                                     f"{path}:\n{err}"), []
        if not isinstance(result, list):
            return [], version_tag, (
                f"[{version_tag}] LEARNED_PREDICATES must be a list, "
                f"got {type(result).__name__}."), []

        kept_names = {
            p.name
            for p in approach._kept_initial_predicates  # pylint: disable=protected-access
        }
        example_state = (
            approach._train_tasks[0].init  # pylint: disable=protected-access
            if approach._train_tasks else None)  # pylint: disable=protected-access

        valid: List[Predicate] = []
        warnings: List[str] = []
        seen_names = set()
        for entry in result:
            if not isinstance(entry, Predicate):
                warnings.append(f"Skipped non-Predicate entry: {entry!r}")
                continue
            if entry.name in kept_names:
                warnings.append(f"Skipped '{entry.name}' (collides "
                                "with a kept env predicate).")
                continue
            if entry.name in seen_names:
                warnings.append(f"Skipped duplicate '{entry.name}'.")
                continue
            if example_state is not None:
                verr = validate_predicate(
                    entry,
                    approach._types,  # pylint: disable=protected-access
                    example_state)
                if verr is not None:
                    warnings.append(
                        f"Predicate '{entry.name}' failed validation: "
                        f"{verr}")
                    continue
            valid.append(entry)
            seen_names.add(entry.name)

        # Mutate approach state so sim.refine sees the draft.
        approach._learned_predicates = set(valid)  # pylint: disable=protected-access
        return valid, version_tag, None, warnings

    def _enumerate_groundings(
        state: State,
        pred_types: Sequence[Type],
        max_groundings: int,
    ) -> List[Tuple[Any, ...]]:
        """Distinct-object groundings of ``pred_types`` from ``state``.

        Capped at ``max_groundings``; sufficient for milestone
        reporting.
        """
        objs_by_type: Dict[str, List[Any]] = {}
        for obj in state:
            objs_by_type.setdefault(obj.type.name, []).append(obj)

        out: List[Tuple[Any, ...]] = []

        def rec(idx: int, picked: List[Any], used: set) -> None:
            if len(out) >= max_groundings:
                return
            if idx == len(pred_types):
                out.append(tuple(picked))
                return
            for c in objs_by_type.get(pred_types[idx].name, []):
                if id(c) in used:
                    continue
                used.add(id(c))
                picked.append(c)
                rec(idx + 1, picked, used)
                picked.pop()
                used.remove(id(c))
                if len(out) >= max_groundings:
                    return

        rec(0, [], set())
        return out

    @tool(
        "evaluate_predicate_quality",
        "Load LEARNED_PREDICATES (fresh from `predicates.py`) and "
        "report milestone behaviour over demo trajectories. For each "
        "predicate × each grounding, evaluates pred.holds(state) at "
        "every step and reports: coverage (ever-true / ever-false), "
        "transition counts, first-flip step, and monotonicity (ideal "
        "milestone flips False->True exactly once and stays true). "
        "After loading, the predicate set used by "
        "sim.refine is updated — so call this tool any "
        "time you edit predicates.py before re-running refinement. "
        "Snapshots the predicates file into predicates_versions/; "
        "output tagged [cycle_XXX_vers_YYY].",
        {
            "type": "object",
            "properties": {
                "max_trajectories": {
                    "type": "integer",
                    "description": "Max trajectories to scan "
                    "(default 10).",
                },
                "max_groundings_per_predicate": {
                    "type":
                    "integer",
                    "description":
                    "Max object groundings to evaluate "
                    "per predicate (default 4).",
                },
            },
        },
    )
    async def evaluate_predicate_quality(
            args: Dict[str, Any]) -> Dict[str, Any]:
        max_trajs = int(args.get("max_trajectories", 10))
        max_groundings = int(args.get("max_groundings_per_predicate", 4))

        try:
            preds, version_tag, err, warnings = (
                _snapshot_and_load_predicates(predicates_file))
        except Exception:  # pylint: disable=broad-except
            return _text(f"Error loading predicates.py:\n"
                         f"{_scrub_host_paths(traceback.format_exc())}")

        if err is not None:
            return _text(err)

        prefix = f"[{version_tag}]"
        scanned = trajectories[:max_trajs]
        lines = [
            f"{prefix} Predicate quality report — "
            f"{len(preds)} predicate(s), {len(scanned)} trajector(ies), "
            f"up to {max_groundings} grounding(s)/predicate.",
        ]
        if warnings:
            lines.append("")
            lines.append("Warnings (entries skipped during load):")
            for w in warnings:
                lines.append(f"  - {w}")

        if not preds:
            lines.append("")
            lines.append("LEARNED_PREDICATES is empty — add "
                         "Predicate(...) entries to predicates.py.")
            return _text("\n".join(lines))

        # Pre-materialise per-step `latent` per trajectory. For
        # recurrent approaches this rolls the trajectory through the
        # agent's simulator and produces `[lat_0, lat_1, ...]` so
        # latent-aware predicates can evaluate against a meaningful
        # latent; for non-recurrent approaches it returns a list of
        # ``None``s and latent-aware classifiers see `latent=None`.
        materialise_latent_fn = getattr(approach, "materialise_latent", None)
        latent_per_traj: Dict[int, List[Optional[Dict[str, Any]]]] = {}
        for ti, traj in enumerate(scanned):
            if materialise_latent_fn is not None and traj.states:
                try:
                    latent_per_traj[ti] = materialise_latent_fn(traj)
                except Exception:  # pylint: disable=broad-except
                    # Approach-side materialisation crashed — fall back
                    # to None so observation-only predicates still work.
                    latent_per_traj[ti] = [None] * len(traj.states)
            else:
                latent_per_traj[ti] = [None] * len(traj.states)

        for pred in preds:
            sig = ", ".join(t.name for t in pred.types)
            lines.append("")
            lines.append(f"{pred.name}({sig})")
            ever_true = ever_false = False
            flip_records: List[Tuple[int, Tuple[Any, ...], int, int,
                                     bool]] = []
            no_grounding_trajs = 0
            error_lines: List[str] = []
            for ti, traj in enumerate(scanned):
                if not traj.states:
                    continue
                groundings = _enumerate_groundings(traj.states[0], pred.types,
                                                   max_groundings)
                if not groundings:
                    no_grounding_trajs += 1
                    continue
                lats = latent_per_traj[ti]
                for gr in groundings:
                    try:
                        truth = [
                            pred.holds(s, gr, latent=lats[si])
                            for si, s in enumerate(traj.states)
                        ]
                    except Exception:  # pylint: disable=broad-except
                        last_line = traceback.format_exc().strip().splitlines(
                        )[-1]
                        error_lines.append(
                            f"  traj {ti} ({', '.join(o.name for o in gr)})"
                            f": classifier raised — {last_line}")
                        continue
                    if any(truth):
                        ever_true = True
                    if not all(truth):
                        ever_false = True
                    flips_up = sum(1 for i in range(1, len(truth))
                                   if truth[i] and not truth[i - 1])
                    flips_dn = sum(1 for i in range(1, len(truth))
                                   if truth[i - 1] and not truth[i])
                    flip_records.append(
                        (ti, gr, flips_up, flips_dn, truth[-1]))

            coverage = ("ever-T + ever-F" if ever_true and ever_false else (
                "always-T (likely useless)" if ever_true else
                ("always-F (likely useless)" if ever_false else "no-data")))
            n_records = len(flip_records)
            n_monotone = sum(1 for _, _, up, dn, _ in flip_records
                             if up == 1 and dn == 0)
            n_never_flipped = sum(1 for _, _, up, dn, _ in flip_records
                                  if up == 0 and dn == 0)
            lines.append(f"  coverage: {coverage}")
            lines.append(f"  groundings scored: {n_records}, "
                         f"monotone (1↑ 0↓): {n_monotone}, "
                         f"never-flipped: {n_never_flipped}, "
                         f"no-grounding trajs: {no_grounding_trajs}")
            for ti, gr, up, dn, final in flip_records[:max_trajs]:
                names = ", ".join(o.name for o in gr)
                lines.append(f"  traj {ti} ({names}): ↑={up}, ↓={dn}, "
                             f"final={'T' if final else 'F'}")
            for el in error_lines[:max_trajs]:
                lines.append(el)

        return _text("\n".join(lines))

    return [evaluate_predicate_quality]
