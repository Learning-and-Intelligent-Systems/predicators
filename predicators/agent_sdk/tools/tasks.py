"""Shared resolution of a tool's ``task_idx`` argument to a task.

Every task-scoped tool follows the same convention: an int ``task_idx``
indexes the train tasks (bounds-checked), and omitting it falls back to
the current solve/explore task. :func:`_resolve_task` is the single
implementation of that convention.
"""
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

from predicators.agent_sdk.tools.context import ToolContext
from predicators.agent_sdk.tools.results import _error_result
from predicators.structs import Task


@dataclass(frozen=True)
class ResolvedTask:
    """A tool call's resolved task plus how to refer to it.

    ``label`` follows the tools' display convention: the int train-task
    index, or the string ``"current"`` for the current solve/explore
    task. Handlers interpolate it directly into report text and pass it
    to ``_resolve_task_evaluator``; ``is_current`` is the boolean the
    capture guards read (it replaces the old ``task_idx == "current"``
    comparisons).
    """
    task: Task
    label: Union[int, str]
    is_current: bool

    @property
    def description(self) -> str:
        """Human-readable phrase: ``train task 3`` or ``current task``."""
        if self.is_current:
            return "current task"
        return f"train task {self.label}"


def _resolve_task(
    ctx: ToolContext, task_idx: Optional[int]
) -> Tuple[Optional[ResolvedTask], Optional[Dict[str, Any]]]:
    """Resolve a tool's ``task_idx`` argument (None ⇒ current task).

    Returns ``(resolved, error)`` with exactly one of the two set;
    ``error`` is a ready-to-return tool error result.
    """
    if task_idx is not None:
        if task_idx < 0 or task_idx >= len(ctx.train_tasks):
            return None, _error_result(
                f"Invalid task_idx {task_idx}. "
                f"Available: 0-{len(ctx.train_tasks)-1}")
        return ResolvedTask(task=ctx.train_tasks[task_idx],
                            label=task_idx,
                            is_current=False), None
    if ctx.current_task is not None:
        return ResolvedTask(task=ctx.current_task,
                            label="current",
                            is_current=True), None
    return None, _error_result("No task_idx provided and no current_task set.")
