"""Persistent per-run solve journal.

One markdown file per run (``<sandbox>/journal.md``) that accumulates
knowledge across solve attempts and test tasks: the harness auto-records
each task's goal + initial state (one entry, at the top of the task's
section) and each attempt's outcome and captured or best refused plan,
and the agent records lessons via the ``record_journal`` MCP tool.
Fresh-context solve sessions read
the journal from their prompt, so knowledge travels through this curated
channel instead of raw transcript history (which also carries the wrong
conclusions of failed attempts - the anchoring failure mode).

Entries are size-capped and the tool guidance asks for facts and
measurements rather than verdicts: a recorded "X is impossible" from a
failed attempt would re-import exactly the anchoring the fresh context
is meant to shed, while "tried yaws 0-15 deg at x in [0.50, 0.54], all
stopped >=5 cm short" steers the next attempt without foreclosing it.

Phase lifecycle: learning-phase entries persist for the whole run and
accumulate across online-learning cycles, so every evaluation starts
from all learning knowledge so far. Test-phase entries live only for
their own evaluation: at ``end_test_phase`` the approach archives the
full journal to the run's log dir (outside the sandbox, so the agent
cannot read it) and rolls the file back to its pre-test content via
:func:`read_raw` / :func:`restore` - entries recorded while solving one
evaluation's test tasks must not leak into the next evaluation.
"""

from __future__ import annotations

import os
from typing import Optional

JOURNAL_FILENAME = "journal.md"

# Per-entry cap. Entries are meant to be skimmable bullet lists; a cap
# keeps one verbose attempt from crowding every later prompt.
MAX_ENTRY_CHARS = 2000

# Harness auto-entries get more room: the first entry per task embeds
# the init-state feature dict (the prompt's own representation) and a
# captured plan. The writer additionally orders the layout block last,
# so tail truncation at this cap can only ever cut layout, never the
# outcome or the captured plan.
MAX_AUTO_ENTRY_CHARS = 4000

# Cap on how much journal is injected into a solve prompt. Tail-biased:
# recent attempts (usually the same task) matter most.
MAX_PROMPT_CHARS = 8000


def journal_path(sandbox_dir: str) -> str:
    """Host path of the run's journal file."""
    return os.path.join(sandbox_dir, JOURNAL_FILENAME)


def append_entry(sandbox_dir: str,
                 header: str,
                 body: str,
                 max_chars: int = MAX_ENTRY_CHARS) -> Optional[str]:
    """Append one entry; returns a truncation notice or None.

    ``header`` becomes a ``### <header>`` line; ``body`` is written
    verbatim below it, truncated at ``max_chars`` (default
    :data:`MAX_ENTRY_CHARS`; harness auto-entries pass
    :data:`MAX_AUTO_ENTRY_CHARS`).
    """
    os.makedirs(sandbox_dir, exist_ok=True)
    note: Optional[str] = None
    body = body.strip()
    if len(body) > max_chars:
        body = body[:max_chars].rstrip()
        body += "\n[entry truncated at the per-entry size cap]"
        note = (f"entry truncated to {max_chars} chars - keep journal "
                "entries short and factual")
    with open(journal_path(sandbox_dir), "a", encoding="utf-8") as f:
        f.write(f"### {header.strip()}\n{body}\n\n")
    return note


def read_raw(sandbox_dir: Optional[str]) -> Optional[str]:
    """Exact journal file content, or None if no journal file exists.

    Unlike :func:`read_journal` there is no prompt trimming and the
    absent-file case is distinguishable from an empty file, so the
    result is a faithful snapshot for :func:`restore`.
    """
    if not sandbox_dir:
        return None
    path = journal_path(sandbox_dir)
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def restore(sandbox_dir: str, snapshot: Optional[str]) -> None:
    """Reset the journal file to a :func:`read_raw` snapshot.

    A ``None`` snapshot means no journal file existed, so the file is
    removed if present.
    """
    path = journal_path(sandbox_dir)
    if snapshot is None:
        if os.path.isfile(path):
            os.remove(path)
        return
    os.makedirs(sandbox_dir, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(snapshot)


def read_journal(sandbox_dir: Optional[str],
                 max_chars: int = MAX_PROMPT_CHARS) -> str:
    """Journal content for prompt injection ('' if absent or empty).

    Over ``max_chars`` the head is dropped at an entry boundary with a
    truncation marker, keeping the most recent entries intact.
    """
    if not sandbox_dir:
        return ""
    path = journal_path(sandbox_dir)
    if not os.path.isfile(path):
        return ""
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if len(content) <= max_chars:
        return content
    tail = content[-max_chars:]
    cut = tail.find("\n### ")
    if cut != -1:
        tail = tail[cut + 1:]
    return ("[journal truncated: older entries omitted, most recent "
            f"kept]\n{tail}")
