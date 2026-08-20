"""Model-dependent thinking configuration for Claude Agent SDK sessions.

Claude models split into two thinking regimes:

- claude-sonnet-5 / claude-opus-4-7 and later (incl. claude-opus-4-8,
  claude-fable-5): ``{"type": "enabled", "budget_tokens": N}`` is
  rejected with a 400; thinking is adaptive and depth is controlled via
  the SDK ``effort`` option.
- Older families (sonnet-4.x, haiku, opus <= 4.6, claude-3): manual
  extended thinking with an explicit token budget. Sonnet/Opus 4.6 also
  accept adaptive, but the fixed budget keeps a hard ceiling on thinking
  spend, which is exactly the overflow failure mode ("32000 output
  token" errors from unbounded thinking) that adaptive thinking was
  adopted to fix on sonnet-5.

This module is imported both host-side and inside the Docker sandbox
runner, so it must stay a pure function of the model name (no CFG).
"""

from typing import Any, Dict

# Thinking budget for legacy models running manual extended thinking.
# Must stay strictly below the session's max output tokens.
LEGACY_THINKING_BUDGET_TOKENS = 16000

# Substring markers for model families that predate adaptive thinking.
# Unknown/future model names default to adaptive, since every model from
# sonnet-5 / opus-4.7 onward rejects budget_tokens.
_LEGACY_MODEL_MARKERS = (
    "sonnet-4-",  # claude-sonnet-4-6, -4-5, -4-20250514
    "haiku",  # claude-haiku-4-5 and older (no adaptive support)
    "opus-4-0",
    "opus-4-1",
    "opus-4-5",
    "opus-4-6",
    "opus-4-2025",  # dated claude-opus-4-20250514
    "claude-3",
    "claude-2",
)


def resolve_thinking_config(model_name: str) -> Dict[str, Any]:
    """Return the ``thinking`` option appropriate for ``model_name``."""
    name = model_name.lower()
    if any(marker in name for marker in _LEGACY_MODEL_MARKERS):
        return {
            "type": "enabled",
            "budget_tokens": LEGACY_THINKING_BUDGET_TOKENS,
        }
    return {"type": "adaptive"}
