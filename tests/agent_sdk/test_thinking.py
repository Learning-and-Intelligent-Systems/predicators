"""Tests for model-dependent thinking configuration resolution.

Characterizes ``resolve_thinking_config``: legacy model families (per
``_LEGACY_MODEL_MARKERS``) get manual extended thinking with a fixed
token budget, while sonnet-5 / opus-4.7+ / unknown future models get
adaptive thinking.
"""
import pytest

from predicators.agent_sdk.thinking import LEGACY_THINKING_BUDGET_TOKENS, \
    resolve_thinking_config

_LEGACY_MODELS = [
    "claude-sonnet-4-6",
    "claude-sonnet-4-5",
    "claude-sonnet-4-20250514",
    "claude-haiku-4-5",
    "claude-haiku-3-5",
    "claude-opus-4-0",
    "claude-opus-4-1",
    "claude-opus-4-5",
    "claude-opus-4-6",
    "claude-opus-4-20250514",
    "claude-3-5-sonnet-20241022",
    "claude-2.1",
]

_ADAPTIVE_MODELS = [
    "claude-sonnet-5",
    "claude-sonnet-5-20260203",
    "claude-opus-4-7",
    "claude-opus-4-8",
    "claude-fable-5",
    # Unknown/future names default to adaptive.
    "claude-nova-9",
    "some-future-model",
]


@pytest.mark.parametrize("model_name", _LEGACY_MODELS)
def test_legacy_models_get_budgeted_thinking(model_name):
    """Legacy families resolve to manual extended thinking with a budget."""
    config = resolve_thinking_config(model_name)
    assert config == {
        "type": "enabled",
        "budget_tokens": LEGACY_THINKING_BUDGET_TOKENS,
    }


@pytest.mark.parametrize("model_name", _ADAPTIVE_MODELS)
def test_adaptive_models_get_adaptive_thinking(model_name):
    """Sonnet-5+/opus-4.7+/unknown models resolve to adaptive thinking."""
    config = resolve_thinking_config(model_name)
    assert config == {"type": "adaptive"}
    assert "budget_tokens" not in config


def test_matching_is_case_insensitive():
    """Model names are lowercased before marker matching."""
    assert resolve_thinking_config("Claude-Sonnet-4-6")["type"] == "enabled"
    assert resolve_thinking_config("CLAUDE-HAIKU-4-5")["type"] == "enabled"
    assert resolve_thinking_config("Claude-Fable-5")["type"] == "adaptive"


def test_legacy_budget_below_max_output_tokens():
    """The legacy budget stays strictly below the 32k output-token cap."""
    assert 0 < LEGACY_THINKING_BUDGET_TOKENS < 32000
