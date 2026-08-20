"""Shared dataclasses for agent plan sketches.

Split out of ``bilevel_sketch`` (see that module's docstring for the
full layout) so the parsing side (``sketch_parsing``, which constructs
``SketchStep`` and ``GroundSampler`` from sketch text), the refinement
side (``sketch_refinement``, which consumes them), and forward plan
execution (``plan_execution``) can all import the types without a cycle.
"""
import dataclasses
import logging
from typing import Optional, Sequence, Set

import numpy as np
from gym.spaces import Box

from predicators.structs import GroundAtom, Object, ParameterizedOption, \
    ParameterizedSampler, State


@dataclasses.dataclass
class GroundSampler:
    """Per-step (ground) sampler compiled from a sketch annotation.

    The ground level of the two-level sampler hierarchy that
    ``_draw_params`` consults: ground sampler (this, most specific) >
    learned parameterized sampler (``parameterized_samplers``, keyed by
    option name) > uniform. A parameterized sampler is authored once
    and sees every ground call of its option; a ground sampler is
    declared inline for ONE step of ONE sketch and dies with the call -
    it lives on the ``SketchStep`` rather than in the option-name-keyed
    registry, which could not hold different distributions for two
    same-option steps in one sketch.

    Two kinds, one per instance:
    - window (``center`` + ``width`` set): the uniform box a
      ``~ [w1, w2]`` region annotation declares around the step's
      proposed params;
    - code (``fn`` + ``name`` set): an agent-written function that a
      ``~ my_sampler`` annotation references by name (loaded fresh per
      refine call from the sandbox's ``GROUND_SAMPLERS``); it shares
      the parameterized-sampler call signature, so it can shape any
      state-conditioned distribution.
    """
    center: Optional[np.ndarray] = None
    width: Optional[np.ndarray] = None
    fn: Optional[ParameterizedSampler] = None
    name: str = ""

    def __post_init__(self) -> None:
        window = self.center is not None and self.width is not None
        assert window != (self.fn is not None), \
            "GroundSampler is either a window (center+width) or a code fn"

    def draw(self, state: State, rng: np.random.Generator, box: Box,
             objects: Sequence[Object],
             subgoal_atoms: Set[GroundAtom]) -> Optional[np.ndarray]:
        """Draw params from the window or the code fn, clipped to ``box``.

        Returns ``None`` when a code fn misbehaves (raises or returns a
        wrong-shaped array); the caller falls back to uniform sampling
        for that draw, mirroring the parameterized-sampler fallback.
        """
        if self.fn is not None:
            try:
                raw = self.fn(state, subgoal_atoms, rng, list(objects))
                params = np.asarray(raw, dtype=np.float32).reshape(-1)
            except Exception as e:  # pylint: disable=broad-except
                logging.warning(
                    "Ground sampler '%s' raised %s: %s; falling back to "
                    "uniform sampling for this draw.", self.name,
                    type(e).__name__, e)
                return None
            if params.shape != (box.shape[0], ):
                logging.warning(
                    "Ground sampler '%s' returned shape %s, expected "
                    "(%d,); falling back to uniform sampling for this "
                    "draw.", self.name, params.shape, box.shape[0])
                return None
            return np.clip(params, box.low, box.high).astype(np.float32)
        center = np.clip(np.asarray(self.center, dtype=np.float32), box.low,
                         box.high)
        width = np.asarray(self.width, dtype=np.float32)
        # uniform(x, x) returns x, so zero widths pin to the center.
        return rng.uniform(np.maximum(center - width, box.low),
                           np.minimum(center + width,
                                      box.high)).astype(np.float32)

    @property
    def deterministic(self) -> bool:
        """True when every draw is identical.

        An all-zero window pins to the center; a code fn may flag itself
        with a ``deterministic`` attribute, exactly like a parameterized
        sampler.
        """
        if self.fn is not None:
            return bool(getattr(self.fn, "deterministic", False))
        return bool(np.all(np.asarray(self.width) == 0))


@dataclasses.dataclass
class SketchStep:
    """One step in an agent-produced plan sketch.

    ``subgoal_atoms`` / ``subgoal_neg_atoms`` are optional: ``None``
    means "no subgoal constraint at this step"; an empty set means "the
    annotation was present but contained no atoms of that polarity".
    """
    option: ParameterizedOption
    objects: Sequence[Object]
    subgoal_atoms: Optional[Set[GroundAtom]]
    subgoal_neg_atoms: Optional[Set[GroundAtom]] = None
    # Optional LLM-proposed continuous parameters for this step. ``None``
    # means "no proposal" (the search samples from the start); otherwise the
    # refinement tries these first (clipped to the option's box) on the first
    # arrival at this step, then falls back to the sampler/uniform draw.
    initial_params: Optional[np.ndarray] = None
    # Optional ground sampler for this step, compiled from a ``~ [w1, w2]``
    # region annotation (requires ``initial_params``, its window center).
    # When set, the exact center is still tried once, and every later draw
    # for this step comes from the ground sampler instead of the full box,
    # taking precedence over any learned parameterized sampler.
    ground_sampler: Optional[GroundSampler] = None
