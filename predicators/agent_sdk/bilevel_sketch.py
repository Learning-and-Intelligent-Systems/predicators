"""Facade for the bilevel plan-sketch pipeline (now split by concern).

This module used to hold the whole pipeline (~2050 lines); it is split
at its natural seams and every public name is re-exported here so
``bilevel_sketch.X`` attribute-style uses and existing imports keep
working. New code should import from the specific module:

- ``sketch_types``: shared dataclasses (``GroundSampler``,
  ``SketchStep``) that parsing constructs and refinement/execution
  consume.
- ``sketch_prompts``: ``build_solve_prompt``, the solve/explore prompt
  builder.
- ``sketch_parsing``: the sketch-line grammar - step/plan formatters
  and the parsers for subgoal / ``~`` ground-sampler annotations and
  continuous params.
- ``sketch_refinement``: ``refine_sketch`` (backtracking search over
  continuous parameters) and ``refine_and_validate_report``.
- ``plan_execution``: forward execution of grounded plans
  (``execute_plan_forward``) and continuous re-validation
  (``validate_plan_forward``).
"""
from predicators.agent_sdk.plan_execution import ForwardResult, StepOutcome, \
    execute_plan_forward, validate_plan_forward
from predicators.agent_sdk.sketch_parsing import format_plan_lines, \
    format_sketch_lines, format_step_line, parse_atoms, \
    parse_region_annotations, parse_sketch_from_text, \
    parse_subgoal_annotations, strip_code_fences, strip_region_annotations, \
    strip_subgoal_annotations
from predicators.agent_sdk.sketch_prompts import build_solve_prompt
from predicators.agent_sdk.sketch_refinement import DeepestFailure, \
    InfoScorer, RefineOutcome, refine_and_validate_report, refine_sketch, \
    resolve_refine_timeout, sample_params
from predicators.agent_sdk.sketch_types import GroundSampler, SketchStep

__all__ = [
    "DeepestFailure",
    "ForwardResult",
    "GroundSampler",
    "InfoScorer",
    "RefineOutcome",
    "SketchStep",
    "StepOutcome",
    "build_solve_prompt",
    "execute_plan_forward",
    "format_plan_lines",
    "format_sketch_lines",
    "format_step_line",
    "parse_atoms",
    "parse_region_annotations",
    "parse_sketch_from_text",
    "parse_subgoal_annotations",
    "refine_and_validate_report",
    "refine_sketch",
    "resolve_refine_timeout",
    "sample_params",
    "strip_code_fences",
    "strip_region_annotations",
    "strip_subgoal_annotations",
    "validate_plan_forward",
]
