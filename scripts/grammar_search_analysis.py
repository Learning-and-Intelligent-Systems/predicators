"""Debugging script for grammar search invention approach."""

import glob
import os
import time
from collections import defaultdict
from operator import le
from typing import Any, DefaultDict, Dict, List, Sequence, Set, Tuple

import pandas as pd

from predicators import utils
from predicators.approaches import create_approach
from predicators.approaches.grammar_search_invention_approach import \
    _ForallClassifier, _SingleAttributeCompareClassifier
from predicators.datasets import create_dataset
from predicators.envs import BaseEnv, create_new_env
from predicators.envs.cover import CoverEnv
from predicators.ground_truth_nsrts import _get_predicates_by_names
from predicators.main import _run_testing
from predicators.predicate_search_score_functions import create_score_function
from predicators.structs import Dataset, Object, Predicate, State, Task

DEFAULT_ENV_NAMES = [
    "cover",
    "blocks",
    "painting",
]

DEFAULT_SCORE_FUNCTION_NAMES = [
    "expected_nodes_created",
]

DEFAULT_SEED = 0

RUN_PLANNING = False


def _run_proxy_analysis(args: Dict[str, Any], env_names: List[str],
                        score_function_names: List[str], run_planning: bool,
                        outdir: str) -> None:
    # Seed needs to be set to instantiate environments.
    utils.reset_config({"seed": args["seed"]})
    if "cover" in env_names:
        env_name = "cover"
        HandEmpty, Holding, Covers = _get_predicates_by_names(
            env_name, ["HandEmpty", "Holding", "Covers"])
        targ_type = Covers.types[1]
        NotHandEmpty = HandEmpty.get_negation()

        def _Clear_holds(state: State, objects: Sequence[Object]) -> bool:
            target, = objects
            for block in state:
                if block.type.name != "block":
                    continue
                if CoverEnv._Covers_holds(state, [block, target]):  # pylint: disable=protected-access
                    return False
            return True

        Clear = Predicate("Clear", [targ_type], _Clear_holds)
        covers_pred_sets: List[Set[Predicate]] = [
            set(),
            {HandEmpty},
            {Holding},
            {HandEmpty, Holding},
            {HandEmpty, Holding, Clear},
            {NotHandEmpty},
            {NotHandEmpty, HandEmpty},
        ]
        _run_proxy_analysis_for_env(args, env_name, covers_pred_sets,
                                    score_function_names, run_planning, outdir)

    if "blocks" in env_names:
        env_name = "blocks"
        Holding, Clear, GripperOpen = _get_predicates_by_names(
            env_name, ["Holding", "Clear", "GripperOpen"])

        # NOT-Forall[0:block].[((0:block).pose_x<=1.33)(0)]
        block_type = Clear.types[0]
        pose_x_classifier = _SingleAttributeCompareClassifier(
            0, block_type, "pose_x", 1.33, 0, le, "<=")
        pose_x_pred = Predicate(str(pose_x_classifier), [block_type],
                                pose_x_classifier)
        forall_pose_x_classifier = _ForallClassifier(pose_x_pred)
        forall_pose_x_pred = Predicate(str(forall_pose_x_classifier), [],
                                       forall_pose_x_classifier)
        not_forall_pose_x_pred = forall_pose_x_pred.get_negation()
        assert str(not_forall_pose_x_pred) == \
            "NOT-Forall[0:block].[((0:block).pose_x<=[idx 0]1.33)(0)]"

        # NOT-((0:block).pose_x<=1.35)
        pose_x35_classifier = _SingleAttributeCompareClassifier(
            0, block_type, "pose_x", 1.35, 0, le, "<=")
        pose_x35_pred = Predicate(str(pose_x35_classifier), [block_type],
                                  pose_x35_classifier)
        not_pose_x35_pred = pose_x35_pred.get_negation()
        assert str(not_pose_x35_pred) == "NOT-((0:block).pose_x<=[idx 0]1.35)"

        blocks_pred_sets: List[Set[Predicate]] = [
            set(),
            {Holding},
            {Clear},
            {GripperOpen},
            {Holding, Clear},
            {Clear, GripperOpen},
            {GripperOpen, Holding},
            {Clear, GripperOpen, Holding},
            {Clear, GripperOpen, Holding, not_forall_pose_x_pred},
            {Clear, GripperOpen, Holding, not_pose_x35_pred},
        ]
        _run_proxy_analysis_for_env(args, env_name, blocks_pred_sets,
                                    score_function_names, run_planning, outdir)

    if "painting" in env_names:
        env_name = "painting"
        (GripperOpen, OnTable, HoldingTop, HoldingSide, Holding, IsWet, IsDry,
         IsDirty, IsClean) = _get_predicates_by_names("painting", [
             "GripperOpen", "OnTable", "HoldingTop", "HoldingSide", "Holding",
             "IsWet", "IsDry", "IsDirty", "IsClean"
         ])

        def all_lids_open_classifier(state: State,
                                     objects: Sequence[Object]) -> bool:
            del objects  # unused
            for o in state:
                if o.type.name == "lid" and state.get(o, "is_open") < 0.5:
                    return False
            return True

        AllLidsOpen = Predicate("AllLidsOpen", [], all_lids_open_classifier)

        all_predicates = {
            GripperOpen, OnTable, HoldingTop, HoldingSide, Holding, IsWet,
            IsDry, IsDirty, IsClean, AllLidsOpen
        }

        # ((0:obj).color<=0.125)
        obj_type = Holding.types[0]
        color_classifier = _SingleAttributeCompareClassifier(
            0, obj_type, "color", 0.125, 0, le, "<=")
        color_pred = Predicate(str(color_classifier), [obj_type],
                               color_classifier)
        assert str(color_pred) == "((0:obj).color<=[idx 0]0.125)"

        NotGripperOpen = GripperOpen.get_negation()

        painting_pred_sets: List[Set[Predicate]] = [
            set(),
            all_predicates - {IsWet, IsDry},
            all_predicates - {IsClean, IsDirty},
            all_predicates - {OnTable},
            all_predicates - {HoldingTop},
            all_predicates - {HoldingSide},
            all_predicates - {HoldingTop, HoldingSide},
            all_predicates - {HoldingTop, HoldingSide, Holding},
            all_predicates - {AllLidsOpen},
            all_predicates,
            {IsClean, GripperOpen, Holding, OnTable},
            {IsClean, GripperOpen, Holding, OnTable, NotGripperOpen},
            {
                IsClean, GripperOpen, Holding, OnTable, NotGripperOpen,
                color_pred
            },
        ]
        _run_proxy_analysis_for_env(args, env_name, painting_pred_sets,
                                    score_function_names, run_planning, outdir)


def _run_proxy_analysis_for_env(args: Dict[str, Any], env_name: str,
                                non_goal_predicate_sets: List[Set[Predicate]],
                                score_function_names: List[str],
                                run_planning: bool, outdir: str) -> None:
    utils.reset_config({
        "env": env_name,
        **args,
    })
    env = create_new_env(env_name)
    train_tasks = env.get_train_tasks()
    dataset = create_dataset(env, train_tasks, env.options)
    start_time = time.perf_counter()

    for non_goal_predicates in non_goal_predicate_sets:
        results_for_predicates = \
            _run_proxy_analysis_for_predicates(env, train_tasks, dataset,
                                               env.goal_predicates,
                                               non_goal_predicates,
                                               score_function_names,
                                               run_planning)
        # Save these results.
        pred_str = ",".join(sorted([str(p.name) for p in non_goal_predicates]))
        if not pred_str:
            pred_str = "[none]"
        filename_prefix = f"{env_name}__{pred_str}__"
        for k, v in results_for_predicates.items():
            filename = filename_prefix + k + ".result"
            filepath = os.path.join(outdir, filename)
            with open(filepath, "w", encoding="utf-8") as f:
                f.write(str(v))
    print(f"Finished proxy analysis for {env_name} in "
          f"{time.perf_counter()-start_time:.3f} seconds")


def _run_proxy_analysis_for_predicates(
    env: BaseEnv,
    train_tasks: List[Task],
    dataset: Dataset,
    initial_predicates: Set[Predicate],
    predicates: Set[Predicate],
    score_function_names: List[str],
    run_planning: bool,
) -> Dict[str, float]:
    utils.flush_cache()
    candidates = {p: 1.0 for p in predicates}
    all_predicates = predicates | initial_predicates
    atom_dataset = utils.create_ground_atom_dataset(dataset.trajectories,
                                                    all_predicates)
    results = {}
    # Compute scores.
    for score_function_name in score_function_names:
        score_function = create_score_function(score_function_name,
                                               initial_predicates,
                                               atom_dataset, candidates,
                                               train_tasks)
        start_time = time.perf_counter()
        score = score_function.evaluate(frozenset(predicates))
        eval_time = time.perf_counter() - start_time
        results[score_function_name + " Score"] = score
        results[score_function_name + " Time"] = eval_time
    # Learn NSRTs and plan.
    if run_planning:
        utils.flush_cache()
        approach = create_approach("nsrt_learning", all_predicates,
                                   env.options, env.types, env.action_space,
                                   train_tasks)
        approach.learn_from_offline_dataset(dataset)
        planning_result = _run_testing(env, approach)
        results.update(planning_result)
    return results


def _make_proxy_analysis_results(outdir: str) -> None:
    all_results: DefaultDict[Tuple[str, str], Dict] = defaultdict(dict)
    for filepath in sorted(glob.glob(f"{outdir}/*.result")):
        with open(filepath, "r", encoding="utf-8") as f:
            raw_result = f.read()
        result = float(raw_result)
        _, filename = os.path.split(filepath[:-len(".result")])
        env_name, preds, metric = filename.split("__")
        all_results[(env_name, preds)][metric] = result
    all_results_table = [{
        "Env": env_name,
        "Preds": preds,
        **metrics
    } for (env_name, preds), metrics in all_results.items()]
    df = pd.DataFrame(all_results_table)
    print(df)
    csv_filepath = os.path.join(outdir, "proxy_analysis.csv")
    df.to_csv(csv_filepath)
    print(f"Wrote out to {csv_filepath}.")


def _main() -> None:
    args = utils.parse_args(env_required=False,
                            approach_required=False,
                            seed_required=False)
    assert args["excluded_predicates"] == "", "This script ignores " + \
        "excluded predicates, so we disallow them."
    if args["env"] is not None:
        env_names = [args["env"]]
    else:
        del args["env"]
        env_names = DEFAULT_ENV_NAMES
    if args["seed"] is None:
        args["seed"] = DEFAULT_SEED
    if "grammar_search_score_function" in args:
        score_function_names = [args["grammar_search_score_function"]]
    else:
        score_function_names = DEFAULT_SCORE_FUNCTION_NAMES
    run_planning = RUN_PLANNING

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)

    _run_proxy_analysis(args, env_names, score_function_names, run_planning,
                        outdir)
    _make_proxy_analysis_results(outdir)


if __name__ == "__main__":
    _main()
