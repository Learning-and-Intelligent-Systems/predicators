"""Debugging script for grammar search invention approach.
"""

from collections import defaultdict
import glob
import os
from typing import Dict, DefaultDict, Set, List, Tuple
import pandas as pd
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_env, BaseEnv
from predicators.src.approaches import create_approach
from predicators.src.approaches.grammar_search_invention_approach import \
    _PredictionErrorHeuristic, _HAddLookaheadHeuristic, \
    _ExactLookaheadHeuristic
from predicators.src.approaches.oracle_approach import _get_predicates_by_names
from predicators.src.main import _run_testing
from predicators.src import utils
from predicators.src.structs import Predicate, Dataset, Task
from predicators.src.settings import CFG


def _run_proxy_analysis(env_names: List[str],
                        heuristic_names: List[str],
                        run_planning: bool,
                        outdir: str) -> None:
    if "cover" in env_names:
        env_name = "cover"
        HandEmpty, Holding = _get_predicates_by_names(
            env_name, ["HandEmpty", "Holding"])
        covers_pred_sets: List[Set[Predicate]] = [
            set(),
            {HandEmpty},
            {Holding},
            {HandEmpty, Holding},
        ]
        _run_proxy_analysis_for_env(env_name, covers_pred_sets,
                                    heuristic_names, run_planning, outdir)

    if "blocks" in env_names:
        env_name = "blocks"
        Holding, Clear, GripperOpen = _get_predicates_by_names(
            env_name, ["Holding", "Clear", "GripperOpen"])
        blocks_pred_sets: List[Set[Predicate]] = [
            set(),
            {Holding},
            {Clear},
            {GripperOpen},
            {Holding, Clear},
            {Clear, GripperOpen},
            {GripperOpen, Holding},
            {Clear, GripperOpen, Holding},
        ]
        _run_proxy_analysis_for_env(env_name, blocks_pred_sets,
                                    heuristic_names, run_planning, outdir)


    if "painting" in env_names:
        env_name = "painting"
        (GripperOpen, OnTable, HoldingTop, HoldingSide, Holding, IsWet, IsDry,
         IsDirty, IsClean) = _get_predicates_by_names("painting",
                ["GripperOpen", "OnTable", "HoldingTop", "HoldingSide",
                 "Holding", "IsWet", "IsDry", "IsDirty", "IsClean"])
        all_predicates = {GripperOpen, OnTable, HoldingTop, HoldingSide,
                          Holding, IsWet, IsDry, IsDirty, IsClean}
        painting_pred_sets: List[Set[Predicate]] = [
            set(),
            all_predicates - {IsWet, IsDry},
            all_predicates - {IsClean, IsDirty},
            all_predicates - {OnTable},
            all_predicates - {HoldingTop, HoldingSide, Holding},
            all_predicates,
        ]
        _run_proxy_analysis_for_env(env_name, painting_pred_sets,
                                    heuristic_names, run_planning, outdir)


def _run_proxy_analysis_for_env(env_name: str,
                                non_goal_predicate_sets: List[Set[Predicate]],
                                heuristic_names: List[str],
                                run_planning: bool,
                                outdir: str) -> None:
    utils.update_config({
        "env": env_name,
        "seed": 0,
    })
    utils.update_config({
        "env": env_name,
        "offline_data_method": "demo+replay",
        "seed": 0,
        "timeout": 1,
        "make_videos": False,
        "grammar_search_max_predicates": 50,
        "excluded_predicates": "",
    })
    env = create_env(env_name)
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)

    for non_goal_predicates in non_goal_predicate_sets:
        results_for_predicates = \
            _run_proxy_analysis_for_predicates(env, dataset, train_tasks,
                                               env.goal_predicates,
                                               non_goal_predicates,
                                               heuristic_names,
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


def _run_proxy_analysis_for_predicates(env: BaseEnv,
                                       dataset: Dataset,
                                       train_tasks: List[Task],
                                       initial_predicates: Set[Predicate],
                                       predicates: Set[Predicate],
                                       heuristic_names: List[str],
                                       run_planning: bool,
                                       ) -> Dict[str, float]:
    heuristics = {
        "prediction_error": _PredictionErrorHeuristic,
        "hadd_lookahead": _HAddLookaheadHeuristic,
        "exact_lookahead": _ExactLookaheadHeuristic,
    }
    utils.flush_cache()
    candidates = {p : 1.0 for p in predicates}
    all_predicates = predicates | initial_predicates
    atom_dataset = utils.create_ground_atom_dataset(dataset, all_predicates)
    results = {}
    # Compute heuristic scores.
    for heuristic_name in heuristic_names:
        heuristic_cls = heuristics[heuristic_name]
        heuristic = heuristic_cls(initial_predicates, atom_dataset,
                                  train_tasks, candidates)
        heuristic_score = heuristic.evaluate(frozenset(predicates))
        results[heuristic_name] = heuristic_score
    # Learn NSRTs and plan.
    if run_planning:
        utils.flush_cache()
        approach = create_approach("nsrt_learning", env.simulate,
            all_predicates, env.options, env.types, env.action_space)
        approach.learn_from_offline_dataset(dataset, train_tasks)
        approach.seed(CFG.seed)
        planning_result = _run_testing(env, approach)
        results.update(planning_result["test"])
        results.update(planning_result["approach"])
    return results


def _make_proxy_analysis_results(outdir: str) -> None:
    all_results: DefaultDict[Tuple[str, str], Dict] = defaultdict(dict)
    for filepath in glob.glob(f"{outdir}/*.result"):
        with open(filepath, "r", encoding="utf-8") as f:
            raw_result = f.read()
        result = float(raw_result)
        _, filename = os.path.split(filepath[:-len(".result")])
        env_name, preds, metric = filename.split("__")
        all_results[(env_name, preds)][metric] = result
    all_results_table = [{"Env": env_name, "Preds": preds, **metrics}
                         for (env_name, preds), metrics in all_results.items()]
    df = pd.DataFrame(all_results_table)
    print(df)
    csv_filepath = os.path.join(outdir, "proxy_analysis.csv")
    df.to_csv(csv_filepath)
    print(f"Wrote out to {csv_filepath}.")


def _main() -> None:
    env_names = [
        "cover",
        "blocks",
        "painting",
    ]
    heuristic_names = [
        "prediction_error",
        "hadd_lookahead",
        "exact_lookahead",
    ]
    run_planning = True

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                          "results")
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    _run_proxy_analysis(env_names, heuristic_names, run_planning, outdir)
    _make_proxy_analysis_results(outdir)


if __name__ == "__main__":
    _main()
