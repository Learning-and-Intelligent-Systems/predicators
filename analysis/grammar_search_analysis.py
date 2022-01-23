"""Debugging script for grammar search invention approach."""

import time
from collections import defaultdict
from operator import le
import glob
import os
from typing import Dict, DefaultDict, Set, List, Tuple
import pandas as pd
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_env, BaseEnv
from predicators.src.approaches import create_approach
from predicators.src.approaches.grammar_search_invention_approach import \
    _create_score_function, _ForallClassifier, _SingleAttributeCompareClassifier
from predicators.src.ground_truth_nsrts import _get_predicates_by_names
from predicators.src.main import _run_testing
from predicators.src import utils
from predicators.src.structs import Predicate, Dataset
from predicators.src.settings import CFG


def _run_proxy_analysis(env_names: List[str], score_function_names: List[str],
                        run_planning: bool, outdir: str) -> None:
    utils.update_config({"seed": 0})
    if "cover" in env_names:
        env_name = "cover"
        HandEmpty, Holding = _get_predicates_by_names(env_name,
                                                      ["HandEmpty", "Holding"])
        NotHandEmpty = HandEmpty.get_negation()
        covers_pred_sets: List[Set[Predicate]] = [
            set(),
            {HandEmpty},
            {Holding},
            {HandEmpty, Holding},
            {NotHandEmpty},
        ]
        _run_proxy_analysis_for_env(env_name, covers_pred_sets,
                                    score_function_names, run_planning, outdir)

    if "blocks" in env_names:
        env_name = "blocks"
        Holding, Clear, GripperOpen = _get_predicates_by_names(
            env_name, ["Holding", "Clear", "GripperOpen"])

        # NOT-Forall[0:block].[((0:block).pose_x<=1.33)(0)]
        block_type = Clear.types[0]
        pose_x_classifier = _SingleAttributeCompareClassifier(
            0, block_type, "pose_x", 1.33, le, "<=")
        pose_x_pred = Predicate(str(pose_x_classifier), [block_type],
                                pose_x_classifier)
        forall_pose_x_classifier = _ForallClassifier(pose_x_pred)
        forall_pose_x_pred = Predicate(str(forall_pose_x_classifier), [],
                                       forall_pose_x_classifier)
        not_forall_pose_x_pred = forall_pose_x_pred.get_negation()
        assert str(not_forall_pose_x_pred) == \
            "NOT-Forall[0:block].[((0:block).pose_x<=1.33)(0)]"

        # NOT-((0:block).pose_x<=1.35)
        pose_x35_classifier = _SingleAttributeCompareClassifier(
            0, block_type, "pose_x", 1.35, le, "<=")
        pose_x35_pred = Predicate(str(pose_x35_classifier), [block_type],
                                  pose_x35_classifier)
        not_pose_x35_pred = pose_x35_pred.get_negation()
        assert str(not_pose_x35_pred) == "NOT-((0:block).pose_x<=1.35)"

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
        _run_proxy_analysis_for_env(env_name, blocks_pred_sets,
                                    score_function_names, run_planning, outdir)

    if "painting" in env_names:
        env_name = "painting"
        (GripperOpen, OnTable, HoldingTop, HoldingSide, Holding, IsWet, IsDry,
         IsDirty, IsClean) = _get_predicates_by_names("painting", [
             "GripperOpen", "OnTable", "HoldingTop", "HoldingSide", "Holding",
             "IsWet", "IsDry", "IsDirty", "IsClean"
         ])
        all_predicates = {
            GripperOpen, OnTable, HoldingTop, HoldingSide, Holding, IsWet,
            IsDry, IsDirty, IsClean
        }

        # ((0:obj).color<=0.125)
        obj_type = Holding.types[0]
        color_classifier = _SingleAttributeCompareClassifier(
            0, obj_type, "color", 0.125, le, "<=")
        color_pred = Predicate(str(color_classifier), [obj_type],
                               color_classifier)
        assert str(color_pred) == "((0:obj).color<=0.125)"

        NotGripperOpen = GripperOpen.get_negation()

        # ((0:obj).dirtiness<=0.619)
        custom_dirtiness_classifier = _SingleAttributeCompareClassifier(
            0, obj_type, "dirtiness", 0.619, le, "<=")
        custom_dirtiness_pred = Predicate(str(custom_dirtiness_classifier),
            [obj_type], custom_dirtiness_classifier)
        assert str(custom_dirtiness_pred) == "((0:obj).dirtiness<=0.619)"

        painting_pred_sets: List[Set[Predicate]] = [
            # set(),
            # all_predicates - {IsWet, IsDry},
            # all_predicates - {IsClean, IsDirty},
            # all_predicates - {OnTable},
            # all_predicates - {HoldingTop, HoldingSide, Holding},
            all_predicates,
            # {IsClean, GripperOpen, Holding, OnTable},
            # {IsClean, GripperOpen, Holding, OnTable, NotGripperOpen},
            # {
            #     IsClean, GripperOpen, Holding, OnTable, NotGripperOpen,
            #     color_pred
            # },

            # {IsDry, IsWet, Holding, GripperOpen},

            #((0:obj).color<=0.125), ((0:obj).wetness<=0.5), NOT-((0:obj).wetness<=0.5), NOT-((0:obj).held<=0.5), NOT-((0:robot).fingers<=0.5)}
            # {color_pred, IsDry, IsWet, Holding, GripperOpen},

            # {color_pred, IsDry, IsWet, Holding, GripperOpen, OnTable},

            # ((0:obj).dirtiness<=0.619)
            # ((0:obj).color<=0.125)
            # ((0:obj).wetness<=0.5)
            # NOT-((0:obj).wetness<=0.5)
            # NOT-((0:obj).held<=0.5)
            # NOT-((0:robot).fingers<=0.5)
            # {custom_dirtiness_pred, color_pred, IsDry, IsWet, Holding, GripperOpen},

            # ((0:robot).gripper_rot<=0.5)
            # NOT-((0:obj).wetness<=0.5)
            # ((0:obj).pose_y<=-0.284)
            # NOT-((0:robot).fingers<=0.5)
            # ((0:obj).color<=0.124)
            # ((0:box).color<=0.782)
            # ((0:obj).wetness<=0.5)
            # NOT-((0:robot).gripper_rot<=0.25)
            # NOT-((0:obj).color<=0.124)
            # ((0:obj).dirtiness<=0.495)
            # NOT-((0:obj).held<=0.5)
            # 6124159

            # all_predicates | {color_pred},
        ]
        _run_proxy_analysis_for_env(env_name, painting_pred_sets,
                                    score_function_names, run_planning, outdir)


def _run_proxy_analysis_for_env(env_name: str,
                                non_goal_predicate_sets: List[Set[Predicate]],
                                score_function_names: List[str],
                                run_planning: bool, outdir: str) -> None:
    utils.update_config({
        "env": env_name,
        "seed": 0,
        "experiment_id": "proxy_analysis",
    })
    utils.update_config({
        "env": env_name,
        "offline_data_method": "demo+replay",
        "seed": 0,
        "timeout": 1,
        "make_videos": False,
        "grammar_search_max_predicates": 50,
        "grammar_search_operator_size_weight": 0.,
        "grammar_search_pred_complexity_weight": 0.,
        "excluded_predicates": "",
    })
    env = create_env(env_name)
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)
    start_time = time.time()

    for non_goal_predicates in non_goal_predicate_sets:
        results_for_predicates = \
            _run_proxy_analysis_for_predicates(env, dataset,
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
          f"{time.time()-start_time:.3f} seconds")


def _run_proxy_analysis_for_predicates(
    env: BaseEnv,
    dataset: Dataset,
    initial_predicates: Set[Predicate],
    predicates: Set[Predicate],
    score_function_names: List[str],
    run_planning: bool,
) -> Dict[str, float]:
    utils.flush_cache()
    candidates = {p: 1.0 for p in predicates}
    all_predicates = predicates | initial_predicates
    atom_dataset = utils.create_ground_atom_dataset(dataset, all_predicates)
    results = {}
    # Compute scores.
    for score_function_name in score_function_names:
        score_function = _create_score_function(score_function_name,
                                                initial_predicates,
                                                atom_dataset, candidates)
        start_time = time.time()
        score = score_function.evaluate(frozenset(predicates))
        eval_time = time.time() - start_time
        results[score_function_name + " Score"] = score
        results[score_function_name + " Time"] = eval_time
    # Learn NSRTs and plan.
    if run_planning:
        utils.flush_cache()
        approach = create_approach("nsrt_learning", env.simulate,
                                   all_predicates, env.options, env.types,
                                   env.action_space)
        approach.learn_from_offline_dataset(dataset)
        approach.seed(CFG.seed)
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
    env_names = [
        # "cover",
        # "blocks",
        "painting",
    ]
    score_function_names = [
        # "prediction_error",
        # "hadd_energy_lookaheaddepth0",
        # "hadd_energy_lookaheaddepth1",
        # "hadd_energy_lookaheaddepth2",
        # "exact_energy",
        # "lmcut_count_lookaheaddepth0",
        # "hadd_count_lookaheaddepth0",
        # "exact_count",
        "refinement_prob",
    ]
    run_planning = False

    outdir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                          "results")
    os.makedirs(outdir, exist_ok=True)

    _run_proxy_analysis(env_names, score_function_names, run_planning, outdir)
    _make_proxy_analysis_results(outdir)


if __name__ == "__main__":
    _main()
