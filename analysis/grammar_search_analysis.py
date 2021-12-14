"""Debugging script for grammar search invention approach.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Tuple, Iterator, DefaultDict, Set, List
from predicators.src.datasets import create_dataset
from predicators.src.envs import create_env, BaseEnv
from predicators.src.approaches import create_approach
from predicators.src.approaches.grammar_search_invention_approach import \
    _create_grammar, _PredicateGrammar, _count_positives_for_ops, \
    _PredictionErrorHeuristic, _HAddLookaheadHeuristic, _TreeModelHeuristic
from predicators.src.approaches.oracle_approach import _get_predicates_by_names
from predicators.src.main import _run_testing
from predicators.src.nsrt_learning import segment_trajectory, \
    learn_strips_operators
from predicators.src import utils
from predicators.src.structs import Predicate, Dataset, Task
from predicators.src.settings import CFG


utils.update_config({
    "env": "painting",
    "excluded_predicates": "IsWet,IsDry",
    "grammar_search_grammar_name": "forall_single_feat_ineqs",
    "approach": "grammar_search_invention",
    "seed": 0,
    "grammar_search_max_predicates": 250,
})


# Replace these strings with anything you want to exclusively enumerate.
_DEBUG_PREDICATE_STRS = [
    # "((0:obj).wetness<=0.5)",
    # "NOT-((0:obj).wetness<=0.5)",
    "NOT-((0:block).held<=0.25)",
    "Forall[0:block].[((0:block).held<=0.5)(0)]",
    "Forall[0:block].[NOT-On(0,1)]",
    "((0:robot).fingers<=0.25)",
    "Forall[1:block].[NOT-On(0,1)]",
]


@dataclass(frozen=True, eq=False, repr=False)
class _DebugGrammar(_PredicateGrammar):
    """A grammar that generates only predicates in _DEBUG_PREDICATE_STRS.
    """
    base_grammar: _PredicateGrammar

    def generate(self, max_num: int) -> Dict[Predicate, float]:
        del max_num
        return super().generate(len(_DEBUG_PREDICATE_STRS))

    def enumerate(self) -> Iterator[Tuple[Predicate, float]]:
        for (predicate, cost) in self.base_grammar.enumerate():
            if str(predicate) in _DEBUG_PREDICATE_STRS:
                yield (predicate, cost)


def _run_analysis() -> None:
    env = create_env(CFG.env)

    for train_tasks in env.train_tasks_generator():
        dataset = create_dataset(env, train_tasks)
        break

    if CFG.excluded_predicates:
        excludeds = set(CFG.excluded_predicates.split(","))
        assert excludeds.issubset({pred.name for pred in env.predicates}), \
            "Unrecognized excluded_predicates!"
        initial_predicates = {pred for pred in env.predicates
                 if pred.name not in excludeds}
        assert env.goal_predicates.issubset(initial_predicates), \
            "Can't exclude a goal predicate!"
    else:
        initial_predicates = env.predicates

    grammar = _create_grammar(CFG.grammar_search_grammar_name,
                              dataset, initial_predicates)
    grammar = _DebugGrammar(grammar)
    candidates = grammar.generate(max_num=CFG.grammar_search_max_predicates)
    print(f"Done: created {len(candidates)} candidates:")
    for predicate in candidates:
        print(predicate)

    print("Applying predicates to data...")
    atom_dataset = utils.create_ground_atom_dataset(
        dataset, set(candidates) | initial_predicates)
    print("Done.")

    print("All candidates:", sorted(candidates))
    print("Running learning & scoring with ALL predicates.")
    all_segments = [seg for traj in atom_dataset
                    for seg in segment_trajectory(traj)]
    all_strips_ops, all_partitions = learn_strips_operators(all_segments,
                                                            verbose=False)
    all_option_specs = [p.option_spec for p in all_partitions]
    all_num_tps, all_num_fps, _, all_fp_idxs = \
        _count_positives_for_ops(all_strips_ops, all_option_specs,
                                 all_segments)
    print("TP/FP:", all_num_tps, all_num_fps)
    print("Running learning & scoring with INITIAL predicates.")
    pruned_atom_data = utils.prune_ground_atom_dataset(atom_dataset,
                                                       initial_predicates)
    init_segments = [seg for traj in pruned_atom_data
                     for seg in segment_trajectory(traj)]
    assert len(all_segments) == len(init_segments), \
        "This analysis assumes that segmentation does not change."
    init_strips_ops, init_partitions = learn_strips_operators(init_segments,
                                                              verbose=False)
    init_option_specs = [p.option_spec for p in init_partitions]
    # Score based on how well the operators fit the data.
    init_num_tps, init_num_fps, _, init_fp_idxs = \
        _count_positives_for_ops(init_strips_ops, init_option_specs,
                                 init_segments)
    print("TP/FP:", init_num_tps, init_num_fps)
    # Generally we would expect false positives to go down with the extra
    # predicates. But we're debugging, and there may be some false positives
    # that appear with the learned predicates that did not appear initially.
    all_combined_fps = {idx for fp_idxs in all_fp_idxs for idx in fp_idxs}
    init_combined_fps = {idx for fp_idxs in init_fp_idxs for idx in fp_idxs}

    # Break down FPs for ALL
    print("########### ALL ###########")
    all_total_per_op : DefaultDict[str, int] = defaultdict(int)
    for idx in sorted(all_combined_fps):
        all_segment = all_segments[idx]
        init_segment = init_segments[idx]
        assert all_segment.get_option() == init_segment.get_option()
        for i, op in enumerate(all_strips_ops):
            if idx in all_fp_idxs[i]:
                all_total_per_op[op.name] += 1

    for op, spec in zip(all_strips_ops, all_option_specs):
        print(op)
        print("    Option Spec:", spec[0].name, spec[1])
        print(f"Total FPs for {op.name}: {all_total_per_op[op.name]}")

    # Break down FPs for INIT
    print("########### INIT ###########")
    init_total_per_op : DefaultDict[str, int] = defaultdict(int)
    for idx in sorted(init_combined_fps):
        all_segment = all_segments[idx]
        init_segment = init_segments[idx]
        assert all_segment.get_option() == init_segment.get_option()
        for i, op in enumerate(init_strips_ops):
            if idx in init_fp_idxs[i]:
                init_total_per_op[op.name] += 1

    for op, spec in zip(init_strips_ops, init_option_specs):
        print(op)
        print("    Option Spec:", spec[0].name, spec[1])
        print(f"Total FPs for {op.name}: {init_total_per_op[op.name]}")


def _run_proxy_analysis() -> None:
    # env_name = "cover"
    # HandEmpty, Holding, IsBlock, IsTarget = _get_predicates_by_names(
    #     env_name, ["HandEmpty", "Holding", "IsBlock", "IsTarget"])
    # non_goal_predicate_sets: List[Set[Predicate]] = [
    #     set(),
    #     {HandEmpty},
    #     {Holding},
    #     {HandEmpty, IsBlock},
    #     {Holding, IsBlock},
    #     {HandEmpty, Holding},
    #     {HandEmpty, Holding, IsBlock},
    # ]

    env_name = "blocks"
    Holding, Clear, GripperOpen = _get_predicates_by_names(
        env_name, ["Holding", "Clear", "GripperOpen"])
    non_goal_predicate_sets: List[Set[Predicate]] = [
        # set(),
        # {Holding},
        # {Clear},
        # {GripperOpen},
        {Holding, Clear},
        # {Clear, GripperOpen},
        # {GripperOpen, Holding},
        {Clear, GripperOpen, Holding},
    ]

    # env_name = "painting"
    # (GripperOpen, OnTable, HoldingTop, HoldingSide, Holding, IsWet, IsDry,
    #  IsDirty, IsClean) = _get_predicates_by_names("painting",
    #         ["GripperOpen", "OnTable", "HoldingTop", "HoldingSide",
    #          "Holding", "IsWet", "IsDry", "IsDirty", "IsClean"])
    # all_predicates = {GripperOpen, OnTable, HoldingTop, HoldingSide, Holding,
    #                   IsWet, IsDry, IsDirty, IsClean}
    # non_goal_predicate_sets: List[Set[Predicate]] = [
    #     # set(),
    #     # all_predicates - {IsWet, IsDry},
    #     # all_predicates - {IsClean, IsDirty},
    #     # all_predicates - {OnTable},
    #     # all_predicates - {HoldingTop, HoldingSide, Holding},
    #     all_predicates,
    # ]

    utils.update_config({
        "env": env_name,
        "offline_data_method": "demo+replay",
        "seed": 0,
        "timeout": 100,
        "make_videos": False,
        "grammar_search_max_predicates": 50,
    })
    env = create_env(env_name)
    train_tasks = next(env.train_tasks_generator())
    dataset = create_dataset(env, train_tasks)

    for non_goal_predicates in non_goal_predicate_sets:
        print(env_name, non_goal_predicates)
        _run_proxy_analysis_for_predicates(env, dataset, train_tasks,
                                           env.goal_predicates,
                                           non_goal_predicates)
    # # Also test full predicate set proposed by grammar
    # grammar = _create_grammar("forall_single_feat_ineqs",
    #                           dataset, env.goal_predicates)
    # candidates = grammar.generate(max_num=CFG.grammar_search_max_predicates)
    # _run_proxy_analysis_for_predicates(env, dataset, train_tasks,
    #                                    env.goal_predicates, set(candidates))
    # Test a specific subset of the candidates
    # grammar = _create_grammar("forall_single_feat_ineqs",
    #                           dataset, env.goal_predicates)
    # grammar = _DebugGrammar(grammar)
    # candidates = grammar.generate(max_num=CFG.grammar_search_max_predicates)
    # _run_proxy_analysis_for_predicates(env, dataset, train_tasks,
    #                                    env.goal_predicates, set(candidates))


def _run_proxy_analysis_for_predicates(env: BaseEnv,
                                       dataset: Dataset,
                                       train_tasks: List[Task],
                                       initial_predicates: Set[Predicate],
                                       predicates: Set[Predicate],
                                       ) -> None:
    utils.flush_cache()
    candidates = {p : 1.0 for p in predicates}
    all_predicates = predicates | initial_predicates
    atom_dataset = utils.create_ground_atom_dataset(dataset, all_predicates)
    # Compute heuristic scores.
    for heuristic_cls in [
        # _PredictionErrorHeuristic,
        # _HAddLookaheadHeuristic,
        _TreeModelHeuristic,
        ]:
        heuristic = heuristic_cls(initial_predicates, atom_dataset,
                                  candidates)
        heuristic_score = heuristic.evaluate(frozenset(predicates))
        if len(predicates) >= 10:
            predicate_str = f"All {len(predicates)} candidates"
        else:
            predicate_str = str(predicates)
        print("\n\n******", env.__class__.__name__, predicate_str,
              heuristic_cls.__name__, heuristic_score)
    # Learn NSRTs and plan.
    # utils.flush_cache()
    # approach = create_approach("nsrt_learning", env.simulate, all_predicates,
    #                            env.options, env.types, env.action_space)
    # approach.learn_from_offline_dataset(dataset, train_tasks)
    # approach.seed(CFG.seed)
    # _run_testing(env, approach)


if __name__ == "__main__":
    # _run_analysis()
    _run_proxy_analysis()
