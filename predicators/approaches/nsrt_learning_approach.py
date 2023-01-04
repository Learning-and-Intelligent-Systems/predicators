"""A bilevel planning approach that learns NSRTs.

Learns operators and samplers. Does not attempt to learn new predicates
or options.
"""

import logging
import re
import time
from typing import Dict, List, Optional, Set

import dill as pkl
from gym.spaces import Box

from predicators import utils
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.envs import get_or_create_env
from predicators.nsrt_learning.nsrt_learning_main import \
    get_ground_atoms_dataset, learn_nsrts_from_data
from predicators.planning import task_plan, task_plan_grounding
from predicators.settings import CFG
from predicators.structs import NSRT, Dataset, LiftedAtom, \
    LowLevelTrajectory, NSRTSampler, ParameterizedOption, Predicate, Segment, \
    Task, Type, Variable


class NSRTLearningApproach(BilevelPlanningApproach):
    """A bilevel planning approach that learns NSRTs."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        self._nsrts: Set[NSRT] = set()
        self._segmented_trajs: List[List[Segment]] = []
        self._seg_to_nsrt: Dict[Segment, NSRT] = {}

    @classmethod
    def get_name(cls) -> str:
        return "nsrt_learning"

    @property
    def is_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

    def _get_from_env_by_names_dict(self, env_name: str,
                                    env_attr: str) -> Dict:  # pragma: no cover
        """Helper for getting dict to load types, predicates, and options by
        name."""
        env = get_or_create_env(env_name)
        name_to_env_obj = {}
        for o in getattr(env, env_attr):
            name_to_env_obj[o.name] = o
        return name_to_env_obj

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # The only thing we need to do here is learn NSRTs,
        # which we split off into a different function in case
        # subclasses want to make use of it.
        self._learn_nsrts(dataset.trajectories, online_learning_cycle=None)

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     online_learning_cycle: Optional[int]) -> None:
        ground_atom_dataset = get_ground_atoms_dataset(
            trajectories, self._get_current_predicates(),
            online_learning_cycle, self._train_tasks)

        self._nsrts, self._segmented_trajs, self._seg_to_nsrt = \
            learn_nsrts_from_data(trajectories,
                                  self._train_tasks,
                                  self._get_current_predicates(),
                                  self._initial_options,
                                  self._action_space,
                                  ground_atom_dataset,
                                  sampler_learner=CFG.sampler_learner)
        save_path = utils.get_approach_save_path_str()
        # Need to save samplers if we are dumping to string
        if CFG.dump_nsrts_as_strings:
            with open(f"{save_path}_{online_learning_cycle}.SAMPLERs",
                      "wb") as f:
                sampler_name_to_sampler = {}
                for nsrt in self._nsrts:
                    sampler_name_to_sampler[nsrt.name] = nsrt.sampler
                pkl.dump(sampler_name_to_sampler, f)

        with open(f"{save_path}_{online_learning_cycle}.NSRTs", "wb") as f:
            if CFG.dump_nsrts_as_strings:
                pkl.dump(str(self._nsrts), f)
            else:
                pkl.dump(self._nsrts, f)
        if CFG.compute_sidelining_objective_value:
            self._compute_sidelining_objective_value(trajectories)

    def load(self, online_learning_cycle: Optional[int]) -> None:
        save_path = utils.get_approach_load_path_str()
        with open(f"{save_path}_{online_learning_cycle}.NSRTs", "rb") as f:
            if CFG.dump_nsrts_as_strings:  # pragma: no cover
                string_nsrts = pkl.load(f)
                assert isinstance(string_nsrts, str)
                with open(f"{save_path}_{online_learning_cycle}.SAMPLERs",
                          "rb") as f:
                    sampler_name_to_sampler = pkl.load(f)
                # Implement string_nsrts to _nsrts
                self._nsrts = self.parse_nsrts_string(string_nsrts,
                                                      sampler_name_to_sampler)
            else:
                self._nsrts = pkl.load(f)
        if CFG.pretty_print_when_loading:
            preds, _ = utils.extract_preds_and_types(self._nsrts)
            name_map = {}
            logging.info("Invented predicates:")
            for idx, pred in enumerate(
                    sorted(set(preds.values()) - self._initial_predicates)):
                vars_str, body_str = pred.pretty_str()
                logging.info(f"\tP{idx+1}({vars_str}) ≜ {body_str}")
                name_map[body_str] = f"P{idx+1}"
        logging.info("\n\nLoaded NSRTs:")
        for nsrt in sorted(self._nsrts):
            if CFG.pretty_print_when_loading:
                logging.info(nsrt.pretty_str(name_map))
            else:
                logging.info(nsrt)
        logging.info("")
        # Seed the option parameter spaces after loading.
        for nsrt in self._nsrts:
            nsrt.option.params_space.seed(CFG.seed)

    def _compute_sidelining_objective_value(
            self, trajectories: List[LowLevelTrajectory]) -> None:
        """Compute the value of the objective function that sidelining is
        trying to approximately optimize.

        Store this into self._metrics.
        """
        # Assert that we have an admissible heuristic, since we will
        # assume in this function that task_plan() generates skeletons
        # of increasing length.
        assert CFG.sesame_task_planning_heuristic == "lmcut"
        logging.info("Computing sidelining objective value...")
        start_time = time.perf_counter()
        preds = self._get_current_predicates()
        strips_ops = [nsrt.op for nsrt in self._nsrts]
        option_specs = [(nsrt.option, list(nsrt.option_vars))
                        for nsrt in self._nsrts]
        # Calculate first term in objective. This is a sum over the training
        # tasks of the number of possible task plans up to the demo length.
        num_plans_up_to_n = 0
        for segment_traj, ll_traj in zip(self._segmented_trajs, trajectories):
            if not ll_traj.is_demo:
                continue
            task = self._train_tasks[ll_traj.train_task_idx]
            init_atoms = utils.abstract(task.init, preds)
            objects = set(task.init)
            ground_nsrts, reachable_atoms = task_plan_grounding(
                init_atoms,
                objects,
                strips_ops,
                option_specs,
                allow_noops=True)
            heuristic = utils.create_task_planning_heuristic(
                CFG.sesame_task_planning_heuristic, init_atoms, task.goal,
                ground_nsrts, preds, objects)
            for skeleton, _, _ in task_plan(init_atoms,
                                            task.goal,
                                            ground_nsrts,
                                            reachable_atoms,
                                            heuristic,
                                            CFG.seed,
                                            timeout=10000000,
                                            max_skeletons_optimized=10000000):
                # Here, we are assuming that task_plan() generates skeletons
                # of increasing length. If the demonstration length is
                # exceeded, we can break.
                if len(skeleton) > len(segment_traj):
                    break
                num_plans_up_to_n += 1
        # Calculate second term in objective. This is the complexity of the
        # operator set, measured as the sum of all operator complexities.
        complexity = 0.0
        for op in strips_ops:
            complexity += op.get_complexity()
        time_taken = time.perf_counter() - start_time
        self._metrics["sidelining_obj_num_plans_up_to_n"] = num_plans_up_to_n
        self._metrics["sidelining_obj_complexity"] = complexity
        self._metrics["sidelining_obj_time_taken"] = time_taken
        logging.info(f"\tFinished in {time_taken:.3f} seconds")
        logging.info(f"\tGot num_plans_up_to_n {num_plans_up_to_n} and "
                     f"complexity {complexity}")

    def parse_nsrts_string(
        self, nsrts_string: str, sampler_name_to_sampler: Dict[str,
                                                               NSRTSampler]
    ) -> Set[NSRT]:  # pragma: no cover
        """Parses BEHAVIOR NSRTs saved as strings by retrieving types,
        predicates, and options from the env and creating a set of NSRTS.

        This function returns a set of NSRTS.
        """
        assert CFG.env == "behavior"
        nsrts = set()
        type_name_to_type = self._get_from_env_by_names_dict(CFG.env, "types")
        pred_name_to_pred = self._get_from_env_by_names_dict(
            CFG.env, "predicates")
        option_name_to_option = self._get_from_env_by_names_dict(
            CFG.env, "options")

        for nsrt_string in nsrts_string.replace("{", "").replace(
                "}", "").split("NSRT-")[1:]:
            name, str_params, str_precond, str_add_effects, \
                str_delete_effects, str_ignore_effects, \
                option_spec = nsrt_string.split(
                "\n    ")
            name = name.replace(":", "")
            params = [
                param.split(":") for param in re.findall(
                    r"\[(.*?)\]", str_params)[0].split(", ")
            ]
            add_effects = re.findall(r"\[(.*?)\]",
                                     str_add_effects)[0][:-1].split("), ")
            precond = re.findall(r"\[(.*?)\]",
                                 str_precond)[0][:-1].split("), ")
            delete_effects = re.findall(
                r"\[(.*?)\]", str_delete_effects)[0][:-1].split("), ")
            ignore_effects = re.findall(r"\[(.*?)\]",
                                        str_ignore_effects)[0].split(", ")
            option_spec = option_spec.replace(", ", "").split(": ")[1]

            # Parameters
            nsrt_parameters = [
                Variable(param[0], type_name_to_type[param[1]])
                for param in params
            ]

            # Predicates (precond, add_effects, delete_effects)
            def unparsed_preds_to_predicates(
                    unparsed_preds: List[str]) -> Set[LiftedAtom]:
                predicates: Set[LiftedAtom] = set()
                if unparsed_preds == [""]:
                    return predicates
                for unparsed_pred in unparsed_preds:
                    unparsed_pred += ")"
                    pred_name = unparsed_pred.split("(")[0]
                    pred_vars = [
                        pred_var.split(":") for pred_var in re.findall(
                            r"\((.*?)\)", unparsed_pred)[0].split(", ")
                    ]
                    if pred_vars[0] == [""]:
                        args = []
                    else:
                        args = [
                            Variable(pred_var[0],
                                     type_name_to_type[pred_var[1]])
                            for pred_var in pred_vars
                        ]
                    predicates.add(
                        LiftedAtom(pred_name_to_pred[pred_name], args))
                return predicates

            nsrt_precond = unparsed_preds_to_predicates(precond)
            nsrt_add_effects = unparsed_preds_to_predicates(add_effects)
            nsrt_delete_effects = unparsed_preds_to_predicates(delete_effects)
            # Ignore Effects
            if ignore_effects == [""]:
                nsrt_ignore_effects = set()
            else:
                nsrt_ignore_effects = {
                    pred_name_to_pred[ignore_effect]
                    for ignore_effect in ignore_effects
                }
            # Option Spec
            nsrt_option = option_name_to_option[option_spec.split("(")[0]]
            option_vars = [
                option_var.split(":") for option_var in re.findall(
                    r"\((.*?)\)", option_spec)[0].split(", ")
            ]
            nsrt_option_vars = [
                Variable(option_var[0], type_name_to_type[option_var[1]])
                for option_var in option_vars
            ]
            # Sampler
            nsrt_sampler = sampler_name_to_sampler[name]

            nsrts.add(
                NSRT(name, nsrt_parameters, nsrt_precond, nsrt_add_effects,
                     nsrt_delete_effects, nsrt_ignore_effects, nsrt_option,
                     nsrt_option_vars, nsrt_sampler))
        return nsrts
