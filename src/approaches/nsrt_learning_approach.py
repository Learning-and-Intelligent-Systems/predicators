"""A bilevel planning approach that learns NSRTs.

Learns operators and samplers. Does not attempt to learn new predicates
or options.
"""

import logging
import os
import time
from typing import Dict, List, Optional, Set

import dill as pkl
from gym.spaces import Box

from predicators.src import utils
from predicators.src.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.src.nsrt_learning.nsrt_learning_main import \
    learn_nsrts_from_data
from predicators.src.planning import task_plan, task_plan_grounding
from predicators.src.settings import CFG
from predicators.src.structs import NSRT, Dataset, LowLevelTrajectory, \
    ParameterizedOption, Predicate, Segment, Task, Type, GroundAtom


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

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # The only thing we need to do here is learn NSRTs,
        # which we split off into a different function in case
        # subclasses want to make use of it.
        self._learn_nsrts(dataset.trajectories, online_learning_cycle=None)

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     online_learning_cycle: Optional[int]) -> None:
        dataset_fname, _ = utils.create_dataset_filename_str(
            saving_ground_atoms=True,
            online_learning_cycle=online_learning_cycle)
        # If CFG.load_atoms is set, then try to load the GroundAtomTrajectory
        # directly from a saved file.
        if CFG.load_atoms:
            os.makedirs(CFG.data_dir, exist_ok=True)
            # Check that the dataset file was previously saved.
            if os.path.exists(dataset_fname):
                # Load the ground atoms dataset.
                with open(dataset_fname, "rb") as f:
                    ground_atom_dataset_trajectories = pkl.load(f)
                logging.info("\n\nLOADED GROUND ATOM DATASET")
                ground_atom_dataset = []
                for i, traj in enumerate(trajectories):
                    ground_atom_seq = ground_atom_dataset_trajectories[i]
                    ground_atom_dataset.append(
                        (traj, [set(atoms) for atoms in ground_atom_seq]))
            else:
                raise ValueError(f"Cannot load ground atoms: {dataset_fname}")
        else:
            # Apply predicates to data, producing a dataset of abstract states.
            ground_atom_dataset = utils.create_ground_atom_dataset(
                trajectories, self._get_current_predicates())
            # Save ground atoms dataset to file.
            if CFG.env == "behavior":  # pragma: no cover
                # In the case of behavior, we cannot directly pickle the ground
                # atoms dataset because the classifiers are linked to the
                # simulator, which cannot be pickled. Thus, we must strip away
                # the classifiers.
                ground_atom_dataset_to_pkl = []
                for gt_traj in ground_atom_dataset:
                    trajectory = []
                    for i, ground_atom_seq in enumerate(gt_traj[1]):
                        trajectory.append({
                            GroundAtom(
                                Predicate(atom.predicate.name,
                                          atom.predicate.types,
                                          lambda s, o: False), atom.entities)
                            for atom in ground_atom_seq
                        })
                    ground_atom_dataset_to_pkl.append(trajectory)
            else:
                ground_atom_dataset_to_pkl = ground_atom_dataset
            with open(dataset_fname, "wb") as f:
                pkl.dump(ground_atom_dataset_to_pkl, f)

        self._nsrts, self._segmented_trajs, self._seg_to_nsrt = \
            learn_nsrts_from_data(trajectories,
                                  self._train_tasks,
                                  self._get_current_predicates(),
                                  self._initial_options,
                                  self._action_space,
                                  ground_atom_dataset,
                                  sampler_learner=CFG.sampler_learner)
        save_path = utils.get_approach_save_path_str()
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
        start_time = time.time()
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
        time_taken = time.time() - start_time
        self._metrics["sidelining_obj_num_plans_up_to_n"] = num_plans_up_to_n
        self._metrics["sidelining_obj_complexity"] = complexity
        self._metrics["sidelining_obj_time_taken"] = time_taken
        logging.info(f"\tFinished in {time_taken:.3f} seconds")
        logging.info(f"\tGot num_plans_up_to_n {num_plans_up_to_n} and "
                     f"complexity {complexity}")
