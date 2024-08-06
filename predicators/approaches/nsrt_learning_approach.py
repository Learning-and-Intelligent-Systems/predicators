"""A bilevel planning approach that learns NSRTs.

Learns operators and samplers. Does not attempt to learn new predicates
or options.
"""

import logging
import time
from typing import Any, Dict, List, Optional, Set

import dill as pkl
from gym.spaces import Box

from predicators import utils
from predicators.approaches.bilevel_planning_approach import \
    BilevelPlanningApproach
from predicators.nsrt_learning.nsrt_learning_main import learn_nsrts_from_data
from predicators.planning import task_plan, task_plan_grounding
from predicators.settings import CFG
from predicators.structs import NSRT, Dataset, GroundAtomTrajectory, \
    LowLevelTrajectory, ParameterizedOption, Predicate, Segment, Task, Type


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

    @property
    def is_offline_learning_based(self) -> bool:
        return True

    def _get_current_nsrts(self) -> Set[NSRT]:
        return self._nsrts

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # The only thing we need to do here is learn NSRTs,
        # which we split off into a different function in case
        # subclasses want to make use of it.
        self._learn_nsrts(dataset.trajectories,
                          online_learning_cycle=None,
                          annotations=(dataset.annotations
                                       if dataset.has_annotations else None))

    def _learn_nsrts(self, trajectories: List[LowLevelTrajectory],
                     online_learning_cycle: Optional[int],
                     annotations: Optional[List[Any]], **kwargs) -> None:
        dataset_fname, _ = utils.create_dataset_filename_str(
            saving_ground_atoms=True,
            online_learning_cycle=online_learning_cycle)
        # If CFG.load_atoms is set, then try to create a GroundAtomTrajectory
        # by loading sets of GroundAtoms directly from a saved file.
        # By default, we don't create a full ground atom dataset, since
        # doing so requires called abstract on all states, including states
        # that might ultimately just be in the middle of segments. When
        # options take many steps, this makes a big time/space difference.
        ground_atom_dataset: Optional[List[GroundAtomTrajectory]] = None
        if CFG.load_atoms:
            ground_atom_dataset = utils.load_ground_atom_dataset(
                dataset_fname, trajectories)
        elif CFG.save_atoms:
            # Apply predicates to data, producing a dataset of abstract states.
            ground_atom_dataset = utils.create_ground_atom_dataset(
                trajectories, self._get_current_predicates())
            utils.save_ground_atom_dataset(ground_atom_dataset, dataset_fname)
        elif CFG.offline_data_method in [
                "demo+labelled_atoms", "saved_vlm_img_demos_folder",
                "demo_with_vlm_imgs", "geo_and_demo+labelled_atoms",
                "geo_and_saved_vlm_img_demos_folder",
                "geo_and_demo_with_vlm_imgs"
        ]:
            # In this case, the annotations are basically ground atoms!
            # We can use these to make GroundAtomTrajectories.
            assert annotations is not None
            assert len(annotations) == len(trajectories)
            ground_atom_dataset = []
            annotations_with_only_selected_preds = []
            selected_preds = self._get_current_predicates()
            for atoms_traj in annotations:
                curr_selected_preds_atoms_traj = []
                for atoms_set in atoms_traj:
                    curr_selected_preds_atoms_set = set(
                        atom for atom in atoms_set
                        if atom.predicate in selected_preds)
                    curr_selected_preds_atoms_traj.append(
                        curr_selected_preds_atoms_set)
                annotations_with_only_selected_preds.append(
                    curr_selected_preds_atoms_traj)
            for ll_traj, atoms in zip(trajectories,
                                      annotations_with_only_selected_preds):
                ground_atom_dataset.append((ll_traj, atoms))
        self._nsrts, self._segmented_trajs, self._seg_to_nsrt = \
            learn_nsrts_from_data(trajectories,
                                  self._train_tasks,
                                  self._get_current_predicates(),
                                  self._initial_options,
                                  self._action_space,
                                  ground_atom_dataset,
                                  sampler_learner=CFG.sampler_learner,
                                  annotations=annotations,
                                  **kwargs)
        save_path = utils.get_approach_save_path_str()
        # NS predicates currently doesn't support saving
        # if not CFG.neu_sym_predicate:
        #     with open(f"{save_path}_{online_learning_cycle}.NSRTs", "wb") as f:
        #         pkl.dump(self._nsrts, f)
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
        start_time = time.perf_counter()
        preds = self._get_current_predicates()
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
                init_atoms, objects, self._nsrts, allow_noops=True)
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
        for nsrt in self._nsrts:
            complexity += nsrt.op.get_complexity()
        time_taken = time.perf_counter() - start_time
        self._metrics["sidelining_obj_num_plans_up_to_n"] = num_plans_up_to_n
        self._metrics["sidelining_obj_complexity"] = complexity
        self._metrics["sidelining_obj_time_taken"] = time_taken
        logging.info(f"\tFinished in {time_taken:.3f} seconds")
        logging.info(f"\tGot num_plans_up_to_n {num_plans_up_to_n} and "
                     f"complexity {complexity}")
