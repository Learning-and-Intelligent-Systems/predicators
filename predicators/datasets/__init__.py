"""Create offline datasets for training, given a set of training tasks for an
environment."""

import os
from typing import List, Sequence, Set

import numpy as np

from predicators import utils
from predicators.datasets.demo_only import create_demo_data
from predicators.datasets.demo_replay import create_demo_replay_data
from predicators.datasets.ground_atom_data import create_ground_atom_data
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Dataset, GroundAtom, \
    LowLevelTrajectory, Object, ParameterizedOption, Predicate, State, Task, \
    Type


def create_dataset(env: BaseEnv, train_tasks: List[Task],
                   known_options: Set[ParameterizedOption]) -> Dataset:
    """Create offline datasets for training, given a set of training tasks for
    an environment.

    Some or all of this data may be loaded from disk.
    """
    if CFG.offline_data_method == "demo":
        return create_demo_data(env,
                                train_tasks,
                                known_options,
                                annotate_with_gt_ops=False)
    if CFG.offline_data_method == "demo+gt_operators":
        return create_demo_data(env,
                                train_tasks,
                                known_options,
                                annotate_with_gt_ops=True)
    if CFG.offline_data_method == "demo+replay":
        return create_demo_replay_data(env, train_tasks, known_options)
    if CFG.offline_data_method == "demo+ground_atoms":
        base_dataset = create_demo_data(env,
                                        train_tasks,
                                        known_options,
                                        annotate_with_gt_ops=False)
        _, excluded_preds = utils.parse_config_excluded_predicates(env)
        n = int(CFG.teacher_dataset_num_examples)
        assert n >= 1, "Must have at least 1 example of each predicate"
        return create_ground_atom_data(env, base_dataset, excluded_preds, n)
    if CFG.offline_data_method == "demo+handlabeled_atoms":
        dataset_fpath = os.path.join(CFG.data_dir, CFG.handmade_demo_filename)
        # First, parse this dataset into a structured form.
        structured_states, structured_actions = utils.parse_handmade_vlmtraj_file_into_structured_trajs(
            dataset_fpath)
        assert len(structured_states) == len(structured_actions)

        # Now, parse out and make all necessary predicates.
        # Also, start making a list of all the names of different objects.
        # For now, assume that there is only one type called "object".
        assert len(env.types) == 1
        type_name_to_type = {t.name: t for t in env.types}
        assert "object" in type_name_to_type
        assert type_name_to_type["object"].dim == 0

        def _stripped_classifier(state: State,
                                 objects: Sequence[Object]) -> bool:
            raise Exception("Stripped classifier should never be called!")

        # NOTE: for now, the predicates aren't typed, but rather all predicates
        # consumer things of type "default_type".
        pred_name_to_pred = {}
        obj_name_to_obj = {}
        atoms_trajs = []
        for traj in structured_states:
            curr_atoms_traj = []
            for structured_state in traj:
                curr_ground_atoms_state = set()
                for pred_name, objs_and_val_dict in structured_state.items():
                    # IMPORTANT: this currently assumes that the data is such
                    # that a predicate with a certain name (e.g. "Sliced")
                    # always appears with the same number of object arguments
                    # (e.g. Sliced(apple), and never
                    # Sliced(apple, cutting_tool)). We might want to explicitly
                    # check for this in the future.
                    if pred_name not in pred_name_to_pred:
                        assert len(objs_and_val_dict.keys()) == 1
                        for obj_args in objs_and_val_dict.keys():
                            pred_name_to_pred[pred_name] = Predicate(
                                pred_name,
                                [type_name_to_type["object"] for _ in range(len(obj_args))],
                                _stripped_classifier)
                            for obj_name in obj_args:
                                if obj_name not in obj_name_to_obj:
                                    obj_name_to_obj[obj_name] = Object(
                                        obj_name, type_name_to_type["object"])
                    # Given that we've now built up predicates and object
                    # dictionaries. We can now convert the current state into
                    # ground atoms!
                    for obj_args, truth_value in objs_and_val_dict.items():
                        if truth_value:
                            curr_ground_atoms_state.add(
                                GroundAtom(
                                    pred_name_to_pred[pred_name],
                                    [obj_name_to_obj[o] for o in obj_args]))
                curr_atoms_traj.append(curr_ground_atoms_state)
            atoms_trajs.append(curr_atoms_traj)

        # Construct a default state that just contains all the objects in the
        # domain with no attributes.
        state_dict = {}
        for obj in obj_name_to_obj.values():
            state_dict[obj] = []
        default_state = State(state_dict)

        # Next, we link option names to actual options.
        option_name_to_option = {o.name: o for o in known_options}
        option_trajs = []
        for traj in structured_actions:
            curr_option_traj = []
            for structured_action in traj:
                option = option_name_to_option[structured_action[0]]
                for obj_name in structured_action[1]:
                    if obj_name not in obj_name_to_obj:
                        obj_name_to_obj[obj_name] = Object(
                            obj_name, type_name_to_type["object"])
                ground_option = option.ground([
                    obj_name_to_obj[obj_name]
                    for obj_name in structured_action[1]
                ], np.array([]))
                # Call initiable here because we will be calling
                # terminal later, and initiable always needs
                # to be called first.
                ground_option.initiable(default_state)
                curr_option_traj.append(ground_option)
            option_trajs.append(curr_option_traj)

        # Next, turn these structures into a Dataset.
        trajs = []
        for traj_num in range(len(structured_actions)):
            curr_traj_states = []
            curr_traj_actions = []
            for idx_within_traj in range(len(structured_actions[traj_num])):
                curr_traj_states.append(default_state)
                curr_traj_actions.append(
                    Action(np.zeros(0),
                           option_trajs[traj_num][idx_within_traj]))
            # Need to append one last default state because
            # there are 1 more states than actions.
            curr_traj_states.append(default_state)
            curr_traj = LowLevelTrajectory(curr_traj_states, curr_traj_actions,
                                           True, traj_num)
            trajs.append(curr_traj)

        # Finally, package everything together into a Dataset!
        return Dataset(trajs, atoms_trajs)

    if CFG.offline_data_method == "empty":
        return Dataset([])
    raise NotImplementedError("Unrecognized dataset method.")
