"""Functions to create offline demonstration data by leveraging VLMs."""

import logging
import os
import re
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np

from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Dataset, GroundAtom, \
    LowLevelTrajectory, Object, ParameterizedOption, Predicate, State, Task, \
    _Option


def _parse_structured_state_into_ground_atoms(
    env: BaseEnv,
    train_tasks: List[Task],
    structured_state_trajs: List[List[Dict[str, Dict[Tuple[str, ...], bool]]]],
) -> List[List[Set[GroundAtom]]]:
    """Convert structured state trajectories into actual trajectories of ground
    atoms."""
    # We check a number of important properties before starting.
    # Firstly, the number of train tasks must equal the number of structured
    # state demos we have.
    assert len(train_tasks) == len(structured_state_trajs)
    # Secondly, we assume there is only one goal predicate, and that it is
    # a dummy goal predicate.
    assert len(env.goal_predicates) == 1
    goal_preds_list = list(env.goal_predicates)
    goal_predicate = goal_preds_list[0]
    assert goal_predicate.name == "DummyGoal"
    # We also assume that there is precisely one "object" type that is
    # a superset of all other object types.
    obj_type = None
    for t in env.types:
        obj_type = t.oldest_ancestor
        assert obj_type.name == "object"
    assert obj_type is not None

    def _stripped_classifier(
            state: State,
            objects: Sequence[Object]) -> bool:  # pragma: no cover.
        raise Exception("Stripped classifier should never be called!")

    pred_name_to_pred = {}
    atoms_trajs = []
    # Loop through all trajectories in the structured_state_trajs and convert
    # each one to a sequence of sets of GroundAtoms.
    for i, traj in enumerate(structured_state_trajs):
        curr_atoms_traj = []
        objs_for_task = set(train_tasks[i].init)
        curr_obj_name_to_obj = {obj.name: obj for obj in objs_for_task}
        # NOTE: We assume that there is precisely one dummy object that is
        # used to track whether the dummy goal has been reached or not.
        assert "dummy_goal_obj" in curr_obj_name_to_obj
        # Create a goal atom for this demonstration using the goal predicate.
        goal_atom = GroundAtom(goal_predicate,
                               [curr_obj_name_to_obj["dummy_goal_obj"]])
        for structured_state in traj:
            curr_ground_atoms_state = set()
            for pred_name, objs_and_val_dict in structured_state.items():
                # IMPORTANT NOTE: this currently assumes that the data is such
                # that a predicate with a certain name (e.g. "Sliced")
                # always appears with the same number of object arguments
                # (e.g. Sliced(apple), and never
                # Sliced(apple, cutting_tool)). We might want to explicitly
                # check for this in the future.
                if pred_name not in pred_name_to_pred:
                    if len(objs_and_val_dict.keys()) == 1:
                        # In this case, we make a predicate that takes in
                        # exactly one types argument.
                        for obj_args in objs_and_val_dict.keys():
                            # We need to construct the types being
                            # fed into this predicate.
                            pred_types = []
                            for obj_name in obj_args:
                                curr_obj = curr_obj_name_to_obj[obj_name]
                                pred_types.append(curr_obj.type)
                            pred_name_to_pred[pred_name] = Predicate(
                                pred_name, pred_types, _stripped_classifier)
                    else:
                        # In this case, we need to make a predicate that
                        # takes in the generic 'object' type such that
                        # multiple different objs could potentially be
                        # subbed in.
                        # Start by checking that the number of object
                        # args are always the same
                        num_args = 0
                        for obj_args in objs_and_val_dict.keys():
                            if num_args == 0:
                                num_args = len(obj_args)
                            else:
                                assert num_args == len(obj_args)
                        # Given this, add one new predicate with num_args
                        # number of 'object' type arguments.
                        pred_name_to_pred[pred_name] = Predicate(
                            pred_name, [obj_type for _ in range(num_args)],
                            _stripped_classifier)

                # Given that we've now built up predicates and object
                # dictionaries. We can now convert the current state into
                # ground atoms!
                for obj_args, truth_value in objs_and_val_dict.items():
                    if truth_value:
                        curr_ground_atoms_state.add(
                            GroundAtom(
                                pred_name_to_pred[pred_name],
                                [curr_obj_name_to_obj[o] for o in obj_args]))
            curr_atoms_traj.append(curr_ground_atoms_state)
        # Add the goal atom at the end of the trajectory.
        curr_atoms_traj[-1].add(goal_atom)
        atoms_trajs.append(curr_atoms_traj)
    return atoms_trajs


def _parse_structured_actions_into_ground_options(
        structured_actions_trajs: List[List[Tuple[str, Tuple[str, ...],
                                                  List[float]]]],
        known_options: Set[ParameterizedOption],
        train_tasks: List[Task]) -> List[List[_Option]]:
    """Convert structured actions trajectories into actual lists of ground
    options trajectories."""
    assert len(structured_actions_trajs) == len(train_tasks)
    option_name_to_option = {o.name: o for o in known_options}
    option_trajs = []
    for i, traj in enumerate(structured_actions_trajs):
        curr_obj_name_to_obj = {
            obj.name: obj
            for obj in set(train_tasks[i].init)
        }
        curr_option_traj = []
        for structured_action in traj:
            option = option_name_to_option[structured_action[0]]
            ground_option = option.ground([
                curr_obj_name_to_obj[obj_name]
                for obj_name in structured_action[1]
            ], np.array(structured_action[2]))
            # Call initiable here because we will be calling
            # terminal later, and initiable always needs
            # to be called first.
            ground_option.initiable(train_tasks[i].init)
            curr_option_traj.append(ground_option)
        option_trajs.append(curr_option_traj)
    return option_trajs


def _create_dummy_goal_state_for_each_task(
        env: BaseEnv, train_tasks: List[Task]) -> List[State]:
    """Uses a lot of assumptions to generate a state in which a dummy goal
    predicate holds for each train task."""
    # FOR NOW, we assume there is only one goal predicate, and that it is
    # a dummy goal predicate. In the future, we will implement and use
    # proper goal predicates.
    assert len(env.goal_predicates) == 1
    goal_preds_list = list(env.goal_predicates)
    goal_predicate = goal_preds_list[0]
    assert goal_predicate.name == "DummyGoal"
    goal_states = []
    for train_task in train_tasks:
        curr_task_obj_name_to_obj = {obj.name: obj for obj in train_task.init}
        assert "dummy_goal_obj" in curr_task_obj_name_to_obj
        dummy_goal_feats = curr_task_obj_name_to_obj[
            "dummy_goal_obj"].type.feature_names
        assert len(dummy_goal_feats) == 1
        assert dummy_goal_feats[0] == "goal_true"
        curr_task_goal_atom = GroundAtom(
            goal_predicate, [curr_task_obj_name_to_obj["dummy_goal_obj"]])
        assert not curr_task_goal_atom.holds(train_task.init)
        curr_goal_state = train_task.init.copy()
        curr_goal_state.set(curr_task_obj_name_to_obj["dummy_goal_obj"],
                            "goal_true", 1.0)
        assert curr_task_goal_atom.holds(curr_goal_state)
        goal_states.append(curr_goal_state)
    return goal_states


def _convert_ground_option_trajs_into_lowleveltrajs(
        option_trajs: List[List[_Option]], dummy_goal_states: List[State],
        train_tasks: List[Task]) -> List[LowLevelTrajectory]:
    """Convert option trajectories into LowLevelTrajectories to be used in
    constructing a Dataset."""
    assert len(option_trajs) == len(dummy_goal_states) == len(train_tasks)
    # NOTE: In this LowLevelTrajectory, we assume the low level states
    # are the same as the init state until the final state.
    trajs = []
    for traj_num in range(len(option_trajs)):
        traj_init_state = train_tasks[traj_num].init
        curr_traj_states = []
        curr_traj_actions = []
        for idx_within_traj in range(len(option_trajs[traj_num])):
            curr_traj_states.append(traj_init_state)
            curr_traj_actions.append(
                Action(np.zeros(0, dtype=float),
                       option_trajs[traj_num][idx_within_traj]))
        # Now, we need to append the final state because there are 1 more
        # states than actions.
        curr_traj_states.append(dummy_goal_states[traj_num])
        curr_traj = LowLevelTrajectory(curr_traj_states, curr_traj_actions,
                                       True, traj_num)
        trajs.append(curr_traj)
    return trajs


def _debug_log_atoms_trajs(
        ground_atoms_trajs: List[List[Set[GroundAtom]]]) -> None:
    """Debug log the changes in atoms trajectories for easy human-checking."""
    # Log trajectory information in a very easy to parse format for
    # debugging.
    for traj in ground_atoms_trajs:
        logging.debug(f"Step 0 atoms: {sorted(traj[0])}")
        for i in range(1, len(traj)):
            logging.debug(f"Step {i} add effs: {sorted(traj[i] - traj[i-1])}")
            logging.debug(f"Step {i} del effs: {sorted(traj[i-1] - traj[i])}")
        logging.debug("\n")


def _parse_options_txt_into_structured_actions(
        text: str) -> List[Tuple[str, Tuple[str, ...], List[float]]]:
    """Given text that contains a series of ground options convert this into a
    structured set of tuples suitable for later conversion into more structured
    GroundAtomTrajectories."""
    structured_actions_output = []
    pattern_option = r'(\w+)\(([^)]*)\)\[([\d.,\s]*)\] ->'
    option_matches = re.findall(pattern_option, text)
    for i in range(len(option_matches)):
        current_option_with_objs = (option_matches[i][0],
                                    tuple(
                                        map(str.strip,
                                            option_matches[i][1].split(','))))
        continuous_params_floats = [
            float(float_str.strip(' '))
            for float_str in option_matches[i][2].split(',')
            if len(float_str) > 0
        ]
        structured_actions_output.append(
            (current_option_with_objs[0], current_option_with_objs[1],
             continuous_params_floats))
    return structured_actions_output


def _parse_atoms_txt_into_structured_state(
        text: str) -> List[Dict[str, Dict[Tuple[str, ...], bool]]]:
    """Given text that contains a series of ground atoms labelled with their
    specific truth values, convert this into a structured dictionary suitable
    for later conversion into more structured GroundAtomTrajectories."""
    pattern_block_of_state = r"\{(.*?[^\d,\s].*?)\}"
    pattern_predicate = r'(\w+)\(([^)]+)\): (\w+).'
    state_blocks_matches = re.findall(pattern_block_of_state, text, re.DOTALL)
    structured_state_output = []
    for state_block_match in state_blocks_matches:
        predicate_matches_within_state_block = re.findall(
            pattern_predicate, state_block_match)
        current_predicate_data: Dict[str, Dict[Tuple[str, ...], bool]] = {}
        for predicate_match in predicate_matches_within_state_block:
            classifier_name = predicate_match[0]
            objects = tuple(map(str.strip, predicate_match[1].split(',')))
            truth_value = predicate_match[2] == 'True'
            if classifier_name not in current_predicate_data:
                current_predicate_data[classifier_name] = {}
            current_predicate_data[classifier_name][objects] = truth_value
        structured_state_output.append(current_predicate_data.copy())
    return structured_state_output


def _parse_vlmtraj_into_structured_traj(
    text: str
) -> Tuple[List[Dict[str, Dict[Tuple[str, ...], bool]]], List[Tuple[str, Tuple[
        str, ...], List[float]]]]:
    """Parse a handwritten trajectory saved as text into a structured
    representation that can be used to convert these into a more structured
    description suitable for later conversion into GroundAtomTrajectories.

    This function outputs two lists. The first contains a dictionary
    whose keys are names of predicates, and whose values are a dict
    mapping a tuple of object names to a boolean value for the ground
    predicate at this particular timestep. The second contains a tuple
    whose first element is the current option name, and the second
    element contains all the objects used by this option.
    """
    structured_state = _parse_atoms_txt_into_structured_state(text)
    structured_actions = _parse_options_txt_into_structured_actions(text)
    assert len(structured_state) == len(
        structured_actions
    ) + 1, "Manual data malformed; num states != 1 + num options."
    return (structured_state, structured_actions)


def _parse_vlmtraj_file_into_structured_trajs(
    filename: str
) -> Tuple[List[List[Dict[str, Dict[Tuple[str, ...], bool]]]], List[List[Tuple[
        str, Tuple[str, ...], List[float]]]]]:
    """Parse a txt file full of handwritten trajectories into a structured
    representation that can be used to convert these into
    GroundAtomTrajectories suitable for predicate invention, operator learning,
    etc.

    We assume the vlmtraj is saved in a txt file with an encoding scheme
    described in:
    `approaches/documentation/grammar_search_invention_approach.md`.
    This function outputs two lists of lists, where each element is the output
    of the above parse_handmade_vlmtraj_into_structured_traj function.
    """
    with open(filename, "r", encoding="utf8") as f:
        full_file_text = f.read()
    pattern = r"(?<====\n)(.*?)(?=\n===)"
    matches = re.findall(pattern, full_file_text, re.DOTALL)
    output_state_trajs, output_action_trajs = [], []
    for match in matches:
        curr_state_traj, curr_action_traj = _parse_vlmtraj_into_structured_traj(
            match)
        output_state_trajs.append(curr_state_traj)
        output_action_trajs.append(curr_action_traj)
    return (output_state_trajs, output_action_trajs)


def create_ground_atom_data_from_labeled_txt(
        env: BaseEnv, train_tasks: List[Task],
        known_options: Set[ParameterizedOption]) -> Dataset:
    """Given a txt file containing trajectories labelled with VLM predicate
    values, construct a dataset that can be passed to the rest of our learning
    pipeline."""
    dataset_fpath = os.path.join(CFG.data_dir, CFG.handmade_demo_filename)
    # First, parse this dataset into a structured form.
    structured_states, structured_actions = \
        _parse_vlmtraj_file_into_structured_trajs(dataset_fpath)
    assert len(structured_states) == len(structured_actions)
    # Next, take this intermediate structured form and further
    # parse it into ground atoms and ground options respectively.
    ground_atoms_trajs = _parse_structured_state_into_ground_atoms(
        env, train_tasks, structured_states)
    _debug_log_atoms_trajs(ground_atoms_trajs)
    option_trajs = _parse_structured_actions_into_ground_options(
        structured_actions, known_options, train_tasks)
    # We also need to create the goal state for every train task.
    goal_states_for_every_traj = _create_dummy_goal_state_for_each_task(
        env, train_tasks)
    # Finally, we need to construct actual LowLevelTrajectories.
    low_level_trajs = _convert_ground_option_trajs_into_lowleveltrajs(
        option_trajs, goal_states_for_every_traj, train_tasks)
    return Dataset(low_level_trajs, ground_atoms_trajs)
