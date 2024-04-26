"""Functions to create offline demonstration data by leveraging VLMs."""

import ast
import glob
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Sequence, Set, Tuple

import numpy as np
import PIL.Image

from predicators import utils
from predicators.envs import BaseEnv
from predicators.envs.vlm_envs import VLMPredicateEnv
from predicators.pretrained_model_interface import GoogleGeminiVLM, \
    VisionLanguageModel
from predicators.settings import CFG
from predicators.structs import Action, Dataset, GroundAtom, \
    ImageOptionTrajectory, LowLevelTrajectory, Object, ParameterizedOption, \
    Predicate, State, Task, _Option


def _generate_prompt_for_atom_proposals(
        traj: ImageOptionTrajectory, trajectory_subsample_freq: int
) -> List[Tuple[str, List[PIL.Image.Image]]]:
    """Prompt for generating proposals for atoms."""
    ret_list = []
    if CFG.grammar_search_vlm_atom_proposal_prompt_type == "naive_each_step":
        prompt = (
            "You are a robotic vision system whose job is to output a "
            "structured set of predicates useful for running a "
            "task and motion "
            "planning system from the following scene. Please provide "
            "predicates in terms of the following objects: "
            f"{[str(obj.name) for obj in traj.objects if obj.name != 'dummy_goal_obj']}. "  # pylint:disable=line-too-long
            "For each predicate, output it in the following format: "
            "predicate_name(obj1, obj2, obj3...) "
            "(for instance is_sliced(apple), is_not_sliced(apple), etc.). "
            "Also, for each predicate you list, list its negation. "
            "List as many predicates as you can possibly think of, even if "
            "they're only "
            "tangentially relevant to what you see in the scene and even "
            "if they're false, "
            "given the following scene taken from a demonstration for the "
            "task."
            "Do not list any other text other than the names and arguments "
            "of predicates. "
            "List each proposal as a bulleted list item on a separate line.")
        i = 0
        while i < len(traj.imgs):
            ret_list.append((prompt, traj.imgs[i]))
            i += trajectory_subsample_freq
    elif CFG.grammar_search_vlm_atom_proposal_prompt_type == \
        "naive_whole_traj":
        prompt = (
            "You are a robotic vision system whose job is to output a "
            "structured set of predicates useful for describing the "
            "important concepts from "
            "the following demonstration. Please provide predicates "
            "in terms of the objects: "
            f"{[obj.name for obj in traj.objects if obj.name != 'dummy_goal_obj']}. "  # pylint:disable=line-too-long
            "For each predicate, output it in the following format: "
            "predicate_name(obj1, obj2, obj3...) "
            "(for instance is_sliced(apple), is_not_sliced(apple), etc.). "
            "Also, for each predicate you list, list its negation. "
            "Generate as many predicates as you can possibly think of, "
            "even if they're only "
            "tangentially relevant to the task goal: 'make a cup of ice tea'"
            "Do not list any other text other than the names and arguments "
            "of predicates. "
            "List each proposal as a bulleted list item on a separate line.")
        # NOTE: we rip out just one img from each of the state images.
        # This is fine/works for the case where we only have one
        # camera view, but probably will need to be amended in the future!
        ret_list.append(
            (prompt, [traj.imgs[i][0] for i in range(len(traj.imgs))]))
    elif CFG.grammar_search_vlm_atom_proposal_prompt_type == \
        "options_labels_whole_traj":
        prompt = (
            "You are a robotic vision system whose job is to output a "
            "structured "
            "set of predicates useful for describing important concepts "
            "in the "
            "following demonstration of a task. You will be provided with "
            "a list "
            "of actions used during the task, as well as images of "
            "states before "
            "and after every action execution. Please provide predicates "
            "in terms "
            "of the following objects: [teabag, hand, plate, cup, ice]. "
            "For each "
            "predicate, output it in the following format: "
            "predicate_name(obj1, obj2, obj3...). Start by generating "
            "predicates "
            "that change before and after each action. After this, "
            "generate any "
            "other predicates that perhaps do not change but are still "
            "important to describing the demonstration shown."
            "\n\nActions executed as part of demonstration:\n")
        prompt += "\n".join(act.name + str(act.objects)
                            for act in traj.actions)
        # NOTE: we rip out just one img from each of the state images.
        # This is fine/works for the case where we only have one camera
        # view, but probably will need to be amended in the future!
        ret_list.append(
            (prompt, [traj.imgs[i][0] for i in range(len(traj.imgs))]))
    else:
        raise ValueError("Unknown VLM prompting option " +
                         f"{CFG.grammar_search_vlm_atom_proposal_prompt_type}")

    return ret_list


def _generate_prompt_for_scene_labelling(
        traj: ImageOptionTrajectory,
        atoms_list: List[str]) -> List[Tuple[str, List[PIL.Image.Image]]]:
    """Prompt for generating labels for truth values of atoms in a scene."""
    ret_list = []
    if "per_scene" in CFG.grammar_search_vlm_atom_label_prompt_type:
        if CFG.grammar_search_vlm_atom_label_prompt_type == "per_scene_naive":
            prompt = (
                "You are a vision system for a robot. Your job is to output "
                "the values of the following predicates based on the provided "
                "visual scene. For each predicate, output True, False, or "
                "Unknown if the relevant objects are not in the scene or the "
                "value of the predicate simply cannot be determined. "
                "Output each predicate value as a bulleted list with each "
                "predicate and value on a different line. "
                "Use the format: <predicate>: <truth_value>. "
                "Ensure there is a period ('.') after every list item. "
                "Do not output any text except the names and truth values of "
                "predicates."
                "\nPredicates:")
        elif CFG.grammar_search_vlm_atom_label_prompt_type == "per_scene_cot":
            prompt = (
                "You are a vision system for a robot. Your job is to output "
                "the values of the following predicates based on the provided "
                "visual scene. For each predicate, output True, False, or "
                "Unknown if the relevant objects are not in the scene or the "
                "value of the predicate simply cannot be determined. "
                "Output each predicate value as a bulleted list with each "
                "predicate and value on a different line. "
                "For each output value, provide an explanation as to why you "
                "labelled this predicate as having this particular value."
                "Use the format: <predicate>: <truth_value>. <explanation>."
                "\nPredicates:")
        else:
            raise ValueError(
                "Unknown option for grammar_search_vlm_atom_label_prompt_type"
                f"{CFG.grammar_search_vlm_atom_label_prompt_type}")
        for atom_str in atoms_list:
            prompt += f"\n{atom_str}"
        for currimgs in traj.imgs:
            # NOTE: we rip out just one img from each of the state
            # images. This is fine/works for the case where we only
            # have one camera view, but probably will need to be
            # amended in the future!
            ret_list.append((prompt, [currimgs[0]]))
    return ret_list


def _sample_init_atoms_from_trajectories(
        trajectories: List[ImageOptionTrajectory],
        vlm: VisionLanguageModel,
        trajectory_subsample_freq: int = 1) -> List[List[str]]:
    """Given a list of ImageOptionTrajectories, query a VLM to generate a list
    of names of ground atoms from which we can extract predicates that might be
    relevant for planning to recreate these trajectories."""
    aggregated_vlm_output_strs = []
    all_vlm_queries_list = []
    for traj in trajectories:
        all_vlm_queries_list += _generate_prompt_for_atom_proposals(
            traj, trajectory_subsample_freq)
    curr_num_queries = 0
    total_num_queries = len(all_vlm_queries_list)
    for query in all_vlm_queries_list:
        aggregated_vlm_output_strs.append(
            vlm.sample_completions(query[0],
                                   query[1],
                                   0.0,
                                   CFG.seed,
                                   num_completions=1))
        curr_num_queries += 1
        logging.info("Completed (%s/%s) init atoms queries to the VLM.",
                     curr_num_queries, total_num_queries)
    return aggregated_vlm_output_strs


def _label_trajectories_with_atom_values(
        trajectories: List[ImageOptionTrajectory], vlm: VisionLanguageModel,
        atoms_list: List[str]) -> List[List[str]]:
    """Given a list of atoms, label every state in ImageOptionTrajectories with
    the truth values of a set of atoms."""
    total_scenes_to_label = sum(len(traj.imgs) for traj in trajectories)
    curr_scenes_labelled = 0
    output_labelled_atoms_txt_list = []
    for traj in trajectories:
        prompts_for_traj = _generate_prompt_for_scene_labelling(
            traj, atoms_list)
        curr_traj_txt_outputs = []
        for prompt in prompts_for_traj:
            # Sample VLM outputs with temperature 0 in an attempt to be
            # accurate.
            curr_vlm_atom_labelling = vlm.sample_completions(prompt[0],
                                                             prompt[1],
                                                             0.0,
                                                             CFG.seed,
                                                             num_completions=1)
            assert len(curr_vlm_atom_labelling) == 1
            sanitized_output = curr_vlm_atom_labelling[0].replace('\\', '')
            curr_traj_txt_outputs.append(sanitized_output)
            curr_scenes_labelled += 1
            logging.info("Completed (%s/%s) label queries to VLM!",
                         curr_scenes_labelled, total_scenes_to_label)
        output_labelled_atoms_txt_list.append(curr_traj_txt_outputs)
    return output_labelled_atoms_txt_list


def _parse_unique_atom_proposals_from_list(
        atom_strs_proposals_list: List[List[str]],
        relevant_objects_across_demos: Set[Object]) -> Set[str]:
    """Given a list of atom proposals that a VLM has constructed for each
    demonstration, parse these to a unique set of proposals.

    This function currently does 3 steps of sanitization: (1) removing
    any unnecessary characters, (2) removing any atoms that involve
    objects that aren't relevant, (3) removing any duplicate atoms.
    """
    atoms_strs_set = set()
    obj_names_set = set(obj.name for obj in relevant_objects_across_demos)
    num_atoms_considered = 0
    for atoms_proposal_for_traj in atom_strs_proposals_list:
        assert len(atoms_proposal_for_traj) == 1
        curr_atoms_proposal = atoms_proposal_for_traj[0]
        # Regex pattern to match predicates
        atom_match_pattern = r"\b[a-z_]+\([a-z0-9, ]+\)"
        # Find all matches in the text
        matches = re.findall(atom_match_pattern, curr_atoms_proposal)
        for atom_proposal_txt in matches:
            num_atoms_considered += 1
            atom_is_valid = True
            atom = re.sub(r"[^\w\s\(\),]", "", atom_proposal_txt).strip(' ')
            obj_names = re.findall(r'\((.*?)\)', atom)
            if obj_names:
                obj_names_list = [
                    name.strip() for name in obj_names[0].split(',')
                ]
                for obj_name in obj_names_list:
                    if obj_name not in obj_names_set:
                        atom_is_valid = False
                        break
            if atom_is_valid:
                atoms_strs_set.add(atom)
            logging.debug(f"Proposed atom: {atom} is valid: {atom_is_valid}")
    logging.info(f"VLM proposed a total of {num_atoms_considered} atoms.")
    logging.info(f"Of these, {len(atoms_strs_set)} were valid and unique.")
    return atoms_strs_set


def save_labelled_trajs_as_txt(
        env: BaseEnv, labelled_atoms_trajs: List[List[str]],
        ground_option_trajs: List[List[_Option]]) -> None:
    """Save a txt file with a text representation of GroundAtomTrajectories.

    This serves as a human-readable intermediary output for debugging,
    and also as a convenient restart point for the pipeline (i.e., these
    txt files can be loaded and the rest of the pipeline run from
    there)!
    """
    # All trajectories are delimited between pairs of "===".
    save_str = "===\n"
    assert len(labelled_atoms_trajs) == len(ground_option_trajs)
    for curr_atoms_traj, curr_option_traj in zip(labelled_atoms_trajs,
                                                 ground_option_trajs):
        assert len(curr_option_traj) + 1 == len(curr_atoms_traj)
        for option_ts in range(len(curr_option_traj)):
            curr_atom_state_str = curr_atoms_traj[option_ts]
            # Wrap the state in curly brackets.
            curr_state_str = "{" + curr_atom_state_str + "} ->"
            curr_option = curr_option_traj[option_ts]
            curr_option_str = curr_option.name + "("
            for obj in curr_option.objects:
                curr_option_str += str(obj.name) + ", "
            curr_option_str = curr_option_str[:-2] + ")" + str(
                curr_option.params.tolist()) + " -> "
            save_str += curr_state_str + "\n\n" + curr_option_str + "\n\n"
        # At the end of the trajectory, we need to append the final state,
        # and a "===" delimiter.
        final_atom_state_str = curr_atoms_traj[-1]
        final_state_str = "{" + final_atom_state_str + "}\n"
        save_str += final_state_str + "===\n"
    # Finally, save this constructed string as a txt file!
    txt_filename = f"{env.get_name()}__demo+labeled_atoms__manual__" + \
    f"{len(labelled_atoms_trajs)}.txt"
    filepath = os.path.join(CFG.data_dir, txt_filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(save_str)
    logging.info(f"Human-readable labelled trajectory saved to {filepath}!")


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

    def _stripped_classifier(state: State, objects: Sequence[Object]) -> bool:
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


def _parse_structuredactions_into_ground_options(
        structuredactions_trajs: List[List[Tuple[str, Tuple[str, ...],
                                                 List[float]]]],
        known_options: Set[ParameterizedOption],
        train_tasks: List[Task]) -> List[List[_Option]]:
    """Convert structured actions trajectories into actual lists of ground
    options trajectories."""
    assert len(structuredactions_trajs) == len(train_tasks)
    option_name_to_option = {o.name: o for o in known_options}
    option_trajs = []
    for i, traj in enumerate(structuredactions_trajs):
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
    # We assume there is only one goal predicate, and that it is
    # a dummy goal predicate.
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
        curr_trajactions = []
        for idx_within_traj in range(len(option_trajs[traj_num])):
            curr_traj_states.append(traj_init_state)
            curr_trajactions.append(
                Action(np.zeros(0, dtype=float),
                       option_trajs[traj_num][idx_within_traj]))
        # Now, we need to append the final state because there are 1 more
        # states than actions.
        curr_traj_states.append(dummy_goal_states[traj_num])
        curr_traj = LowLevelTrajectory(curr_traj_states, curr_trajactions,
                                       True, traj_num)
        trajs.append(curr_traj)
    return trajs


def _pretty_print_atoms_trajs(
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


def create_ground_atom_data_from_img_trajs(
        env: BaseEnv, train_tasks: List[Task],
        known_options: Set[ParameterizedOption]) -> Dataset:
    """Given a folder containing trajectories that have images of scenes for
    each state, as well as options that transition between these states, output
    a dataset."""
    trajectories_folder_path = os.path.join(CFG.data_dir,
                                            CFG.vlm_trajs_folder_name)
    # First, run some checks on the folder name to make sure
    # we're not accidentally loading the wrong one.
    folder_name_components = CFG.vlm_trajs_folder_name.split('__')
    assert folder_name_components[0] == CFG.env
    assert folder_name_components[1] == "vlm_demos"
    assert int(folder_name_components[2]) == CFG.seed
    assert int(folder_name_components[3]) == CFG.num_train_tasks
    num_trajs = len(os.listdir(trajectories_folder_path))
    assert num_trajs == CFG.num_train_tasks
    option_name_to_option = {opt.name: opt for opt in known_options}
    image_option_trajs = []
    all_task_objs = set()
    for train_task_idx, path in enumerate(
            sorted(Path(trajectories_folder_path).iterdir())):
        assert path.is_dir()
        state_folders = [f.path for f in os.scandir(path) if f.is_dir()]
        num_states_in_traj = len(state_folders)
        state_traj = []
        for state_num in range(num_states_in_traj):
            currimgs = []
            curr_state_path = path.joinpath(str(state_num))
            # NOTE: we assume all images are saved as jpg files.
            img_files = glob.glob(str(curr_state_path) + "/*.jpg")
            for img in img_files:
                currimgs.append(PIL.Image.open(img))
            state_traj.append(currimgs)
        # Get objects from train tasks to be used for future parsing.
        curr_train_task = train_tasks[train_task_idx]
        curr_task_objs = set(curr_train_task.init)
        all_task_objs |= curr_task_objs
        curr_task_obj_name_to_obj = {obj.name: obj for obj in curr_task_objs}
        # Parse out actions for the trajectory.
        options_traj_file_list = glob.glob(str(path) + "/*.txt")
        assert len(options_traj_file_list) == 1
        options_traj_file = options_traj_file_list[0]
        with open(options_traj_file, "r", encoding="utf-8") as f:
            options_file_str = f.read()
        option_names_list = re.findall(r'(\w+)\(', options_file_str)
        parsed_str_objects = re.findall(r'\((.*?)\)', options_file_str)
        object_args_list = [obj.split(', ') for obj in parsed_str_objects]
        # Remove empty square brackets from the object_args_list.
        for object_arg_sublist in object_args_list:
            object_arg_sublist.remove('[]')
        parameters = [
            ast.literal_eval(obj) if obj else []
            for obj in re.findall(r'\[(.*?)\]', options_file_str)
        ]
        ground_option_traj: List[_Option] = []
        # Now actually create ground options.
        for option_name, option_objs_strs_list, option_params in zip(
                option_names_list, object_args_list, parameters):
            objects = [
                curr_task_obj_name_to_obj[opt_arg]
                for opt_arg in option_objs_strs_list
            ]
            option = option_name_to_option[option_name]
            ground_option = option.ground(objects, np.array(option_params))
            # NOTE: we assert the option was initiable in the env's initial
            # state because during learning, we will assert that the option's
            # initiable function was previously called.
            assert ground_option.initiable(curr_train_task.init)
            ground_option_traj.append(ground_option)
        # Given ground options, we can finally make ImageOptionTrajectories.
        image_option_trajs.append(
            ImageOptionTrajectory(list(curr_task_objs), state_traj,
                                  ground_option_traj, True, train_task_idx))
    # Given trajectories, we can now query the VLM to get proposals for ground
    # atoms that might be relevant to decision-making.
    gemini_vlm = GoogleGeminiVLM(CFG.vlm_model_name)
    if not CFG.grammar_search_vlm_atom_proposal_use_debug:
        logging.info("Querying VLM for candidate atom proposals...")
        atom_strs_proposals_list = _sample_init_atoms_from_trajectories(
            image_option_trajs, gemini_vlm, 1)
        logging.info("Done querying VLM for candidate atoms!")
        # We now parse and sanitize this set of atoms.
        atom_proposals_set = _parse_unique_atom_proposals_from_list(
            atom_strs_proposals_list, all_task_objs)
    else:
        assert isinstance(env, VLMPredicateEnv)
        atom_proposals_set = env.get_vlm_debug_atom_strs

    assert len(atom_proposals_set) > 0, "Atom proposals set is empty!"
    # Given this set of unique atom proposals, we now ask the VLM
    # to label these in every scene from the demonstrations.
    # NOTE: we convert to a sorted list here to get rid of randomness from set
    # ordering.
    unique_atoms_list = sorted(atom_proposals_set)
    # Now, query the VLM!
    logging.info("Querying VLM to label every scene...")
    atom_labels = _label_trajectories_with_atom_values(image_option_trajs,
                                                       gemini_vlm,
                                                       unique_atoms_list)
    logging.info("Done querying VLM for scene labelling!")

    # Save the output as a human-readable txt file.
    save_labelled_trajs_as_txt(
        env, atom_labels, [io_traj.actions for io_traj in image_option_trajs])
    # Now, parse this information into a Dataset!
    # Start by converting all the labelled atoms into a more structured
    # dict. This requires each set of labelled atoms text to be enclosed
    # by curly brackets.
    structured_state_trajs = []
    for atom_traj in atom_labels:
        atoms_txt_strs = [
            '{' + curr_ts_atoms_txt + '}' for curr_ts_atoms_txt in atom_traj
        ]
        full_traj_atoms_str = '\n\n'.join(atoms_txt_strs)
        structured_state_trajs.append(
            utils.parse_atoms_txt_into_structured_state(full_traj_atoms_str))
    # Given this, we now convert each trajectory consisting of a series of
    # structured states into a trajectory of GroundAtoms.
    ground_atoms_trajs = _parse_structured_state_into_ground_atoms(
        env, train_tasks, structured_state_trajs)
    _pretty_print_atoms_trajs(ground_atoms_trajs)
    # Now, we just need to create a goal state for every train task where
    # the dummy goal predicate holds. This is just bookkeeping necessary
    # for NSRT learning and planning such that the goal doesn't hold
    # in the initial state and holds in the final state of each demonstration
    # trajectory.
    goal_states_for_every_traj = _create_dummy_goal_state_for_each_task(
        env, train_tasks)
    # Finally, we need to construct actual LowLevelTrajectories.
    low_level_trajs = _convert_ground_option_trajs_into_lowleveltrajs(
        [traj.actions for traj in image_option_trajs],
        goal_states_for_every_traj, train_tasks)
    return Dataset(low_level_trajs, ground_atoms_trajs)


def create_ground_atom_data_from_labeled_txt(
        env: BaseEnv, train_tasks: List[Task],
        known_options: Set[ParameterizedOption]) -> Dataset:
    """Given a txt file containing trajectories labelled with VLM predicate
    values, construct a dataset that can be passed to the rest of our learning
    pipeline."""
    dataset_fpath = os.path.join(CFG.data_dir, CFG.handmade_demo_filename)
    # First, parse this dataset into a structured form.
    structured_states, structuredactions = utils.\
        parse_vlmtraj_file_into_structured_trajs(dataset_fpath)
    assert len(structured_states) == len(structuredactions)
    # Next, take this intermediate structured form and further
    # parse it into ground atoms and ground options respectively.
    ground_atoms_trajs = _parse_structured_state_into_ground_atoms(
        env, train_tasks, structured_states)
    _pretty_print_atoms_trajs(ground_atoms_trajs)
    option_trajs = _parse_structuredactions_into_ground_options(
        structuredactions, known_options, train_tasks)
    # We need to create the goal state for every train task, just
    # as in the above function.
    goal_states_for_every_traj = _create_dummy_goal_state_for_each_task(
        env, train_tasks)
    # Finally, we need to construct actual LowLevelTrajectories.
    low_level_trajs = _convert_ground_option_trajs_into_lowleveltrajs(
        option_trajs, goal_states_for_every_traj, train_tasks)
    return Dataset(low_level_trajs, ground_atoms_trajs)
