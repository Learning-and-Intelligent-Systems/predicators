"""Functions to create offline demonstration data by leveraging VLMs."""

import ast
import glob
import os
import re
from pathlib import Path
from typing import List, Sequence, Set

import numpy as np
import PIL.Image

from predicators import utils
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Action, Dataset, GroundAtom, \
    ImageOptionTrajectory, LowLevelTrajectory, Object, ParameterizedOption, \
    Predicate, State, Task, _Option
from predicators.vlm_interface import GoogleGeminiVLM, VisionLanguageModel


def sample_init_atoms_from_trajectories(
        trajectories: List[ImageOptionTrajectory],
        vlm: VisionLanguageModel,
        trajectory_subsample_freq=1) -> List[str]:
    """Given a list of ImageOptionTrajectories, query a VLM to generate a list
    of names of ground atoms from which we can extract predicates that might be
    relevant for planning to recreate these trajectories."""

    aggregated_vlm_output_strs = []
    for traj in trajectories:
        prompt = (
            "You are a robotic vision system who's job is to output a "
            "structured set of predicates useful for running a task and motion "
            "planning system from the following scene. Please provide predicates "
            f"in terms of the following objects: {[str(obj.name) for obj in traj._objects if obj.name != 'dummy_goal_obj']}. "
            "For each predicate, output it in the following format: "
            "predicate_name(obj1, obj2, obj3...) "
            "(for instance is_sliced(apple), is_not_sliced(apple), etc.). "
            "Also, for each predicate you list, also list its negation. "
            "List as many predicates as you can possibly think of, even if they're only "
            "tangentially relevant to what you see in the scene and even if they're false, "
            "given the following scene taken from a demonstration for the task. "
        )
        i = 0
        while i < len(traj._state_imgs):
            # Get outputs with temperature 1.
            curr_vlm_output_atoms_str = vlm.sample_completions(
                prompt, traj._state_imgs[i], 1.0, CFG.seed)
            aggregated_vlm_output_strs.append(curr_vlm_output_atoms_str)
            i += trajectory_subsample_freq

    import ipdb; ipdb.set_trace()


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
    # TODO: might want to make sure this loop actually accesses the
    # trajectory folders in ascending order.
    for train_task_idx, path in enumerate(
            Path(trajectories_folder_path).iterdir()):
        assert path.is_dir()
        state_folders = [f.path for f in os.scandir(path) if f.is_dir()]
        num_states_in_traj = len(state_folders)
        state_traj = []
        for state_num in range(num_states_in_traj):
            curr_state_imgs = []
            curr_state_path = path.joinpath(str(state_num))
            # NOTE: we assume all images are saved as jpg files.
            img_files = glob.glob(str(curr_state_path) + "/*.jpg")
            for img in img_files:
                curr_state_imgs.append(PIL.Image.open(img))
            state_traj.append(curr_state_imgs)
        # Get objects from train tasks to be used for future parsing.
        curr_train_task = train_tasks[train_task_idx]
        curr_task_objs = set(curr_train_task.init)
        curr_task_obj_name_to_obj = {obj.name: obj for obj in curr_task_objs}
        # Parse out actions for the trajectory.
        options_traj_file_list = glob.glob(str(path) + "/*.txt")
        assert len(options_traj_file_list) == 1
        options_traj_file = options_traj_file_list[0]
        with open(options_traj_file, "r") as f:
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
            ground_option_traj.append(ground_option)
        # Given ground options, we can finally make ImageOptionTrajectories.
        image_option_trajs.append(
            ImageOptionTrajectory(list(curr_task_objs), state_traj,
                                  ground_option_traj, True, train_task_idx))
    # Given trajectories, we can now query the VLM.
    gemini_vlm = GoogleGeminiVLM("gemini-pro-vision")
    sample_init_atoms_from_trajectories(image_option_trajs, gemini_vlm, 1)


def create_ground_atom_data_from_labeled_txt(
        env: BaseEnv, train_tasks: List[Task],
        known_options: Set[ParameterizedOption]) -> Dataset:
    """Given a txt file containing trajectories labelled with VLM predicate
    values, construct a dataset that can be passed to the rest of our learning
    pipeline."""
    dataset_fpath = os.path.join(CFG.data_dir, CFG.handmade_demo_filename)
    # First, parse this dataset into a structured form.
    structured_states, structured_actions = utils.parse_handmade_vlmtraj_file_into_structured_trajs(
        dataset_fpath)
    assert len(structured_states) == len(structured_actions)

    # Now, parse out and make all necessary predicates.
    # Also, start making a list of all the names of different objects.
    # For now, assume that there is only one type called "object",
    # and it has only one attribute (that the environment uses
    # to check whether the goal has been achieved).
    assert len(env.types) == 1
    type_name_to_type = {t.name: t for t in env.types}
    assert "object" in type_name_to_type
    assert type_name_to_type["object"].dim == 1

    def _stripped_classifier(state: State, objects: Sequence[Object]) -> bool:
        raise Exception("Stripped classifier should never be called!")

    # HACK! Create a bunch of necessary, hardcoded stuff
    # to make the initial states and goals line up with the training
    # tasks and domain.
    objs = list(set(train_tasks[0].init.data))
    dummy_obj = objs[0]
    assert len(env.goal_predicates) == 1
    goal_preds = list(env.goal_predicates)
    goal_atom = GroundAtom(goal_preds[0], [dummy_obj])

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
                            pred_name, [
                                type_name_to_type["object"]
                                for _ in range(len(obj_args))
                            ], _stripped_classifier)
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
                            GroundAtom(pred_name_to_pred[pred_name],
                                       [obj_name_to_obj[o] for o in obj_args]))
            curr_atoms_traj.append(curr_ground_atoms_state)
        # Add the goal atom at the end of the trajectory.
        curr_atoms_traj[-1].add(goal_atom)
        atoms_trajs.append(curr_atoms_traj)

    # HACK! Create a bunch of necessary, hardcoded stuff
    # to make the initial states and goals line up with the training
    # tasks and domain.
    state_dict = {}
    for obj in obj_name_to_obj.values():
        state_dict[obj] = [0.0]
    state_dict.update(train_tasks[0].init.data)
    default_state = State(state_dict)
    objs = list(set(train_tasks[0].init.data))
    dummy_obj = objs[0]
    # NOTE: Given the assumption that the goal predicate is a simple
    # classifier on the dummy_object from the initial state, we need
    # to set the goal to be true in the final state.
    final_state = default_state.copy()
    final_state.set(dummy_obj, "goal_true", 1.0)
    # NOTE: We also add in the dummy goal predicate here.
    assert len(env.goal_predicates) == 1
    goal_preds = list(env.goal_predicates)
    goal_atom = GroundAtom(goal_preds[0], [dummy_obj])

    # Construct a default state that just contains all the objects in the
    # domain with no attributes. Also include any any objects that are
    # in the initial state of the train tasks.
    # NOTE: For now, assume that all train tasks have the same initial
    # state.
    for task in train_tasks:
        assert len(task.init.data) == 1
        for obj in set(task.init.data):
            assert obj.name == "dummy_goal_obj"
        assert len(task.goal) == 1
        for atom in task.goal:
            assert "DummyGoal" in atom.predicate.name

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
                obj_name_to_obj[obj_name] for obj_name in structured_action[1]
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
                Action(np.zeros(0), option_trajs[traj_num][idx_within_traj]))
        # Now, we need to append the final state because there are 1 more
        # states than actions.
        curr_traj_states.append(final_state)
        curr_traj = LowLevelTrajectory(curr_traj_states, curr_traj_actions,
                                       True, traj_num)
        trajs.append(curr_traj)

    # Finally, package everything together into a Dataset!
    return Dataset(trajs, atoms_trajs)
