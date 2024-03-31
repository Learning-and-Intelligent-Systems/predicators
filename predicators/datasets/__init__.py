"""Create offline datasets for training, given a set of training tasks for an
environment."""

from typing import List, Set, Sequence
import os
import numpy as np

from predicators import utils
from predicators.datasets.demo_only import create_demo_data
from predicators.datasets.demo_replay import create_demo_replay_data
from predicators.datasets.ground_atom_data import create_ground_atom_data
from predicators.envs import BaseEnv
from predicators.settings import CFG
from predicators.structs import Dataset, ParameterizedOption, Task, DefaultState, Action, Type, Predicate, State, Object


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
        dataset_fpath = os.path.join(
            CFG.data_dir,
            CFG.handmade_demo_filename)
        # First, parse this dataset into a structured form.
        structured_states, structured_actions = utils.parse_handmade_vlmtraj_file_into_structured_trajs(dataset_fpath)
        assert len(structured_states) == len(structured_actions)
        # Now, parse out and make all necessary predicates.

        def _stripped_classifier(state: State, objects: Sequence[Object]) -> bool:
            raise Exception("Stripped classifier should never be called!")

        # NOTE: for now, the predicates aren't typed, but rather all predicates
        # consumer things of type "default_type".
        default_type = Type("default_type", [])
        pred_name_to_pred = {}
        for traj in structured_states:
            for structured_state in traj:
                for pred_name, objs_and_val_dict in structured_state.items():
                    # IMPORTANT: this currently assumes that the data is such
                    # that a predicate with a certain name (e.g. "Sliced")
                    # always appears with the same number of object arguments
                    # (e.g. Sliced(apple), and never 
                    # Sliced(apple, cutting_tool)). We might want to explicitly
                    # check for this in the future.
                    if pred_name not in pred_name_to_pred:
                        assert len(objs_and_val_dict.keys()) == 1
                        for objs in objs_and_val_dict.keys():
                            pred_name_to_pred[pred_name] = Predicate(pred_name, [default_type for i in range(len(objs))], _stripped_classifier)
                            

        # Next, we link option names to actual options.
        option_name_to_option = {o.name: o for o in known_options}
        for traj in structured_actions:
            for structured_action in traj:
                print(structured_action)
        import ipdb; ipdb.set_trace()

        # Next, turn these structures into a Dataset.
        trajs = []
        for traj_num in range(len(structured_actions)):
            curr_traj_states = []
            curr_traj_actions = []
            for idx_within_traj in range(len(structured_actions[traj_num])):
                curr_traj_states.append(DefaultState)
                # TODO: include the correct option here after parsing options
                # above!
                curr_traj_actions.append(Action(np.zeros(0)))



        import ipdb; ipdb.set_trace()
    
    if CFG.offline_data_method == "empty":
        return Dataset([])
    raise NotImplementedError("Unrecognized dataset method.")
