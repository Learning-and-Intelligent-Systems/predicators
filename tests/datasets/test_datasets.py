"""Test cases for dataset generation."""
import os
import shutil
from contextlib import nullcontext as does_not_raise

import pytest

from predicators import utils
from predicators.datasets import create_dataset
from predicators.datasets.vlm_input_data import \
    create_ground_atom_data_from_img_trajs
from predicators.envs.blocks import BlocksEnv
from predicators.envs.cluttered_table import ClutteredTableEnv
from predicators.envs.cover import CoverEnv, CoverMultistepOptions
from predicators.envs.vlm_envs import IceTeaMakingEnv
from predicators.ground_truth_models import _get_predicates_by_names, \
    get_gt_options, parse_config_included_options
from predicators.pretrained_model_interface import VisionLanguageModel
from predicators.settings import CFG
from predicators.structs import Dataset, GroundAtom, Task


def test_demo_dataset():
    """Test demo-only dataset creation with Covers env."""
    # Test that data does not contain options since
    # option_learner is not "no_learning"
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "option_learner": "arbitrary_dummy",
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = parse_config_included_options(env)
    dataset = create_dataset(env, train_tasks, options)
    assert len(dataset.trajectories) == 7
    assert len(dataset.trajectories[0].states) == 3
    assert len(dataset.trajectories[0].actions) == 2
    for traj in dataset.trajectories:
        assert traj.is_demo
        for action in traj.actions:
            assert not action.has_option()
    # Test that data contains options since option_learner is "no_learning"
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "option_learner": "no_learning",
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    assert len(dataset.trajectories) == 7
    assert len(dataset.trajectories[0].states) == 3
    assert len(dataset.trajectories[0].actions) == 2
    for traj in dataset.trajectories:
        assert traj.is_demo
        for action in traj.actions:
            assert action.has_option()
    assert not dataset.has_annotations
    with pytest.raises(AssertionError):
        _ = dataset.annotations
    # Test that only options in the included_options flag are included
    utils.reset_config({
        "env": "cover_multistep_options",
        "cover_multistep_bhr_percent": 0.99,
        "cover_multistep_thr_percent": 0.99,
        "approach": "random_actions",
        "offline_data_method": "demo",
        "option_learner": "arbitrary_dummy",
        "num_train_tasks": 3,
        "included_options": "Pick"
    })
    env = CoverMultistepOptions()
    Pick, Place = sorted(get_gt_options(env.get_name()))
    assert Pick.name == "Pick"
    assert Place.name == "Place"
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = parse_config_included_options(env)
    assert options == {Pick}
    dataset = create_dataset(env, train_tasks, options)
    assert len(dataset.trajectories) == 3
    at_least_one_pick_found = False
    at_least_one_place_found = False
    for traj in dataset.trajectories:
        assert traj.is_demo
        for action in traj.actions:
            if action.has_option():
                assert action.get_option().parent == Pick
                at_least_one_pick_found = True
            else:
                at_least_one_place_found = True
    assert at_least_one_pick_found
    assert at_least_one_place_found
    # Test what happens if the goal is unachievable.
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 0.1,
        "option_learner": "no_learning",
        "num_train_tasks": 7,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    init = train_tasks[0].init
    HandEmpty = [pred for pred in env.predicates
                 if pred.name == "HandEmpty"][0]
    Holding = [pred for pred in env.predicates if pred.name == "Holding"][0]
    imposs_goal = {GroundAtom(HandEmpty, []), Holding([list(init)[0]])}
    train_tasks[0] = Task(init, imposs_goal)
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    assert len(dataset.trajectories) < 7
    # Test max_initial_demos.
    utils.reset_config({
        "env": "cover",
        "offline_data_method": "demo",
        "num_train_tasks": 7,
        "max_initial_demos": 3,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    assert len(train_tasks) == 7
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    assert len(dataset.trajectories) == 3
    utils.update_config({
        "offline_data_method": "not a real method",
    })
    with pytest.raises(NotImplementedError):
        create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    utils.update_config({
        "offline_data_method":
        "demo",
        "offline_data_task_planning_heuristic":
        "not a real heuristic",
    })
    with pytest.raises(ValueError):
        create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    # Test demo video generation.
    video_dir = os.path.join(os.path.dirname(__file__), "_fake_videos")
    utils.reset_config({
        "env": "cover",
        "offline_data_method": "demo",
        "num_train_tasks": 1,
        "make_demo_videos": True,
        "cover_num_blocks": 1,
        "cover_num_targets": 1,
        "cover_block_widths": [0.1],
        "cover_target_widths": [0.05],
        "cover_initial_holding_prob": 1.0,
        "video_dir": video_dir,
    })
    video_file = os.path.join(video_dir, "cover__123__demo__task0.mp4")
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    assert len(train_tasks) == 1
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    assert len(dataset.trajectories) == 1
    assert os.path.exists(video_file)
    shutil.rmtree(video_dir)
    # Test demo collection with bilevel_plan_without_sim.
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "num_train_tasks": 5,
        "option_learner": "arbitrary_dummy",
        "bilevel_plan_without_sim": True,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = parse_config_included_options(env)
    dataset = create_dataset(env, train_tasks, options)
    assert 0 < len(dataset.trajectories) < 5
    # Use bilevel planning to collect data, but don't use otherwise.
    utils.reset_config({
        "env": "cover",
        "approach": "nsrt_learning",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "num_train_tasks": 5,
        "option_learner": "arbitrary_dummy",
        "bilevel_plan_without_sim": True,
        "offline_data_bilevel_plan_without_sim": False,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = parse_config_included_options(env)
    dataset = create_dataset(env, train_tasks, options)
    assert len(dataset.trajectories) == 5


@pytest.mark.parametrize(
    "num_train_tasks,load_data,demonstrator,expectation,do_wipe_data_dir",
    [(7, True, "oracle", pytest.raises(ValueError), True),
     (7, False, "oracle", does_not_raise(), False),
     (7, True, "oracle", does_not_raise(), False),
     (7, True, "human", pytest.raises(ValueError), False),
     (3, True, "oracle", does_not_raise(), False),
     (10, True, "oracle", does_not_raise(), False)])
def test_demo_dataset_loading(num_train_tasks, load_data, demonstrator,
                              expectation, do_wipe_data_dir):
    """Test demo-only dataset creation using `--load_data`."""
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo",
        "offline_data_planning_timeout": 500,
        "option_learner": "no_learning",
        "num_train_tasks": num_train_tasks,
        "load_data": load_data,
        "demonstrator": demonstrator,
    })
    if do_wipe_data_dir:
        shutil.rmtree(CFG.data_dir)
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    with expectation as e:
        dataset = create_dataset(env, train_tasks,
                                 get_gt_options(env.get_name()))
    if e is None:
        assert len(dataset.trajectories) == num_train_tasks
        assert all(traj.train_task_idx < len(train_tasks)
                   for traj in dataset.trajectories)
    else:
        assert "Cannot load data" in str(e)


@pytest.mark.parametrize(
    "num_train_tasks,load_data,demonstrator,do_wipe_data_dir",
    [(10, False, "oracle", True), (20, True, "oracle", False),
     (8, True, "oracle", False)])
def test_demo_dataset_loading_tricky_case(num_train_tasks, load_data,
                                          demonstrator, do_wipe_data_dir):
    """Test a tricky case of demo-only dataset creation using `--load_data`."""
    utils.reset_config({
        "env": "blocks",
        "approach": "random_actions",
        "offline_data_method": "demo",
        # Add a strong timeout to make planning fail sometimes.
        "offline_data_planning_timeout": 0.01,
        "option_learner": "no_learning",
        "num_train_tasks": num_train_tasks,
        "load_data": load_data,
        "demonstrator": demonstrator,
    })
    if do_wipe_data_dir:
        shutil.rmtree(CFG.data_dir)
    env = BlocksEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    # Note the use of <= here rather than ==.
    assert len(dataset.trajectories) <= num_train_tasks
    assert all(traj.train_task_idx < len(train_tasks)
               for traj in dataset.trajectories)


def test_demo_replay_dataset():
    """Test demo+replay dataset creation with Covers env."""
    # Test that data contains options since
    # option_learner is "no_learning"
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "option_learner": "no_learning",
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    assert len(dataset.trajectories) == 5 + 3
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1
    num_demos = 0
    for traj in dataset.trajectories:
        num_demos += int(traj.is_demo)
        for action in traj.actions:
            assert action.has_option()
    assert num_demos == 5
    # Test that data does not contain options since
    # option_learner is not "no_learning"
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "option_learner": "arbitrary_dummy",
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = parse_config_included_options(env)
    dataset = create_dataset(env, train_tasks, options)
    assert len(dataset.trajectories) == 5 + 3
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1
    num_demos = 0
    for traj in dataset.trajectories:
        num_demos += int(traj.is_demo)
        for action in traj.actions:
            assert not action.has_option()
    assert num_demos == 5
    # Test that only options in the included_options flag are included
    utils.reset_config({
        "env": "cover_multistep_options",
        "cover_multistep_bhr_percent": 0.99,
        "cover_multistep_thr_percent": 0.99,
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "option_learner": "arbitrary_dummy",
        "num_train_tasks": 3,
        "included_options": "Pick"
    })
    env = CoverMultistepOptions()
    Pick, Place = sorted(get_gt_options(env.get_name()))
    assert Pick.name == "Pick"
    assert Place.name == "Place"
    train_tasks = [t.task for t in env.get_train_tasks()]
    options = parse_config_included_options(env)
    assert options == {Pick}
    dataset = create_dataset(env, train_tasks, options)
    assert len(dataset.trajectories) == 3 + 3
    at_least_one_pick_found = False
    at_least_one_place_found = False
    for traj in dataset.trajectories:
        for action in traj.actions:
            if action.has_option():
                assert action.get_option().parent == Pick
                at_least_one_pick_found = True
            else:
                at_least_one_place_found = True
    assert at_least_one_pick_found
    assert at_least_one_place_found
    # Test cluttered table data collection
    utils.reset_config({
        "env": "cluttered_table",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 10,
        "num_train_tasks": 5,
    })
    env = ClutteredTableEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    assert len(dataset.trajectories[-1].states) == 2
    assert len(dataset.trajectories[-1].actions) == 1


def test_dataset_with_annotations():
    """Test the creation of a Dataset with annotations."""
    utils.reset_config({
        "env": "cover",
        "approach": "random_actions",
        "offline_data_method": "demo+replay",
        "offline_data_planning_timeout": 500,
        "offline_data_num_replays": 3,
        "option_learner": "no_learning",
        "num_train_tasks": 5,
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    trajectories = create_dataset(env, train_tasks,
                                  get_gt_options(env.get_name())).trajectories
    # The annotations and trajectories need to be the same length.
    with pytest.raises(AssertionError):
        dataset = Dataset(trajectories, [])
    annotations = ["label" for _ in trajectories]
    dataset = Dataset(list(trajectories), list(annotations))
    assert dataset.has_annotations
    assert dataset.annotations == annotations
    # Can't add a data point without an annotation.
    with pytest.raises(AssertionError):
        dataset.append(trajectories)
    dataset.append(trajectories[0], annotations[0])
    assert dataset.has_annotations
    assert len(dataset.trajectories) == len(dataset.annotations) == \
        len(trajectories) + 1


def test_ground_atom_dataset():
    """Test creation of demo+ground_atoms dataset."""
    utils.reset_config({
        "env": "cover",
        "approach": "interactive_learning",
        "num_train_tasks": 15,
        "offline_data_method": "demo+ground_atoms",
        "teacher_dataset_num_examples": 1,
        "excluded_predicates": "Holding,Covers",
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    assert len(dataset.trajectories) == 15
    assert len(dataset.annotations) == 15
    Covers, HandEmpty, Holding = _get_predicates_by_names(
        "cover", ["Covers", "HandEmpty", "Holding"])
    all_predicates = {Covers, HandEmpty, Holding}
    # Test that the right number of atoms are annotated.
    pred_name_to_counts = {p.name: [0, 0] for p in all_predicates}
    for traj, ground_atoms_seq in zip(dataset.trajectories,
                                      dataset.annotations):
        assert len(traj.states) == len(ground_atoms_seq)
        for ground_atom_sets, s in zip(ground_atoms_seq, traj.states):
            assert len(ground_atom_sets
                       ) == 2, "Should be two sets of ground atoms per state"
            all_ground_atoms = utils.abstract(s, all_predicates)
            all_ground_atom_names = set()
            for ground_truth_atom in all_ground_atoms:
                all_ground_atom_names.add((ground_truth_atom.predicate.name,
                                           tuple(ground_truth_atom.objects)))
            for label, ground_atoms in enumerate(ground_atom_sets):
                for annotated_atom in ground_atoms:
                    pred_name_to_counts[
                        annotated_atom.predicate.name][label] += 1
                    # Make sure the annotations are correct.
                    annotated_atom_name = (annotated_atom.predicate.name,
                                           tuple(annotated_atom.objects))
                    if label:
                        assert annotated_atom_name in all_ground_atom_names
                    else:
                        assert annotated_atom_name not in all_ground_atom_names
                    # Make sure we're not leaking information.
                    with pytest.raises(Exception) as e:
                        annotated_atom.holds(s)
                    assert "Stripped classifier should never" in str(e)
    # HandEmpty was included, so no annotations.
    assert pred_name_to_counts["HandEmpty"] == [0, 0]
    # Holding and Covers were excluded.
    target_num = CFG.teacher_dataset_num_examples
    for name in ["Holding", "Covers"]:
        assert pred_name_to_counts[name] == [target_num, target_num]
    # Test error when not enough examples to sample from
    utils.reset_config({
        "env": "cover",
        "approach": "interactive_learning",
        "num_train_tasks": 15,
        "offline_data_method": "demo+ground_atoms",
        "teacher_dataset_num_examples": 100,
        "excluded_predicates": "Holding,Covers",
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    with pytest.raises(ValueError):
        create_dataset(env, train_tasks, get_gt_options(env.get_name()))


def test_empty_dataset():
    """Test creation of empty dataset."""
    utils.reset_config({
        "env": "cover",
        "offline_data_method": "empty",
    })
    env = CoverEnv()
    train_tasks = [t.task for t in env.get_train_tasks()]
    dataset = create_dataset(env, train_tasks, get_gt_options(env.get_name()))
    assert len(dataset.trajectories) == 0
    with pytest.raises(AssertionError):
        _ = dataset.annotations


@pytest.mark.parametrize(
    "atom_proposal_prompt_type, atom_labeling_prompt_type",
    [("naive_each_step", "per_scene_naive"),
     ("options_labels_whole_traj", "per_scene_naive"),
     ("naive_whole_traj", "per_scene_cot"),
     ("not_a_real_prompt_type", "per_scene_cot"),
     ("naive_whole_traj", "not_a_real_prompt_type")])
def test_loading_img_demos(atom_proposal_prompt_type,
                           atom_labeling_prompt_type):
    """Test loading a dataset from a txt file.

    NOTE: if you're having issues with this test locally, delete the
    pretrained_model_cache just to make sure previous incorrect results
    aren't being cached.
    """

    class _DummyVLM(VisionLanguageModel):

        def get_id(self):
            return "dummy"

        def _sample_completions(self,
                                prompt,
                                imgs,
                                temperature,
                                seed,
                                stop_token=None,
                                num_completions=1):
            del imgs  # unused.
            completions = []
            for _ in range(num_completions):
                # If the query is asking for atom proposals.
                if "Please provide predicates" in prompt:
                    completion = "*Holding(spoon)\n*Fizz(buzz)\n" + \
                        "Submerged(teabag)\nSubmerged(spoon)"
                # Else, if the query is asking for particular values.
                elif "values of the following predicates" in prompt:
                    completion = "*Holding(spoon): True.\n" + \
                        "*Submerged(teabag): False.\n*Submerged(spoon): False."
                completions.append(completion)
            return completions

    utils.reset_config({
        "env":
        "iced_tea_making",
        "num_train_tasks":
        1,
        "offline_data_method":
        "img_demos",
        "data_dir":
        "tests/datasets/mock_vlm_datasets",
        "seed":
        456,
        "vlm_trajs_folder_name":
        "iced_tea_making__vlm_demos__456__1",
        "grammar_search_vlm_atom_proposal_prompt_type":
        atom_proposal_prompt_type,
        "grammar_search_vlm_atom_label_prompt_type":
        atom_labeling_prompt_type
    })
    env = IceTeaMakingEnv()
    train_tasks = env.get_train_tasks()
    vlm = _DummyVLM()
    if atom_proposal_prompt_type != "not_a_real_prompt_type" and \
        atom_labeling_prompt_type != "not_a_real_prompt_type":
        loaded_dataset = create_ground_atom_data_from_img_trajs(
            env, train_tasks, get_gt_options(env.get_name()), vlm)
        assert len(loaded_dataset.trajectories) == 1
        assert len(loaded_dataset.annotations) == 1
        assert len(loaded_dataset.annotations[0][0]) == 1
        assert "Holding(spoon:spoon)" in str(loaded_dataset.annotations[0][0])
        assert "DummyGoal" in str(loaded_dataset.annotations[0][-1])
    else:
        with pytest.raises(ValueError) as e:
            loaded_dataset = create_ground_atom_data_from_img_trajs(
                env, train_tasks, get_gt_options(env.get_name()), vlm)
        assert "Unknown" in str(e)


def test_loading_txt_files():
    """Test loading a dataset from a txt file."""
    utils.reset_config({
        "env":
        "ice_tea_making",
        "num_train_tasks":
        1,
        "offline_data_method":
        "demo+labeled_atoms",
        "data_dir":
        "tests/datasets/mock_vlm_datasets",
        "handmade_demo_filename":
        "ice_tea_making__demo+labeled_atoms__manual__1.txt"
    })
    env = IceTeaMakingEnv()
    train_tasks = env.get_train_tasks()
    loaded_dataset = create_dataset(env, train_tasks,
                                    get_gt_options(env.get_name()))
    assert len(loaded_dataset.trajectories) == 1
