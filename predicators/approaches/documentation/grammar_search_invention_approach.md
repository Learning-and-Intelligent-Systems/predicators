# Grammar Search Invention Approach
This approach is primarily useful for inventing predicates via program synthesis from demonstrations, as described originally in:
[Predicate Invention for Bilevel Planning](https://arxiv.org/abs/2203.09634). Silver*, Chitnis*, Kumar, McClinton, Lozano-Perez, Kaelbling, Tenenbaum. AAAI 2023.

An example command for running the approach from that paper is:
```
python predicators/main.py --env cover --approach grammar_search_invention --excluded_predicates all --num_train_tasks 50
```

Last updated: 04/27/2024

## Inventing predicates by leveraging a VLM
We can leverage a VLM to propose concepts that form the basis of the grammar used for predicate invention. This has two advantages: (1) invented predicates operate directly on images, (2) the names of predicates correspond to common-sense concepts.

To do this, we need to supply demonstrations in the form of a sequence of images and labelled options corresponding to the `_Option` that the robot used to get between subsequent states corresponding to subsequent images. 

### Creating datasets for VLM predicate invention
Demonstrations should be saved as a subfolder in the `saved_datasets` folder. The folder should be named `<env_name>__vlm_demos__<seed>_<num_demos>`. For instance, `apple_coring__vlm_demos__456__1`.
Within the folder, there should be 1 subfolder for every demonstration trajectory. So in the above example, there should be exactly 1 subfolder. Name each of these subfolders `traj_<demonstration_number>` with 0-indexing (e.g., `traj_0` for the first demo).
Within each traj subfolder, there should be two things:
1. a subfolder corresponding to each timestep for the demonstration.
2. an `options_traj.txt` file that lists out the series of options executed between each of the states.

The `options_traj.txt` file should contain strings corresponding to the options executed as part of the trajectory. The format for each option should be `<option_name>(<objects>, [<continuous_params>])`.
An example file might look like:
```
pick(apple, [])
place_on(apple, plate, [])
pick(slicing_tool, [])
slice(slicing_tool, apple, hand, [])
```

Given this, a sample folder structure for a demonstration might look like:
apple_coring__vlm_demos__456__2
| traj0
    | 0
        | 0.jpg
    | 1
        | 1.jpg
    | 2
        | 2.jpg
    | 3
        | 3.jpb
    | 4
        | 4.jpg
    | 5
        | 5.jpg
    | options.txt
| traj1
    | 0
        | 0.jpg
    | 1
        | 1.jpg
    | 2
        | 2.jpg
    | 3
        | 3.jpb
    | 4
        | 4.jpg
    | 5
        | 5.jpg
    | options.txt

### Running predicate invention using these image demos
To use the Gemini VLM, you need to set the `GOOGLE_API_KEY` environment variable in your terminal. You can make/get an API key [here](https://aistudio.google.com/app/apikey).

Example command: `python predicators/main.py --env apple_coring --seed 456 --approach grammar_search_invention --excluded_predicates all --num_train_tasks 1 --num_test_tasks 0 --offline_data_method img_demos --vlm_trajs_folder_name apple_coring__vlm_demos__456__1`

The important flags here are the `--offline_data_method img_demos` and the `--vlm_trajs_folder_name apple_coring__vlm_demos__456__1`. The latter should point to the folder housing the demonstration set of interest!

Note that VLM responses are always cached, so if you run the command on a demonstration set and then rerun it, it should be much faster since it's using cached responses!.

Also, the code saves a human-readable txt file to the `saved_datasets` folder that contains a text representation of the GroundAtomTrajectories. You can manually inspect and even edit this file, and then rerun the rest of the predicate invention pipeline starting from this file alone (and not the original demos) as input. Here's an example command that does that:
`python predicators/main.py --env apple_coring --seed 456 --approach grammar_search_invention --excluded_predicates all --num_train_tasks 1 --offline_data_method demo+labeled_atoms --handmade_demo_filename apple_coring__demo+labeled_atoms__manual__1.txt`

where `apple_coring__demo+labeled_atoms__manual__1.txt` is the human-readable txt file.

### Structure of human-readable txt files
We assume the txt files have a particular structure that we leverage for parsing. To explain these components, consider this below example:

```
===
{*Holding(spoon): True.
*Submerged(teabag): False.
*Submerged(spoon): False.} ->

pick(teabag, hand)[] -> 

{*Holding(spoon): True.
*Submerged(teabag): False.
*Submerged(spoon): False.} ->

place_in(teabag, cup)[] -> 

{*Holding(spoon): True.
*Submerged(teabag): False.
*Submerged(spoon): False.} ->

pick(spoon, hand)[] -> 

{*Holding(spoon): True.
*Submerged(teabag): False.
*Submerged(spoon): False.} ->

place_in(spoon, cup)[] -> 

{*Holding(spoon): True.
*Submerged(teabag): False.
*Submerged(spoon): False.}
===
```

**Components**
- Separator: '===' is used to separate one trajectory from another (so a trajectory is sandwiched between two lines that have only '===' on them). In the above example, there is exactly one demonstration trajectory.
- State: Each state is a bulleted list of atoms enclosed between set brackets {}. In the above example, there are 5 states. Note importantly that the format of every atom should be `*<predicate_name>(<ob1_name>, <obj2_name>, ...).`. The `*` at the start, and the period `.` at the end are very important.
- Skill: Each skill is sandwiched between two states and takes the format: `<skill_name>(<ob1_name>, <obj2_name>, ...)[<continuous_param_vector>]`. In the above example, there are 4 skills. Note that after every state, there is a `->` character, followed by a newline, then a skill followed by another `->` character and newline. This is also critical to parsing. Note also that the above example doesn't feature any continuous parameters.


### Future features to be added
* Enable pipeline to consider demonstrations that have low-level object-oriented state, as well as image observations.
* Enable invented VLM predicates to actually be used and run at test-time.
* Consider different VLM's
