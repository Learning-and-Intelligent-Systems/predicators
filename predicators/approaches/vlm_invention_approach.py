"""
Example command line:
    export OPENAI_API_KEY=<your API key>
"""
import base64
import json
import logging
import re
from typing import List, Set

import imageio
from gym.spaces import Box

from predicators import utils
from predicators.approaches.grammar_search_invention_approach import \
    create_score_function
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.ground_truth_models import get_gt_nsrts
from predicators.llm_interface import OpenAILLM, OpenAILLMNEW
from predicators.settings import CFG
from predicators.structs import AnnotatedPredicate, Dataset, \
    GroundAtomTrajectory, LowLevelTrajectory, Optional, ParameterizedOption, \
    Predicate, Task, Type


class VlmInventionApproach(NSRTLearningApproach):
    """Predicate Invention with VLMs."""

    def __init__(self, initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption], types: Set[Type],
                 action_space: Box, train_tasks: List[Task]) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        # Initial Predicates
        nsrts = get_gt_nsrts(CFG.env, self._initial_predicates,
                             self._initial_options)
        self._nsrts = nsrts

        self._learned_predicates: Set[Predicate] = set()
        self._num_inventions = 0
        # Set up the VLM
        self._vlm = OpenAILLMNEW(CFG.vlm_model_name)
        self._type_dict = {type.name: type for type in self._types}

    @classmethod
    def get_name(cls) -> str:
        return "vlm_invention"

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates

    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        # Generate a candidate set of predicates.
        logging.info("Generating candidate predicates...")
        invention_prompt = self._create_invention_prompt(dataset)
        vlm_predictions = self._vlm.sample_completions(
            invention_prompt, temperature=CFG.llm_temperature,
            seed=CFG.seed)[0]
        candidates = self._parse_predicate_predictions(vlm_predictions)

        for idx, candidate in enumerate(candidates):
            logging.info(f"Predicate: {candidate}")
            interpretation_prompt = self._create_interpretation_prompt(
                candidate, idx)
            # currently the response is pasted from ChatGPT
            # response = self._vlm.sample_completions(interpretation_prompt)
            with open(f'./prompts/interpret_1_cover_{idx}_{candidate.name}'\
                       '.response', 'r') as file:
                response = file.read()
            code = self._parse_classifier_response(response)
            if code != '':
                with open(f'./prompts/interpret_3_cover_{idx}_{candidate.name}'\
                       '.py', 'w') as file:
                    file.write(code)

        logging.info(f"Done: created {len(candidates)} candidates:")
        breakpoint()

        # Apply the candidate predicates to the data.
        logging.info("Applying predicates to data...")
        # Get the template str for the dataset filename for saving
        # a ground atom dataset.
        dataset_fname, _ = utils.create_dataset_filename_str(True)
        # Add a bunch of things relevant to grammar search to the
        # dataset filename string.
        dataset_fname = dataset_fname[:-5] + \
            f"_{CFG.grammar_search_max_predicates}" + \
            f"_{CFG.grammar_search_grammar_includes_givens}" + \
            f"_{CFG.grammar_search_grammar_includes_foralls}" + \
            f"_{CFG.grammar_search_grammar_use_diff_features}" + \
            f"_{CFG.grammar_search_use_handcoded_debug_grammar}" + \
            dataset_fname[-5:]

        # Load pre-saved data if the CFG.load_atoms flag is set.
        atom_dataset: Optional[List[GroundAtomTrajectory]] = None
        if CFG.load_atoms:
            atom_dataset = utils.load_ground_atom_dataset(
                dataset_fname, dataset.trajectories)
        else:
            atom_dataset = utils.create_ground_atom_dataset(
                dataset.trajectories,
                set(candidates) | self._initial_predicates)
            # Save this atoms dataset if the save_atoms flag is set.
            if CFG.save_atoms:
                utils.save_ground_atom_dataset(atom_dataset, dataset_fname)
        logging.info("Done.")

        # Select a subset of the candidates to keep.
        logging.info("Selecting a subset...")
        if CFG.grammar_search_pred_selection_approach == "score_optimization":
            # Create the score function that will be used to guide search.
            score_function = create_score_function(
                CFG.grammar_search_score_function, self._initial_predicates,
                atom_dataset, candidates, self._train_tasks)
            self._learned_predicates = \
                self._select_predicates_by_score_hillclimbing(
                candidates, score_function, self._initial_predicates,
                atom_dataset, self._train_tasks)
        else:
            raise NotImplementedError
        logging.info("Done.")

        # Finally, learn NSRTs via superclass, using all the kept predicates.
        annotations = None
        if dataset.has_annotations:
            annotations = dataset.annotations
        self._learn_nsrts(dataset.trajectories,
                          online_learning_cycle=None,
                          annotations=annotations)

    def _parse_classifier_response(self, response: str) -> str:
        # Define the regex pattern to match Python code block
        pattern = r'```python(.*?)```'

        # Use regex to find the Python code block in the response
        match = re.search(pattern, response, re.DOTALL)

        # If a match is found, return the Python code block
        if match:
            return match.group(1).strip()

        # If no match is found, return an empty string
        return ''

    def _parse_predicate_predictions(self, response: str) -> Set[Predicate]:
        # Regular expression to match the predicate format
        pattern = r"`(.*?)` -- (.*?)\n"
        matches = re.findall(pattern, response)

        # Create a list of AnnotatedPredicate instances
        predicates = []
        for match in matches:
            pred_str = match[0]
            description = match[1]
            name = pred_str.split('(')[0]
            args = pred_str.split('(')[1].replace(')', '').split(', ')
            types = [self._type_dict[arg.split(':')[1]] for arg in args]
            predicate = AnnotatedPredicate(name=name,
                                           types=types,
                                           description=description,
                                           _classifier=None)
            predicates.append(predicate)
        for pred in predicates:
            logging.info(pred)
        return predicates

    def _create_invention_prompt(self, dataset: Dataset) -> str:
        """Compose a prompt for VLM for predicate invention."""
        ########### These doesn't use the dataset ###########
        # Read the template
        with open('./prompts/invent_0_template_raw.prompt', 'r') as file:
            template = file.read()

        # Domain description
        with open('./prompts/domain_description_cover.prompt', 'r') as file:
            domain_description = file.read()
        template = template.replace('[TEXTUAL_DOMAIN_DESCRIPTION]',
                                    domain_description)

        # Domain entity types
        types = []
        for i, type in enumerate(self._types):
            types.append(type.name)
        types_str = ' '.join(types)
        template = template.replace('[DOMAIN_ENTITY_TYPES]', types_str)

        ########### These use the dataset ###########
        # Goals
        goals = []
        # Loop over all tasks in self._train_tasks
        for i, task in enumerate(self._train_tasks):
            # Add the goal of the current task to the goals list
            goal_str = str(task.goal).strip('{}')
            goals.append(f"Task {i}: {goal_str}")
        # Join all goals with a newline character to create a multi-line string
        goals_str = '\n'.join(goals)
        text_prompt = template.replace('[TARGET_OBJECTIVES]', goals_str)

        # Save the text prompt
        with open('./prompts/invent_1_cover_text.prompt', 'w') as file:
            file.write(text_prompt)

        if CFG.rgb_observation:
            # Visual observation
            images = []
            for i, trajectory in enumerate(dataset.trajectories):
                # Get the init observation in the trajectory
                img_save_path = f'./prompts/init_obs_{i}.png'
                observation = trajectory.states[0].rendered_state['scene'][0]
                imageio.imwrite(img_save_path, observation)

                # Encode the image
                image_str = encode_image(img_save_path)
                # Add the image to the images list
                images.append(image_str)

        ########### Make the prompt ###########
        # Create the text entry
        text_entry = {"type": "text", "text": text_prompt}

        prompt = [text_entry]
        if CFG.rgb_observation:
            # Create the image entries
            image_entries = []
            for image_str in images:
                image_entry = {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image_str}"
                    }
                }
                image_entries.append(image_entry)

            # Combine the text entry and image entries and Create the final prompt
            prompt += image_entries

        prompt = [{"role": "user", "content": prompt}]

        # Convert the prompt to JSON string
        prompt_json = json.dumps(prompt, indent=2)
        with open('./prompts/invent_2_cover_final.prompt', 'w') as file:
            file.write(str(prompt_json))
        # Can be loaded with:
        # with open('./prompts/2_invention_cover_final.prompt', 'r') as file:
        #     prompt = json.load(file)
        return prompt

    def _create_interpretation_prompt(self, pred: AnnotatedPredicate,
                                      idx: int) -> str:
        with open('./prompts/interpret_0.prompt', 'r') as file:
            template = file.read()
        text_prompt = template.replace('[INSERT_QUERY_HERE]', pred.__str__())

        # Save the text prompt
        with open(f'./prompts/interpret_1_cover_{idx}_{pred.name}_text.prompt', 'w') \
            as file:
            file.write(text_prompt)

        text_entry = {"type": "text", "text": text_prompt}
        prompt = [{"role": "user", "content": text_entry}]

        # Convert the prompt to JSON string
        prompt_json = json.dumps(prompt, indent=2)
        with open(f'./prompts/interpret_2_cover_{idx}_{pred.name}.prompt', 'w') \
            as file:
            file.write(str(prompt_json))
        return prompt


# Function to encode the image
def encode_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
