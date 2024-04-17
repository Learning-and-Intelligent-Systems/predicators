"""
Example command line:
    export OPENAI_API_KEY=<your API key>
"""
import re
import ast
import json
import base64
import logging
from typing import Set, List, Dict, Sequence, Tuple, Any
import subprocess
import inspect
from inspect import getsource
from copy import deepcopy
import importlib.util

from gym.spaces import Box
import numpy as np
import imageio

from predicators import utils
from predicators.settings import CFG
from predicators.ground_truth_models import get_gt_nsrts
from predicators.llm_interface import OpenAILLM, OpenAILLMNEW
from predicators.approaches import ApproachFailure, ApproachTimeout
from predicators.approaches.nsrt_learning_approach import NSRTLearningApproach
from predicators.structs import Dataset, LowLevelTrajectory, Predicate, \
    ParameterizedOption, Type, Task, Optional, GroundAtomTrajectory, \
    AnnotatedPredicate, State, Object, _TypedEntity
from predicators.approaches.grammar_search_invention_approach import \
    create_score_function 
from predicators.envs import BaseEnv
from predicators.envs.stick_button import StickButtonEnv
from predicators.utils import option_plan_to_policy, OptionExecutionFailure, \
    EnvironmentFailure

from predicators.envs.stick_button import StickButtonEnv

import_str = """
from typing import Sequence
import numpy as np
from predicators.structs import State, Object, Predicate, Type
from predicators.envs.stick_button import StickButtonEnv
        
_hand_type = StickButtonEnv._hand_type
_button_type = StickButtonEnv._button_type
_stick_type = StickButtonEnv._stick_type
_holder_type = StickButtonEnv._holder_type
"""

class VlmInventionApproach(NSRTLearningApproach):
    """Predicate Invention with VLMs"""
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
        return "vlm_online_invention"

    def _get_current_predicates(self) -> Set[Predicate]:
        return self._initial_predicates | self._learned_predicates
    
    def load(self, online_learning_cycle: Optional[int]) -> None:
        super().load(online_learning_cycle)

        preds, _ = utils.extract_preds_and_types(self._nsrts)
        self._learned_predicates = (set(preds.values()) -
                                    self._initial_predicates)

    def _solve_tasks(self, tasks: List[Task]) -> List[Any]:
        '''Solve a dataset of tasks
        '''
        results = []
        for i, task in enumerate(tasks):
            logging.info(f"Solving Task {i}")
            # the longest refinements and the error
            # print(f"task {i}, objects {list(task.init)}")
            try:
                policy = self.solve(task, timeout=CFG.timeout) 
            except (ApproachTimeout, ApproachFailure) as e:
                logging.info(f"--> Failed: {str(e)}")
                result = e
            else:
                logging.info("--> Succeeded")
                result = {
                        "option_plan": self._last_plan,
                        "nsrt_plan": self._last_nsrt_plan,
                        "policy": policy}
                # print(result.info['partial_refinements'][0][0])
            results.append(result)
        return results
    
    def _collect_experience(self, env: BaseEnv, tasks:List[Task]) -> Dataset:
        '''Collect experience from solving tasks
        '''
        trajectories = []
        for idx, task in enumerate(tasks):
            logging.info(f"Solving Task {i}")
            try:
                policy = self.solve(task, timeout=CFG.timeout)
            except (ApproachTimeout, ApproachFailure) as e:
                logging.info(f"--> Failed: {str(e)}")
                continue
            else:
                logging.info(f"--> Succeeded")
                policy = utils.option_plan_to_policy(self._last_plan)
                traj, _ = utils.run_policy(
                    policy,
                    env,
                    "train",
                    idx,
                    termination_function=lambda s: False,
                    max_num_steps=CFG.horizon,
                    exceptions_to_break_on={
                        utils.OptionExecutionFailure,
                    },
                )
                traj = LowLevelTrajectory(traj.states,
                                        traj.actions,
                                        _is_demo=True,
                                        _train_task_idx=idx)
                trajectories.append(traj)

        dataset = Dataset(trajectories)
        return dataset


    def learn_from_offline_dataset(self, dataset: Dataset) -> None:
        pass

    def learn_from_tasks(self, env: BaseEnv, tasks: List[Task]) -> None:
        '''Learn from interacting with the offline dataset
        '''
        tasks_for_prompt = tasks[:CFG.num_tasks_for_prompt]
        # Ask it to try to solve the tasks
        # results = self._solve_tasks(tasks[:tasks_for_prompt])
        # # Use the results to prompt the llm
        # invention_prompt = self._create_invention_prompt(env, results, tasks)
        
        manual_prompt = True
        response_file = './prompts/online_invent_1.response'

        # if manual_prompt:
        #     # create a empty txt file for pasting chatGPT response
        #     with open(response_file, 'w') as file:
        #         file.write('')
        # else:
        #     vlm_predictions = self._vlm.sample_completions(invention_prompt,
        #                                         temperature=CFG.llm_temperature,
        #                                         seed=CFG.seed,
        #                                         save_file=response_file)[0]

        candidates = self._parse_predicate_predictions(response_file)
        logging.info(f"Done: created {len(candidates)} candidates:")

        # Use the predicates to do another round of planning
        self._learned_predicates = candidates

        # # Apply the candidate predicates to the data.
        # logging.info("Applying predicates to data...")
        # # Get the template str for the dataset filename for saving
        # # a ground atom dataset.
        # dataset_fname, _ = utils.create_dataset_filename_str(True)
        # # Add a bunch of things relevant to grammar search to the
        # # dataset filename string.
        # dataset_fname = dataset_fname[:-5] + \
        #     f"_{CFG.grammar_search_max_predicates}" + \
        #     f"_{CFG.grammar_search_grammar_includes_givens}" + \
        #     f"_{CFG.grammar_search_grammar_includes_foralls}" + \
        #     f"_{CFG.grammar_search_grammar_use_diff_features}" + \
        #     f"_{CFG.grammar_search_use_handcoded_debug_grammar}" + \
        #     dataset_fname[-5:]

        # # Load pre-saved data if the CFG.load_atoms flag is set.
        # atom_dataset: Optional[List[GroundAtomTrajectory]] = None
        # if CFG.load_atoms:
        #     atom_dataset = utils.load_ground_atom_dataset(
        #         dataset_fname, dataset.trajectories)
        # else:
        #     atom_dataset = utils.create_ground_atom_dataset(
        #         dataset.trajectories,
        #         set(candidates) | self._initial_predicates)
        #     # Save this atoms dataset if the save_atoms flag is set.
        #     if CFG.save_atoms:
        #         utils.save_ground_atom_dataset(atom_dataset, dataset_fname)
        # logging.info("Done.")

        # # Select a subset of the candidates to keep.
        # logging.info("Selecting a subset...")
        # if CFG.grammar_search_pred_selection_approach == "score_optimization":
        #     # Create the score function that will be used to guide search.
        #     score_function = create_score_function(
        #         CFG.grammar_search_score_function, self._initial_predicates,
        #         atom_dataset, candidates, self._train_tasks)
        #     self._learned_predicates = \
        #         self._select_predicates_by_score_hillclimbing(
        #         candidates, score_function, self._initial_predicates,
        #         atom_dataset, self._train_tasks)
        # else:
        #     raise NotImplementedError
        # logging.info("Done.")

        # Finally, learn NSRTs via superclass, using all the kept predicates.
        dataset = self._collect_experience(env, tasks)
        annotations = None
        if dataset.has_annotations:
            annotations = dataset.annotations
        self._learn_nsrts(dataset.trajectories,
                          online_learning_cycle=None,
                          annotations=annotations)
        
    def _parse_predicate_predictions(self, prediction_file: str
                                     ) -> Set[Predicate]:
        # Read the prediction file
        with open(prediction_file, 'r') as file:
            response = file.read()

        # Regular expression to match Python code blocks
        pattern = re.compile(r'```python(.*?)```', re.DOTALL)
        python_blocks = []
        # Find all Python code blocks in the text
        for match in pattern.finditer(response):
            # Extract the Python code block and add it to the list
            python_blocks.append(match.group(1).strip())
        
        candidates = set()
        for code_str in python_blocks:
            # Extract name from code block
            match = re.search(r'(\w+)\s*=\s*Predicate', code_str)
            if match:
                pred_name =  match.group(1)
            else:
                raise ValueError("No predicate name found in the code block")
            logging.info(f"Found definition for predicate {pred_name}")
            
            # # Type check the code
            # passed = False
            # while not passed:
            #     result, passed = self.type_check_proposed_predicates(pred_name, 
            #                                                          code_str)
            #     if not passed:
            #         # Ask the LLM or the User to fix the code
            #         pass
            #     else:
            #         break

            # Instantiate the predicate
            context = {}
            exec(import_str + '\n' + code_str, context)
            candidates.add(context[pred_name])

        return candidates

    def type_check_proposed_predicates(self,
                                       predicate_name: str,
                                       code_block: str) -> Tuple[str, bool]:
        # Write the definition to a python file
        predicate_fname = f'./prompts/oi1_predicate_{predicate_name}.py'
        with open(predicate_fname, 'w') as f:
            f.write(import_str + '\n' + code_block)

        # Type check
        logging.info(f"Start type checking the predicate "+
                        f"{predicate_name}...")
        result = subprocess.run(["mypy", 
                                    "--strict-equality", 
                                    "--disallow-untyped-calls", 
                                    "--warn-unreachable",
                                    "--disallow-incomplete-defs",
                                    "--show-error-codes",
                                    "--show-column-numbers",
                                    "--show-error-context",
                                predicate_fname], 
                                capture_output=True, text=True)
        stdout = result.stdout
        passed = result.returncode == 0
        return stdout, passed
        
    def _create_invention_prompt(self, env: BaseEnv, 
                                 results: Dict=None, 
                                 tasks: Task=None) -> str:
        '''Compose a prompt for VLM for predicate invention
        '''
        ########### These doesn't use the dataset ###########
        # Read the template
        with open('./prompts/online_invent_0_template.prompt', 'r') as file:
            template = file.read()

        def add_python_quote(text: str) -> str:
            return f"```python\n{text}\n```\n"

        def d2s(dict_with_arrays):
            # Convert State data with numpy arrays to lists, and to string
            return str({k: [round(i, 2) for i in v.tolist()] for k, v in 
                    dict_with_arrays.items()})

        ##### Meta Environment
        # Predicate Class
        predicate_class_str = getsource(Predicate)
        delimitor = "return self._classifier(state, objects)"
        keep_index = predicate_class_str.find(delimitor)
        keep_index += len(delimitor) + 1
        template = template.replace('[PREDICATE_CLASS_DEFINITION]', 
                    add_python_quote(predicate_class_str[:keep_index]))

        # State Class
        state_class_str = getsource(State)
        template = template.replace('[STATE_CLASS_DEFININTION]',
                                    add_python_quote(state_class_str))

        # Object Class
        type_str = getsource(Type)
        typed_entity_str = getsource(_TypedEntity)
        object_class_str = getsource(Object)
        template = template.replace('[OBJECT_CLASS_DEFINITION]',
                        add_python_quote(type_str + "\n" +
                                         typed_entity_str +  "\n" +
                                         object_class_str))

        ##### Environment
        # Type Initialization
        source_code = getsource(StickButtonEnv)
        type_pattern = r"(    # Types.*?)(?=\n\s*\n|$)"        
        type_block = re.search(type_pattern, source_code, re.DOTALL)
        type_init_str = type_block.group()
        template = template.replace("[TYPES_IN_ENV]", 
                                    add_python_quote(type_init_str))

        # Predicates
        predicate_pattern = r"(# Predicates.*?)(?=\n\s*\n|$)"        
        predicate_block = re.search(predicate_pattern, source_code, re.DOTALL)
        predicate_init_str = predicate_block.group()
        # Predicates with classifiers
        initial_predicate_str = []
        initial_predicate_str.append(str(self._initial_predicates) + "\n")

        for p in self._initial_predicates:
            p_name = p.name
            # Init code
            p_init_pattern = r"(self\._" + re.escape(p_name) +\
                      r" = Predicate\(.*?\n.*?\))"
            block = re.search(p_init_pattern, predicate_init_str, re.DOTALL)
            p_init_str = block.group()
            # Predicate description with classifiers
            initial_predicate_str.append(
                f"Predicate `{p_name}` is created by\n" + 
                add_python_quote(p_init_str) +
                "This creates Predicate " +
                p.predicate_str())
        initial_predicate_str = '\n'.join(initial_predicate_str)
        template = template.replace("[PREDICATES_IN_ENV]",
                                    initial_predicate_str)
        
        # NSRTS
        nsrt_str = []
        for nsrt in self._nsrts:
            nsrt_str.append(str(nsrt))
        template = template.replace("[NSRTS_IN_ENV]", '\n'.join(nsrt_str))

        ##### Task Information
        # Set the print options
        np.set_printoptions(precision=1)

        task_str = []
        for i, t in enumerate(tasks):
            # Task specifications
            task_str.append(f"Task {i}:")
            task_str.append("Initial State with objects and feature names: \n" + 
                            t.init.pretty_str() )
            task_str.append("Initial Abstract State: " + str(utils.abstract(
                t.init, self._initial_predicates)) + '\n')
            task_str.append("Task goal: " + str(t.goal) + '\n')

            # Planning Results
            # Planning succeeded
            result = results[i]
            if isinstance(result, dict):
                print(f"Task {i}: planning succeeded.")
                nsrt_plan = result['nsrt_plan']
                option_plan = result['option_plan'].copy()
                # policy = utils.option_plan_to_policy(result['option_plan'])
                task_str.append("The bilevel planning succeeded.\n")

                # High level plan
                task_str.append("The high-level plan is:")
                high_level_plan = []
                for i, nsrt in enumerate(nsrt_plan):
                    high_level_plan.append(f"Step {i}: " + str(nsrt))
                task_str.append('\n'.join(high_level_plan))
                task_str.append("Done.\n")

                # Executing the plan
                state = env.reset(train_or_test='train', task_idx=i)
                task_str.append("The state-action trajectory is:")
                task_str.append("State: " + d2s(state.data) + "\n")
                def policy(_): raise OptionExecutionFailure("")
                nsrt_counter = 0
                for _ in range(CFG.horizon):
                    try:
                        act = policy(state)
                        task_str.append("Action: " + str(act._arr) + "\n")
                        state = env.step(act)
                        task_str.append("State: " + d2s(state.data) + "\n")
                    except OptionExecutionFailure as e:
                        # When the one-option policy reaches terminal state
                        try:
                            policy = utils.option_plan_to_policy(
                                    [option_plan.pop(0)])
                        except IndexError:
                            # When the option plan is empty
                            task_str.append(f"Abstract State: " + 
                                            str(utils.abstract(state, 
                                                self._initial_predicates)) +
                                            "\n")
                            break
                        else:
                            task_str.append(f"Abstract State: " + 
                                            str(utils.abstract(state, 
                                                   self._initial_predicates)) +
                                            "\n")
                            task_str.append(
                                f"Start executing Step {nsrt_counter}: "+
                                    f"{str(nsrt_plan[nsrt_counter])}\n")
                            nsrt_counter += 1
            # Planning failed
            else:
                print(f"Task {i}: planning failed.")
                # Take the first task plan
                # len(nsrt_plan) >= len(longest_refinement)
                nsrt_plan = result.info['partial_refinements'][0][0]
                longest_refinement = result.info['partial_refinements'][0][1]
                longest_refinement_len = len(longest_refinement)

                # REASON_OF_BILEVEL_PLANNING_FAILURE
                task_str.append("The bilevel planning failed because: " +
                                str(result) + "\n")
                # Shortest_task_plan
                task_str.append("The first task plan (skeleton) the planner "+
                                "found is: ")
                for j, nsrt in enumerate(nsrt_plan):
                    task_str.append(f"Step {j}: " + str(nsrt))
                task_str.append("")

                # Where did the refinement fail?
                task_str.append("The motion plan (refinment) got stuck at "+
                                "step " + str(longest_refinement_len-1) + '\n')
                print("The motion plan (refinment) got stuck at step " + 
                                str(longest_refinement_len-1))

                # Executing the plan
                state = env.reset(train_or_test='train', task_idx=i)
                task_str.append("The state-action trajectory is:")
                task_str.append("State: " + d2s(state.data) + "\n")
                nsrt_counter = 0
                if longest_refinement_len > 1:
                    head_options = longest_refinement[:-1]
                    def policy(_): raise OptionExecutionFailure("")
                    for _ in range(CFG.horizon):
                        try:
                            act = policy(state)
                            task_str.append("Action: " + str(act._arr) + "\n")
                            state = env.step(act)
                            task_str.append("State: " + d2s(state.data) + "\n")
                        except OptionExecutionFailure:
                            # When the one-option policy reaches terminal state
                            try:
                                policy = utils.option_plan_to_policy(
                                        [head_options.pop(0)])
                            except IndexError:
                                # The head_options plan is empty
                                break
                            else:
                                task_str.append(f"Abstract State: " + 
                                                str(utils.abstract(state, 
                                                    self._initial_predicates)) +
                                                "\n")
                                task_str.append(
                                    f"Start executing Step {nsrt_counter}: " +
                                    f"{str(nsrt_plan[nsrt_counter])}\n")
                                nsrt_counter += 1
                    
                tail_policy = option_plan_to_policy(longest_refinement[-1:], 
                                            max_option_steps=None,
                                            raise_error_on_repeated_state=True)

                # State before executing the last option
                task_str.append(f"Abstract State: " + str(utils.abstract(
                    state, self._initial_predicates)) + "\n")
                task_str.append(f"Start executing Step {nsrt_counter} which " +
                                f"failed: {str(nsrt_plan[nsrt_counter])}\n")
                # Action trajectories before failure
                for _ in range(CFG.horizon):
                    try:
                        act = tail_policy(state)
                        task_str.append("Action: " + str(act._arr) + "\n")
                    except OptionExecutionFailure as e:
                        task_str.append("Action attempt: " + str(act._arr)+"\n")
                        task_str.append("The option failed because: " +
                                            str(e) + '\n')
                        break
                    try:
                        state = env.step(act)
                        task_str.append("State: " + d2s(state.data) + '\n')
                    except EnvironmentFailure as e:
                        task_str.append("The option failed because: " +
                                            str(e)+'\n')
                        break
                
        task_str = '\n'.join(task_str)
        template = template.replace("[INTERACTION_RESULTS]", task_str)

        # Save the text prompt
        with open('./prompts/online_invent_1.prompt', 'w') as file:
            file.write(template)

        prompt = template

        # if CFG.rgb_observation:
        #     # Visual observation
        #     images = []
        #     for i, trajectory in enumerate(dataset.trajectories):
        #         # Get the init observation in the trajectory
        #         img_save_path = f'./prompts/init_obs_{i}.png'
        #         observation = trajectory.states[0].rendered_state['scene'][0]
        #         imageio.imwrite(img_save_path, observation)

        #         # Encode the image
        #         image_str = encode_image(img_save_path)
        #         # Add the image to the images list
        #         images.append(image_str)

        # ########### Make the prompt ###########
        # # Create the text entry
        # text_entry = {
        #     "type": "text",
        #     "text": text_prompt
        # }
        
        # prompt = [text_entry]
        # if CFG.rgb_observation:
        #     # Create the image entries
        #     image_entries = []
        #     for image_str in images:
        #         image_entry = {
        #             "type": "image_url",
        #             "image_url": {
        #                 "url": f"data:image/png;base64,{image_str}"
        #             }
        #         }
        #         image_entries.append(image_entry)

        #     # Combine the text entry and image entries and Create the final prompt
        #     prompt += image_entries

        # prompt = [{
        #         "role": "user",
        #         "content": prompt
        #     }]

        # # Convert the prompt to JSON string
        # prompt_json = json.dumps(prompt, indent=2)
        # with open('./prompts/invent_2_cover_final.prompt', 'w') as file:
        #     file.write(str(prompt_json))
        # # Can be loaded with:
        # # with open('./prompts/2_invention_cover_final.prompt', 'r') as file:
        # #     prompt = json.load(file)
        return prompt

    def _create_interpretation_prompt(self, pred: AnnotatedPredicate, idx: int) -> str:
        with open('./prompts/interpret_0.prompt', 'r') as file:
            template = file.read()
        text_prompt = template.replace('[INSERT_QUERY_HERE]', pred.__str__())

        # Save the text prompt
        with open(f'./prompts/interpret_1_cover_{idx}_{pred.name}_text.prompt', 'w') \
            as file:
            file.write(text_prompt)
        
        text_entry = {
            "type": "text",
            "text": text_prompt
        }
        prompt = [{
            "role": "user",
            "content": text_entry
        }]

        # Convert the prompt to JSON string
        prompt_json = json.dumps(prompt, indent=2)
        with open(f'./prompts/interpret_2_cover_{idx}_{pred.name}.prompt', 'w') \
            as file:
            file.write(str(prompt_json))
        return prompt

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

    def _parse_predicate_signature_predictions(self, response: str) -> Set[Predicate]:

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
            predicate = AnnotatedPredicate(name=name, types=types, 
                                           description=description,
                                           _classifier=None)
            predicates.append(predicate)
        for pred in predicates: 
            logging.info(pred)
        return predicates

# Function to encode the image
def encode_image(image_path: str) -> str:
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')