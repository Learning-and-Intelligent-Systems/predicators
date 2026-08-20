"""A human-in-the-loop approach where the user manually selects options via
terminal prompts at each decision point.

This is the high-level (option/skill) counterpart to
``human_low_level_control_approach``: that one prompts for raw actions,
this one prompts for parameterized skills (e.g. ``SwitchOn``, ``Wait``)
and their arguments. ``Wait`` runs until the abstract atoms change.

Example (PyBullet Fan)::

    python predicators/main.py --env pybullet_fan \\
        --approach human_option_control --seed 0 \\
        --pybullet_max_vel_norm 0.1 \\
        --pybullet_sim_steps_per_action 5 --use_gui \\
        --num_train_tasks 1 --num_test_tasks 1

Swap ``--env`` for any other env that ships a Wait/process model
(``pybullet_boil``, ``pybullet_coffee``, ``pybullet_grow``).  Set
``--human_option_control_approach_use_scripted_option True`` together
with ``--scripted_option_dir`` / ``--script_option_file_name`` to replay
a pre-written option plan instead of prompting interactively.
"""

from typing import Callable, List, Optional, Sequence, Set, cast

import numpy as np
from gym.spaces import Box

from predicators import utils
from predicators.approaches import ApproachFailure
from predicators.approaches.process_planning_approach import \
    BilevelProcessPlanningApproach
from predicators.ground_truth_models import get_gt_processes
from predicators.settings import CFG
from predicators.structs import NSRT, Action, CausalProcess, \
    EndogenousProcess, GroundAtom, Object, ParameterizedOption, Predicate, \
    State, Task, Type, _GroundEndogenousProcess, _Option


class HumanOptionControlApproach(BilevelProcessPlanningApproach):
    """A human-in-the-loop approach for process-based planning.

    At each decision point, displays applicable processes to the user
    and prompts for selection via terminal input.
    """

    def __init__(self,
                 initial_predicates: Set[Predicate],
                 initial_options: Set[ParameterizedOption],
                 types: Set[Type],
                 action_space: Box,
                 train_tasks: List[Task],
                 processes: Optional[Set[CausalProcess]] = None) -> None:
        super().__init__(initial_predicates, initial_options, types,
                         action_space, train_tasks)
        if processes is None:
            # use only_endogenous for the no_invent baseline
            processes = get_gt_processes(
                CFG.env,
                self._initial_predicates,
                self._initial_options,
                only_endogenous=CFG.running_no_invent_baseline)

        self._processes = processes
        # No learning components needed for human interaction

    @classmethod
    def get_name(cls) -> str:
        return "human_option_control"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_processes(self) -> Set[CausalProcess]:
        """Get the current set of processes.

        This should be overridden if learning processes, otherwise
        returns empty set (assumes processes come from oracle/initial).
        """
        return self._processes

    def _solve(self,
               task: Task,
               timeout: int,
               _allow_replan: bool = True) -> Callable[[State], Action]:
        """Create a policy that prompts the user for process selection."""
        del timeout, _allow_replan  # Unused parameters

        # If scripted option is enabled, use the scripted plan
        if CFG.human_option_control_approach_use_scripted_option:
            try:
                option_plan = self._load_scripted_option_plan(task)
            except Exception as e:
                raise ApproachFailure(
                    f"Failed to load scripted option plan. Reason: {e}")

            policy = utils.option_plan_to_policy(
                option_plan,
                abstract_function=lambda s: utils.abstract(
                    s, self._get_current_predicates()))

            def _policy(s: State) -> Action:
                try:
                    return policy(s)
                except utils.OptionExecutionFailure as e:
                    raise ApproachFailure(e.args[0], e.info)

            return _policy

        # Otherwise, use interactive user prompting
        def _option_policy(state: State) -> _Option:
            option = self.prompt_user_for_option(state, task.goal)
            return option

        return utils.option_policy_to_policy(
            _option_policy,
            max_option_steps=CFG.max_num_steps_option_rollout,
            abstract_function=lambda s: utils.abstract(
                s, self._get_current_predicates()))

    def _load_scripted_option_plan(self, task: Task) -> Sequence[_Option]:
        """Load and parse a scripted option plan from a file.

        The file format is the same as what VLM open loop approach expects:
        a text file with "Plan:\n" followed by parsable option plan.

        Args:
            task: The task to solve

        Returns:
            Sequence of ground options
        """
        # Construct the file path
        filepath = utils.get_path_to_predicators_root() + \
            f"/scripts/{CFG.scripted_option_dir}/{CFG.script_option_file_name}"

        # Read the file
        with open(filepath, "r", encoding="utf-8") as f:
            plan_text = f.read()

        # Parse the plan (similar to vlm_open_loop_approach)
        option_plan: List[_Option] = []
        try:
            start_index = plan_text.index("Plan:\n") + len("Plan:\n")
            parsable_plan_prediction = plan_text[start_index:]
        except ValueError:
            raise ValueError("Scripted plan file is badly formatted; cannot "
                             "parse plan! Expected 'Plan:\\n' prefix.")

        # Get objects and parse the plan
        objects_list = sorted(set(task.init))
        parsed_option_plan = utils.parse_model_output_into_option_plan(
            parsable_plan_prediction, objects_list, self._types,
            self._initial_options, True)

        # Convert to grounded options
        for option_tuple in parsed_option_plan:
            # Convert params to float32
            params = np.array(option_tuple[2], dtype=np.float32)
            option_plan.append(option_tuple[0].ground(option_tuple[1], params))

        return option_plan

    def prompt_user_for_option(self, state: State,
                               goal: Set[GroundAtom]) -> _Option:
        """Prompt the user to select an option at the current state.

        First prompts for parameterized skill selection, then prompts
        for each argument one-by-one.

        Args:
            state: Current state
            goal: Goal atoms

        Returns:
            Selected option
        """
        predicates = self._get_current_predicates()

        # Display current state
        current_atoms = utils.abstract(state, predicates)
        print("\n" + "=" * 60)
        print("CURRENT STATE:")
        for atom in sorted(current_atoms, key=str):
            print(f"  {atom}")

        if CFG.human_option_control_approach_use_all_options:
            return self._prompt_user_for_option_from_all(state, goal)

        return self._prompt_user_for_option_from_processes(
            state, goal, predicates)

    def _prompt_user_for_option_from_all(  # pylint: disable=unused-argument
            self, state: State, goal: Set[GroundAtom]) -> _Option:
        """Present all initial parameterized options without process
        filtering."""
        options_list = sorted(self._initial_options, key=lambda o: o.name)

        if not options_list:
            raise ApproachFailure("No parameterized options available!")

        # Step 1: Prompt for parameterized option selection
        print("\nAVAILABLE OPTIONS (all):")
        for i, option in enumerate(options_list, 1):
            type_names = [t.name for t in option.types]
            print(f"  {i}. {option.name}({', '.join(type_names)})")

        selected_option: Optional[ParameterizedOption] = None
        while selected_option is None:
            user_input = input(
                f"\nSelect option (1-{len(options_list)}, or 'q' to quit): "
            ).strip().lower()

            if user_input == 'q':
                raise ApproachFailure("User quit option selection")

            try:
                selection = int(user_input)
                if 1 <= selection <= len(options_list):
                    selected_option = options_list[selection - 1]
                    print(f"Selected option: {selected_option.name}")
                else:
                    print(f"Invalid selection. Please enter a number "
                          f"between 1 and {len(options_list)}")
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")

        # Step 2: Prompt for object arguments one-by-one
        objects = sorted(state.data.keys(), key=str)
        selected_objects: List[Object] = []
        for param_idx, param_type in enumerate(selected_option.types):
            valid_objects = [
                o for o in objects
                if o.is_instance(param_type) and o not in selected_objects
            ]
            valid_objects = sorted(valid_objects, key=str)

            print(f"\nSelect argument {param_idx + 1} "
                  f"(type: {param_type.name}):")
            for i, obj in enumerate(valid_objects, 1):
                print(f"  {i}. {obj.name}")

            selected_obj: Optional[Object] = None
            if len(valid_objects) == 1:
                selected_obj = valid_objects[0]
                print(f"Only one valid object. "
                      f"Automatically selected: {selected_obj}")
            elif not valid_objects:
                raise ApproachFailure(
                    f"No valid objects for type {param_type.name}")
            else:
                while selected_obj is None:
                    user_input = input(
                        f"Select object (1-{len(valid_objects)}, "
                        f"or 'q' to quit): ").strip().lower()

                    if user_input == 'q':
                        raise ApproachFailure("User quit argument selection")

                    try:
                        sel = int(user_input)
                        if 1 <= sel <= len(valid_objects):
                            selected_obj = valid_objects[sel - 1]
                            print(f"Selected: {selected_obj.name}")
                        else:
                            print(f"Invalid selection. Please enter a "
                                  f"number between 1 and "
                                  f"{len(valid_objects)}")
                    except ValueError:
                        print("Invalid input. Please enter a number "
                              "or 'q' to quit.")

            assert selected_obj is not None
            selected_objects.append(selected_obj)

        # Step 3: Sample random params from the option's params_space
        params = self._rng.uniform(selected_option.params_space.low,
                                   selected_option.params_space.high)
        return selected_option.ground(selected_objects,
                                      params.astype(np.float32))

    def _prompt_user_for_option_from_processes(
            self, state: State, goal: Set[GroundAtom],
            predicates: Set[Predicate]) -> _Option:
        """Present options filtered by applicable processes."""
        applicable_processes = self._get_applicable_processes_at_state(
            state, self._get_current_processes(), predicates)

        if not applicable_processes:
            raise ApproachFailure("No applicable processes available!")

        # Group applicable processes by their parent (parameterized skill)
        # pylint: disable=import-outside-toplevel
        from collections import defaultdict

        # pylint: enable=import-outside-toplevel
        lift_processes = defaultdict(list)
        for ground_process in applicable_processes:
            parent = ground_process.parent
            lift_processes[parent].append(ground_process)

        # Step 1: Prompt for parameterized skill selection
        lift_endo_processes = list(lift_processes.keys())
        print("\nAVAILABLE SKILLS:")
        for i, parent in enumerate(lift_endo_processes, 1):
            assert isinstance(parent, EndogenousProcess)
            param_names = [p.name for p in parent.option_vars]
            print(f"  {i}. {parent.option.name}({', '.join(param_names)})")

        selected_parent: Optional[EndogenousProcess] = None
        while selected_parent is None:
            user_input = input("\nSelect skill "
                               f"(1-{len(lift_endo_processes)}"
                               ", or 'q' to quit): ").strip().lower()

            if user_input == 'q':
                raise ApproachFailure("User quit process selection")

            try:
                selection = int(user_input)
                if 1 <= selection <= len(lift_endo_processes):
                    selected_parent = cast(EndogenousProcess,
                                           lift_endo_processes[selection - 1])
                    print(f"Selected skill: {selected_parent.name}")
                else:
                    print("Invalid selection. Please "
                          "enter a number between "
                          f"1 and "
                          f"{len(lift_endo_processes)}")
            except ValueError:
                print("Invalid input. Please enter a number or 'q' to quit.")

        # Step 2: Prompt for arguments one-by-one
        applicable_for_skill = lift_processes[selected_parent]
        selected_objects = self._prompt_for_arguments(selected_parent,
                                                      applicable_for_skill,
                                                      state)

        # Find the ground process matching the selected objects
        for ground_process in applicable_for_skill:
            if ground_process.option_objs == selected_objects:
                return ground_process.sample_option(state, goal, self._rng)

        raise ApproachFailure(
            "Could not find process matching selected objects")

    def _prompt_for_arguments(
            self, parent_skill: EndogenousProcess,
            applicable_processes: List[_GroundEndogenousProcess],
            _state: State) -> List:
        """Prompt user to select arguments one-by-one for the given skill.

        Args:
            parent_skill: The parameterized skill (NSRT)
            applicable_processes: List of ground processes for this skill
            _state: Current state (unused, kept for future extensibility)

        Returns:
            List of selected objects matching the skill's parameters
        """
        selected_objects: List[Object] = []

        # For each parameter in the skill
        for param_idx, param in enumerate(parent_skill.option_vars):
            # Get valid objects for this parameter position
            valid_objects = set()
            for process in applicable_processes:
                # Only consider processes that match previously selected objects
                if all(process.objects[i] == selected_objects[i]
                       for i in range(len(selected_objects))):
                    valid_objects.add(process.objects[param_idx])

            valid_objects_list = sorted(valid_objects, key=str)

            # Display available objects for this parameter
            print(f"\nSelect argument for parameter"
                  f" '{param.name}'"
                  f" (type: {param.type.name}):")
            for i, obj in enumerate(valid_objects_list, 1):
                print(f"  {i}. {obj.name}")

            if len(valid_objects_list) == 1:
                selected_obj = valid_objects_list[0]
                print("Only one valid object. "
                      "Automatically selected: "
                      f"{selected_obj}")
            else:
                # Prompt for selection
                selected_obj = None
                while selected_obj is None:
                    user_input = input("Select object "
                                       f"(1-{len(valid_objects_list)}"
                                       ", or 'q' to quit): ").strip().lower()

                    if user_input == 'q':
                        raise ApproachFailure("User quit argument selection")

                    try:
                        selection = int(user_input)
                        if 1 <= selection <= len(valid_objects_list):
                            selected_obj = valid_objects_list[selection - 1]
                            print(f"Selected: {selected_obj.name}")
                        else:
                            print("Invalid selection."
                                  " Please enter a "
                                  "number between 1 "
                                  "and "
                                  f"{len(valid_objects_list)}")
                    except ValueError:
                        print("Invalid input. Please "
                              "enter a number or "
                              "'q' to quit.")

            assert selected_obj is not None
            selected_objects.append(selected_obj)

        return selected_objects

    def _get_applicable_processes_at_state(
            self, state: State, processes: Set[CausalProcess],
            predicates: Set[Predicate]) -> List[_GroundEndogenousProcess]:
        """Get all applicable ground processes at the current state.

        Args:
            state: Current state
            processes: Available processes
            predicates: Available predicates

        Returns:
            List of applicable ground processes
        """
        # Abstract the state
        current_atoms = utils.abstract(state, predicates)

        # Get objects from state
        objects = set(state.data.keys())

        # Ground all processes
        # pylint: disable=import-outside-toplevel,reimported
        # pylint: disable=redefined-outer-name
        from predicators.structs import EndogenousProcess, _GroundNSRT
        all_ground_processes: Set[_GroundNSRT] = set()
        for process in processes:
            # Only consider endogenous processes (action-like)
            if isinstance(process, EndogenousProcess):
                ground_processes = utils.all_ground_nsrts(process, objects)
                all_ground_processes.update(ground_processes)

        # Filter to applicable ones
        applicable = list(
            utils.get_applicable_operators(all_ground_processes,
                                           current_atoms))

        # Cast to the expected type since we only process EndogenousProcess
        return cast(List[_GroundEndogenousProcess], applicable)

    def _get_current_nsrts(self) -> Set[NSRT]:
        """Get the current set of NSRTs."""
        return set()
