"""An approach that uses hardcoded blocks policy."""

from typing import Callable

from predicators.src.approaches import BaseApproach
from predicators.src.structs import Action, State, Task


class HardcodedBlocksApproach(BaseApproach):
    """Samples random low-level actions."""

    @classmethod
    def get_name(cls) -> str:
        return "hardcoded_blocks_approach"

    @property
    def is_learning_based(self) -> bool:
        return False

    def _solve(self, task: Task, timeout: int) -> Callable[[State], Action]:

        unstacked_all = False
        stacked_phase_one = False

        def _policy(state: State) -> Action:

            nonlocal unstacked_all, stacked_phase_one

            param_options = sorted(self._initial_options, key=lambda o: o.name)
            pick_up = param_options[0]
            put_down = param_options[1]
            stack = param_options[2]
            unstack = param_options[3]

            holding_objects = {
                ga.objects[0]
                for ga in state.simulator_state
                if str(ga.predicate) == "holding"
            }
            clear_objects = {
                ga.objects[0]
                for ga in state.simulator_state if str(ga.predicate) == "clear"
            }
            on_top_objects = {
                ga.objects[0]
                for ga in state.simulator_state if str(ga.predicate) == "on"
            }
            stacked_pairs = {
                ga.objects[0]: ga.objects[1]
                for ga in state.simulator_state if str(ga.predicate) == "on"
            }

            on_table_goal = {
                ga.objects[0]
                for ga in task.goal if str(ga.predicate) == "ontable"
            }
            right_above_table_goal = {
                ga.objects[0]
                for ga in task.goal
                if str(ga.predicate) == "on" and ga.objects[1] in on_table_goal
            }
            stacked_pairs_goal = {
                ga.objects[0]: ga.objects[1]
                for ga in task.goal if str(ga.predicate) == "on"
            }
            stacked_pairs_reversed_goal = {
                ga.objects[1]: ga.objects[0]
                for ga in task.goal if str(ga.predicate) == "on"
            }

            #Phase 1: Unstacking everything

            #Phase 1a: Unstacking picked up object
            if not unstacked_all and len(holding_objects) > 0:
                opt = put_down.ground([sorted(holding_objects)[0]], [])
                return opt.policy(state)

            #Phase 1b: Unstacking picked up object
            while not unstacked_all and len(on_top_objects) > 0:
                valid_objects_to_unstack = clear_objects & on_top_objects
                top_object = sorted(valid_objects_to_unstack)[0]
                bottom_object = stacked_pairs[top_object]
                opt = unstack.ground([top_object, bottom_object], [])

                return opt.policy(state)

            unstacked_all = True

            #Stacking everything above a block on the ground
            if not stacked_phase_one:
                if len(holding_objects) > 0:
                    top_object = sorted(holding_objects)[0]
                    bottom_object = stacked_pairs_goal[top_object]
                    opt = stack.ground([top_object, bottom_object], [])
                    return opt.policy(state)
                else:
                    valid_objects_to_stack_level_one = right_above_table_goal - on_top_objects
                    if len(valid_objects_to_stack_level_one) > 0:
                        opt = pick_up.ground(
                            [sorted(valid_objects_to_stack_level_one)[0]], [])
                        return opt.policy(state)

            stacked_phase_one = True

            #Stacking everything that is not immediately above an object on the table
            valid_objects_to_stack_on = clear_objects & {
                ga
                for ga in stacked_pairs_reversed_goal.keys()
            } & {ga
                 for ga in stacked_pairs.keys()}

            if len(holding_objects) == 0:
                bottom_object = sorted(valid_objects_to_stack_on)[0]
                top_object = stacked_pairs_reversed_goal[bottom_object]
                opt = pick_up.ground([top_object], [])
                return opt.policy(state)
            else:
                top_object = sorted(holding_objects)[0]
                bottom_object = stacked_pairs_goal[top_object]
                opt = stack.ground([top_object, bottom_object], [])
                return opt.policy(state)

        return _policy
