from typing import cast

import pytest

from predicators import utils
from predicators.approaches import create_approach
from predicators.approaches.delivery_approach import DeliverySpecificApproach
from predicators.envs.pddl_env import ProceduralTasksDeliveryPDDLEnv, \
    _PDDLEnvState
from predicators.ground_truth_models import get_gt_options
from predicators.settings import CFG
from predicators.structs import GroundAtom, Task


def test_delivery_approach():
    """Tests for DeliverySpecificApproach class."""

    utils.reset_config({
        "env": "",
        "approach": f"delivery_approach",
        "timeout": 1,
    })

    env = ProceduralTasksDeliveryPDDLEnv()
    tasks = map(lambda env_task: env_task.task, env.get_test_tasks())
    approach = create_approach(CFG.approach, env.predicates,
                               get_gt_options(env.get_name()), env.types,
                               env.action_space, [])
    assert not approach.is_learning_based
    for idx, task in enumerate(tasks):
        # Checks the case when the robot has to end up
        # at some location and start from non-home-base
        if idx == 0:
            init_state = cast(_PDDLEnvState, task.init)
            at_pred = next(filter(lambda p: p.name == "at", env.predicates))
            starting_loc_ground_atom = next(
                filter(lambda atom: atom.predicate == at_pred,
                       init_state.get_ground_atoms()))
            starting_loc = starting_loc_ground_atom.objects[0]

            # Set final location to be the original starting location
            loc_type = next(filter(lambda t: t.name == "loc", env.types))
            task.goal.add(GroundAtom(at_pred, [starting_loc]))

            # Move starting location
            loc = next(
                filter(lambda loc: loc != starting_loc,
                       task.init.get_objects(loc_type)))
            init_state.get_ground_atoms().remove(starting_loc_ground_atom)
            init_state.get_ground_atoms().add(GroundAtom(at_pred, [loc]))

        policy = approach.solve(task, CFG.timeout)
        traj = utils.run_policy_with_simulator(policy,
                                               env.simulate,
                                               task.init,
                                               task.goal_holds,
                                               max_num_steps=CFG.horizon)
        assert task.goal_holds(traj.states[-1])


if __name__ == "__main__":  # pragma: no cover
    test_delivery_approach()
