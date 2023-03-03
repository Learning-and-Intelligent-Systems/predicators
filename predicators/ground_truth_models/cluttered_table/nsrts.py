"""Ground-truth NSRTs for the cluttered table environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class ClutteredTableGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the cluttered table environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"cluttered_table", "cluttered_table_place"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        if env_name == "cluttered_table":
            with_place = False
        else:
            assert env_name == "cluttered_table_place"
            with_place = True

        # Types
        can_type = types["can"]

        # Predicates
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        Untrashed = predicates["Untrashed"]

        # Options
        Grasp = options["Grasp"]
        if with_place:
            Place = options["Place"]
        else:
            Dump = options["Dump"]

        nsrts = set()

        # Grasp
        can = Variable("?can", can_type)
        parameters = [can]
        option_vars = [can]
        option = Grasp
        preconditions = {
            LiftedAtom(HandEmpty, []),
            LiftedAtom(Untrashed, [can])
        }
        add_effects = {LiftedAtom(Holding, [can])}
        delete_effects = {LiftedAtom(HandEmpty, [])}

        def grasp_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del goal  # unused
            assert len(objs) == 1
            can = objs[0]
            # Need a max here in case the can is trashed already, in which case
            # both pose_x and pose_y will be -999.
            end_x = max(0.0, state.get(can, "pose_x"))
            end_y = max(0.0, state.get(can, "pose_y"))
            if with_place:
                start_x, start_y = 0.2, 0.1
            else:
                start_x, start_y = rng.uniform(0.0, 1.0,
                                               size=2)  # start from anywhere
            return np.array([start_x, start_y, end_x, end_y], dtype=np.float32)

        grasp_nsrt = NSRT("Grasp", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          grasp_sampler)
        nsrts.add(grasp_nsrt)

        if not with_place:
            # Dump
            can = Variable("?can", can_type)
            parameters = [can]
            option_vars = []
            option = Dump
            preconditions = {
                LiftedAtom(Holding, [can]),
                LiftedAtom(Untrashed, [can])
            }
            add_effects = {LiftedAtom(HandEmpty, [])}
            delete_effects = {
                LiftedAtom(Holding, [can]),
                LiftedAtom(Untrashed, [can])
            }
            dump_nsrt = NSRT("Dump", parameters, preconditions, add_effects,
                             delete_effects, set(), option, option_vars,
                             null_sampler)
            nsrts.add(dump_nsrt)

        else:
            # Place
            can = Variable("?can", can_type)
            parameters = [can]
            option_vars = [can]
            option = Place
            preconditions = {
                LiftedAtom(Holding, [can]),
                LiftedAtom(Untrashed, [can])
            }
            add_effects = {LiftedAtom(HandEmpty, [])}
            delete_effects = {LiftedAtom(Holding, [can])}

            def place_sampler(state: State, goal: Set[GroundAtom],
                              rng: np.random.Generator,
                              objs: Sequence[Object]) -> Array:
                start_x, start_y = 0.2, 0.1
                # Goal-conditioned sampling
                if CFG.cluttered_table_place_goal_conditioned_sampling:
                    # Get the pose of the goal object
                    assert len(goal) == 1
                    goal_atom = next(iter(goal))
                    assert goal_atom.predicate == Holding
                    goal_obj = goal_atom.objects[0]
                    goal_x = state.get(goal_obj, "pose_x")
                    goal_y = state.get(goal_obj, "pose_y")
                    # Place up w.r.t the goal, and to some distance left
                    # or right such that we're not going out of x bounds
                    # 0 to 0.4.
                    end_y = goal_y * 1.1
                    end_x = goal_x + 0.2
                    if end_x > 0.4:
                        end_x = goal_x - 0.2
                    return np.array([start_x, start_y, end_x, end_y],
                                    dtype=np.float32)
                # Non-goal-conditioned sampling
                del state, goal, objs
                return np.array([
                    start_x, start_y,
                    rng.uniform(0, 0.4),
                    rng.uniform(0, 1.0)
                ],
                                dtype=np.float32)

            place_nsrt = NSRT("Place", parameters, preconditions, add_effects,
                              delete_effects, set(), option, option_vars,
                              place_sampler)
            nsrts.add(place_nsrt)

        return nsrts
