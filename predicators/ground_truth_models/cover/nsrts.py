"""Ground-truth NSRTs for the cover environment."""

import logging
from typing import Dict, Sequence, Set

import numpy as np

from predicators import utils
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class CoverGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the cover environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "cover", "cover_hierarchical_types", "cover_typed_options",
            "cover_regrasp", "cover_multistep_options", "pybullet_cover",
            "pybullet_cover_typed_options", "pybullet_cover_weighted", 
            "cover_handempty", "bumpy_cover", "cover_place_hard"
        }

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        block_type = types["block"]
        target_type = types["target"]
        robot_type = types["robot"]

        # Objects
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        target = Variable("?target", target_type)

        # Predicates
        IsBlock = predicates["IsBlock"]
        IsTarget = predicates["IsTarget"]
        Covers = predicates["Covers"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        IsLight = predicates["IsLight"]

        # Options
        if env_name in ("cover", "pybullet_cover", "cover_hierarchical_types",
                        "cover_regrasp", "cover_handempty"):
            PickPlace = options["PickPlace"]
        elif env_name in ("cover_typed_options", "cover_multistep_options",
                          "bumpy_cover", "cover_place_hard",
                          "pybullet_cover_typed_options",
                          "pybullet_cover_weighted"):
            Pick, Place = options["Pick"], options["Place"]

        nsrts = set()

        # Pick
        parameters = [block]
        holding_predicate_args = [block]
        handempty_predicate_args = []
        if env_name == "cover_multistep_options":
            parameters.append(robot)
            holding_predicate_args.append(robot)
        elif env_name == "cover_handempty":
            parameters.append(robot)
            handempty_predicate_args.append(robot)
        preconditions = {
            LiftedAtom(IsBlock, [block]),
            LiftedAtom(HandEmpty, handempty_predicate_args)
        }
        if env_name in ("pybullet_cover_weighted"):
            preconditions.add(LiftedAtom(IsLight, [block]))
        add_effects = {LiftedAtom(Holding, holding_predicate_args)}
        delete_effects = {LiftedAtom(HandEmpty, handempty_predicate_args)}

        if env_name in ("cover", "pybullet_cover", "cover_hierarchical_types",
                        "cover_regrasp", "cover_handempty"):
            option = PickPlace
            option_vars = []
        elif env_name == "bumpy_cover":
            option = Pick
            option_vars = [block]
        elif env_name in ("cover_typed_options", "cover_place_hard",
                          "pybullet_cover_typed_options",
                          "pybullet_cover_weighted"):
            option = Pick
            option_vars = [block]
        elif env_name == "cover_multistep_options":
            option = Pick
            option_vars = [block, robot]

        if env_name == "cover_multistep_options":

            def pick_sampler(state: State, goal: Set[GroundAtom],
                             rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
                # The only things that change are the block's grasp, and the
                # robot's grip, holding, x, and y.
                assert len(objs) == 2
                block, robot = objs
                assert block.is_instance(block_type)
                assert robot.is_instance(robot_type)
                bx, by = state.get(block, "x"), state.get(block, "y")
                rx, ry = state.get(robot, "x"), state.get(robot, "y")
                bw = state.get(block, "width")

                if CFG.cover_multistep_goal_conditioned_sampling:
                    # Goal conditioned sampling currently assumes one goal.
                    assert len(goal) == 1
                    goal_atom = next(iter(goal))
                    t = goal_atom.objects[1]
                    tx, tw = state.get(t, "x"), state.get(t, "width")
                    thr_found = False  # target hand region
                    # Loop over objects in state to find target hand region,
                    # whose center should overlap with the target.
                    for obj in state.data:
                        if obj.type.name == "target_hand_region":
                            tlb = state.get(obj, "lb")
                            tub = state.get(obj, "ub")
                            tm = (tlb + tub) / 2  # midpoint of hand region
                            if tx - tw / 2 < tm < tx + tw / 2:
                                thr_found = True
                                break
                    assert thr_found

                if CFG.cover_multistep_degenerate_oracle_samplers:
                    desired_x = float(bx)
                elif CFG.cover_multistep_goal_conditioned_sampling:
                    # Block position adjusted by target/ thr offset
                    desired_x = bx + (tm - tx)
                else:
                    desired_x = rng.uniform(bx - bw / 2, bx + bw / 2)
                # This option changes the grasp for the block from -1.0 to 1.0,
                # so the delta is 1.0 - (-1.0) = 2.0
                block_param = [2.0]
                # The grip changes from -1.0 to 1.0.
                # The holding changes from -1.0 to 1.0.
                # x, y, grip, holding
                robot_param = [desired_x - rx, by - ry, 2.0, 2.0]
                param = block_param + robot_param
                return np.array(param, dtype=np.float32)
        else:

            def pick_sampler(state: State, goal: Set[GroundAtom],
                             rng: np.random.Generator,
                             objs: Sequence[Object]) -> Array:
                del goal  # unused
                if env_name == "cover_handempty":
                    assert len(objs) == 2
                else:
                    assert len(objs) == 1
                b = objs[0]
                assert b.is_instance(block_type)
                if env_name == "cover_typed_options":
                    lb = float(-state.get(b, "width") / 2)
                    ub = float(state.get(b, "width") / 2)
                elif env_name in (
                        "cover",
                        "pybullet_cover",
                        "cover_hierarchical_types",
                        "cover_regrasp",
                        "cover_handempty",
                        "bumpy_cover",
                        "pybullet_cover_typed_options",
                        "pybullet_cover_weighted",
                ):
                    lb = float(
                        state.get(b, "pose_y_norm") -
                        state.get(b, "width") / 2)
                    lb = max(lb, 0.0)
                    lb = min(lb, 1.0)
                    ub = float(
                        state.get(b, "pose_y_norm") +
                        state.get(b, "width") / 2)
                    ub = min(ub, 1.0)
                    ub = max(ub, 0.0)
                elif env_name == ("cover_place_hard"):
                    return np.array([state.get(b, "pose_y_norm")],
                                    dtype=np.float32)
                return np.array(rng.uniform(lb, ub, size=(1, )),
                                dtype=np.float32)

        pick_nsrt = NSRT("Pick", parameters, preconditions, add_effects,
                         delete_effects, set(), option, option_vars,
                         pick_sampler)
        nsrts.add(pick_nsrt)

        # Place (to Cover)
        parameters = [block, target]
        holding_predicate_args = [block]
        if env_name == "cover_multistep_options":
            parameters = [block, robot, target]
            holding_predicate_args.append(robot)
        elif env_name == "cover_handempty":
            parameters.append(robot)
        preconditions = {
            LiftedAtom(IsBlock, [block]),
            LiftedAtom(IsTarget, [target]),
            LiftedAtom(Holding, holding_predicate_args)
        }
        add_effects = {
            LiftedAtom(HandEmpty, handempty_predicate_args),
            LiftedAtom(Covers, [block, target])
        }
        delete_effects = {LiftedAtom(Holding, holding_predicate_args)}
        if env_name in ("cover_regrasp", "bumpy_cover"):
            Clear = predicates["Clear"]
            preconditions.add(LiftedAtom(Clear, [target]))
            delete_effects.add(LiftedAtom(Clear, [target]))

        if env_name in ("cover", "pybullet_cover", "cover_hierarchical_types",
                        "cover_regrasp", "cover_handempty"):
            option = PickPlace
            option_vars = []
        elif env_name == "bumpy_cover":
            option = Place
            option_vars = [block, target]
        elif env_name in ("cover_typed_options"):
            option = Place
            option_vars = [target]
        elif env_name in ("pybullet_cover_typed_options",
                          "pybullet_cover_weighted"):
            option = Place
            option_vars = [block, target]
        elif env_name == "cover_place_hard":
            option = Place
            option_vars = [block, target]
        elif env_name == "cover_multistep_options":
            option = Place
            option_vars = [block, robot, target]

        if env_name == "cover_multistep_options":

            def place_sampler(state: State, goal: Set[GroundAtom],
                              rng: np.random.Generator,
                              objs: Sequence[Object]) -> Array:

                if CFG.cover_multistep_goal_conditioned_sampling:
                    # Goal conditioned sampling currently assumes one goal.
                    assert len(goal) == 1
                    goal_atom = next(iter(goal))
                    t = goal_atom.objects[1]
                    tx, tw = state.get(t, "x"), state.get(t, "width")
                    thr_found = False  # target hand region
                    # Loop over objects in state to find target hand region,
                    # whose center should overlap with the target.
                    for obj in state.data:
                        if obj.type.name == "target_hand_region":
                            lb = state.get(obj, "lb")
                            ub = state.get(obj, "ub")
                            m = (lb + ub) / 2  # midpoint of hand region
                            if tx - tw / 2 < m < tx + tw / 2:
                                thr_found = True
                                break
                    assert thr_found

                assert len(objs) == 3
                block, robot, target = objs
                assert block.is_instance(block_type)
                assert robot.is_instance(robot_type)
                assert target.is_instance(target_type)
                rx = state.get(robot, "x")
                tx, tw = state.get(target, "x"), state.get(target, "width")
                if CFG.cover_multistep_degenerate_oracle_samplers:
                    desired_x = float(tx)
                elif CFG.cover_multistep_goal_conditioned_sampling:
                    desired_x = m  # midpoint of hand region
                else:
                    desired_x = rng.uniform(tx - tw / 2, tx + tw / 2)
                delta_x = desired_x - rx
                # This option changes the grasp for the block from 1.0 to -1.0,
                # so the delta is -1.0 - 1.0 = -2.0.
                # x, grasp
                block_param = [delta_x, -2.0]
                # The grip changes from 1.0 to -1.0.
                # The holding changes from 1.0 to -1.0.
                # x, grip, holding
                robot_param = [delta_x, -2.0, -2.0]
                param = block_param + robot_param
                return np.array(param, dtype=np.float32)
        else:

            def place_sampler(state: State, goal: Set[GroundAtom],
                              rng: np.random.Generator,
                              objs: Sequence[Object]) -> Array:
                del goal  # unused
                if env_name == "cover_handempty":
                    assert len(objs) == 3
                    t = objs[1]
                else:
                    assert len(objs) == 2
                    t = objs[-1]
                assert t.is_instance(target_type)
                if env_name == "bumpy_cover":
                    center = float(state.get(t, "pose_y_norm"))
                    if CFG.bumpy_cover_right_targets:
                        center += 3 * state.get(t, "width") / 4
                    lb = center - state.get(t, "width") / 2
                    ub = center + state.get(t, "width") / 2
                elif env_name == "cover_place_hard":
                    lb = float(
                        state.get(t, "pose_y_norm") - state.get(t, "width"))
                    ub = float(
                        state.get(t, "pose_y_norm") + state.get(t, "width"))
                else:
                    lb = float(
                        state.get(t, "pose_y_norm") -
                        state.get(t, "width") / 10)
                    ub = float(
                        state.get(t, "pose_y_norm") +
                        state.get(t, "width") / 10)
                lb = max(lb, 0.0)
                lb = min(lb, 1.0)
                ub = min(ub, 1.0)
                ub = max(ub, 0.0)
                return np.array(rng.uniform(lb, ub, size=(1, )),
                                dtype=np.float32)

        place_nsrt = NSRT("Place", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          place_sampler)
        nsrts.add(place_nsrt)

        # Place (not on any target)
        if env_name == "cover_regrasp":
            parameters = [block]
            preconditions = {
                LiftedAtom(IsBlock, [block]),
                LiftedAtom(Holding, [block])
            }
            add_effects = {
                LiftedAtom(HandEmpty, handempty_predicate_args),
            }
            delete_effects = {LiftedAtom(Holding, [block])}
            option = PickPlace
            option_vars = []

            def place_on_table_sampler(state: State, goal: Set[GroundAtom],
                                       rng: np.random.Generator,
                                       objs: Sequence[Object]) -> Array:
                # Always at the current location.
                del goal, rng  # this sampler is deterministic
                assert len(objs) == 1
                held_obj = objs[0]
                x = state.get(held_obj, "pose_y_norm") + state.get(
                    held_obj, "grasp")
                return np.array([x], dtype=np.float32)

            place_on_table_nsrt = NSRT("PlaceOnTable", parameters,
                                       preconditions, add_effects,
                                       delete_effects, set(), option,
                                       option_vars, place_on_table_sampler)
            nsrts.add(place_on_table_nsrt)

        return nsrts


class RegionalBumpyCoverGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the RegionalBumpyCoverEnv."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"regional_bumpy_cover"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        block_type = types["block"]
        target_type = types["target"]

        # Objects
        block = Variable("?block", block_type)
        target = Variable("?target", target_type)

        # Predicates
        Covers = predicates["Covers"]
        HandEmpty = predicates["HandEmpty"]
        Holding = predicates["Holding"]
        Clear = predicates["Clear"]
        InBumpyRegion = predicates["InBumpyRegion"]
        InSmoothRegion = predicates["InSmoothRegion"]

        # Options
        PickFromSmooth = options["PickFromSmooth"]
        PickFromBumpy = options["PickFromBumpy"]
        PickFromTarget = options["PickFromTarget"]
        PlaceOnTarget = options["PlaceOnTarget"]
        PlaceOnBumpy = options["PlaceOnBumpy"]

        nsrts = set()

        # Pick from smooth region
        parameters = [block]
        preconditions = {
            LiftedAtom(HandEmpty, []),
            LiftedAtom(InSmoothRegion, [block])
        }
        add_effects = {
            LiftedAtom(Holding, [block]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, []),
            LiftedAtom(InSmoothRegion, [block])
        }
        option = PickFromSmooth
        option_vars = parameters

        def pick_sampler(state: State, goal: Set[GroundAtom],
                         rng: np.random.Generator,
                         objs: Sequence[Object]) -> Array:
            del goal  # unused
            b = objs[0]
            assert b.is_instance(block_type)
            lb = float(state.get(b, "pose_y_norm") - state.get(b, "width") / 2)
            lb = max(lb, 0.0)
            ub = float(state.get(b, "pose_y_norm") + state.get(b, "width") / 2)
            ub = min(ub, 1.0)
            return np.array(rng.uniform(lb, ub, size=(1, )), dtype=np.float32)

        pick_from_smooth_nsrt = NSRT("PickFromSmooth", parameters,
                                     preconditions, add_effects,
                                     delete_effects, set(), option,
                                     option_vars, pick_sampler)
        nsrts.add(pick_from_smooth_nsrt)

        # Pick from bumpy region
        parameters = [block]
        preconditions = {
            LiftedAtom(HandEmpty, []),
            LiftedAtom(InBumpyRegion, [block])
        }
        add_effects = {
            LiftedAtom(Holding, [block]),
        }
        delete_effects = {
            LiftedAtom(HandEmpty, []),
            LiftedAtom(InBumpyRegion, [block])
        }
        option = PickFromBumpy
        option_vars = parameters

        pick_from_bumpy_nsrt = NSRT("PickFromBumpy", parameters,
                                    preconditions, add_effects, delete_effects,
                                    set(), option, option_vars, pick_sampler)
        nsrts.add(pick_from_bumpy_nsrt)

        # Pick from already covering target (in smooth region)
        parameters = [block, target]
        preconditions = {
            LiftedAtom(Covers, [block, target]),
            LiftedAtom(HandEmpty, []),
            LiftedAtom(InSmoothRegion, [block])
        }
        add_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [target])
        }
        delete_effects = {
            LiftedAtom(Covers, [block, target]),
            LiftedAtom(HandEmpty, []),
            LiftedAtom(InSmoothRegion, [block])
        }
        option = PickFromTarget
        option_vars = parameters

        pick_from_target_nsrt = NSRT("PickFromTarget", parameters,
                                     preconditions, add_effects,
                                     delete_effects, set(), option,
                                     option_vars, pick_sampler)
        nsrts.add(pick_from_target_nsrt)

        # Place on target
        parameters = [block, target]
        preconditions = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [target]),
        }
        add_effects = {
            LiftedAtom(HandEmpty, []),
            LiftedAtom(InSmoothRegion, [block]),
            LiftedAtom(Covers, [block, target]),
        }
        delete_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [target])
        }
        option = PlaceOnTarget
        option_vars = parameters

        def place_on_target_sampler(state: State, goal: Set[GroundAtom],
                                    rng: np.random.Generator,
                                    objs: Sequence[Object]) -> Array:
            del goal, rng  # unused
            # Degenerate oracle placing.
            block, target = objs
            target_center = state.get(target, "pose_y_norm")
            grasp = state.get(block, "grasp")
            place_pose = np.clip(target_center + grasp, 0.0, 1.0)
            return np.array([place_pose], dtype=np.float32)

        place_on_target_nsrt = NSRT("PlaceOnTarget", parameters,
                                    preconditions, add_effects, delete_effects,
                                    set(), option, option_vars,
                                    place_on_target_sampler)
        nsrts.add(place_on_target_nsrt)

        # Place in bumpy region. Note that targets are never in bumpy regions.
        parameters = [block]
        preconditions = {LiftedAtom(Holding, [block])}
        add_effects = {
            LiftedAtom(HandEmpty, []),
            LiftedAtom(InBumpyRegion, [block])
        }
        delete_effects = {LiftedAtom(Holding, [block])}
        option = PlaceOnBumpy
        option_vars = parameters

        def place_on_bumpy_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
            del goal  # unused
            max_sampling_attempts = 10000
            b, = objs
            w = state.get(b, "width") / 2
            lb = CFG.bumpy_cover_bumpy_region_start + w
            ub = 1.0 - w
            other_blocks = [
                block for block in list(state)
                if block.type.name == 'block' and block != b
            ]
            curr_pose_sample = rng.uniform(lb, ub, size=(1, ))

            # Rejection sample to avoid possible collisions between this block
            # and others that might exist already in the bumpy region.
            for num_samples in range(max_sampling_attempts):
                for other_block in other_blocks:
                    if (abs(
                            state.get(other_block, "pose_y_norm") -
                            curr_pose_sample) <=
                        (w + 0.5 * state.get(other_block, "width"))):
                        break
                else:
                    break
                curr_pose_sample = rng.uniform(lb, ub, size=(1, ))

            if num_samples == max_sampling_attempts - 1:
                logging.info(
                    "Could not find a good sample to place block in bumpy")
            return np.array(curr_pose_sample, dtype=np.float32)

        place_on_bumpy_nsrt = NSRT("PlaceOnBumpy", parameters,
                                   preconditions, add_effects, delete_effects,
                                   set(), option, option_vars,
                                   place_on_bumpy_sampler)
        nsrts.add(place_on_bumpy_nsrt)

        # Optionally include an NSRT that appears to directly pick and place
        # blocks onto targets, but actually fails completely in practice.
        if CFG.regional_bumpy_cover_include_impossible_nsrt:

            ImpossiblePickPlace = options["ImpossiblePickPlace"]

            parameters = [block, target]
            preconditions = {
                LiftedAtom(HandEmpty, []),
                LiftedAtom(Clear, [target]),
            }
            add_effects = {
                LiftedAtom(HandEmpty, []),
                LiftedAtom(InSmoothRegion, [block]),
                LiftedAtom(Covers, [block, target]),
            }
            delete_effects = {
                LiftedAtom(HandEmpty, []),
                LiftedAtom(InBumpyRegion, [block])
            }
            option = ImpossiblePickPlace
            option_vars = parameters

            impossible_pick_place_nsrt = NSRT("ImpossiblePickPlace",
                                              parameters, preconditions,
                                              add_effects, delete_effects,
                                              set(), option, option_vars,
                                              utils.null_sampler)
            nsrts.add(impossible_pick_place_nsrt)

        return nsrts
