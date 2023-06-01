"""Ground-truth NSRTs for the painting environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.painting import PaintingEnv
from predicators.envs.repeated_nextto_painting import RepeatedNextToPaintingEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable
from predicators.utils import null_sampler


class PaintingGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the painting environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"painting", "repeated_nextto_painting"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:
        # Types
        obj_type = types["obj"]
        box_type = types["box"]
        lid_type = types["lid"]
        shelf_type = types["shelf"]
        robot_type = types["robot"]

        # Predicates
        InBox = predicates["InBox"]
        InShelf = predicates["InShelf"]
        IsBoxColor = predicates["IsBoxColor"]
        IsShelfColor = predicates["IsShelfColor"]
        GripperOpen = predicates["GripperOpen"]
        OnTable = predicates["OnTable"]
        NotOnTable = predicates["NotOnTable"]
        HoldingTop = predicates["HoldingTop"]
        HoldingSide = predicates["HoldingSide"]
        Holding = predicates["Holding"]
        IsWet = predicates["IsWet"]
        IsDry = predicates["IsDry"]
        IsDirty = predicates["IsDirty"]
        IsClean = predicates["IsClean"]
        IsOpen = predicates["IsOpen"]

        # Options
        Pick = options["Pick"]
        Wash = options["Wash"]
        Dry = options["Dry"]
        Paint = options["Paint"]
        Place = options["Place"]
        OpenLid = options["OpenLid"]

        # Additional predicates and options:
        if env_name == "repeated_nextto_painting":
            NextTo = predicates["NextTo"]
            NextToBox = predicates["NextToBox"]
            NextToShelf = predicates["NextToShelf"]
            NextToTable = predicates["NextToTable"]
            MoveToObj = options["MoveToObj"]
            MoveToBox = options["MoveToBox"]
            MoveToShelf = options["MoveToShelf"]

        nsrts = set()

        # PickFromTop
        obj = Variable("?obj", obj_type)
        robot = Variable("?robot", robot_type)
        parameters = [obj, robot]
        option_vars = [robot, obj]
        option = Pick
        preconditions = {
            LiftedAtom(GripperOpen, [robot]),
            LiftedAtom(OnTable, [obj])
        }
        if env_name == "repeated_nextto_painting":
            preconditions.add(LiftedAtom(NextTo, [robot, obj]))
        add_effects = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(HoldingTop, [obj])
        }
        delete_effects = {LiftedAtom(GripperOpen, [robot])}

        def pickfromtop_sampler(state: State, goal: Set[GroundAtom],
                                rng: np.random.Generator,
                                objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs  # unused
            return np.array([1.0], dtype=np.float32)

        pickfromtop_nsrt = NSRT("PickFromTop", parameters,
                                preconditions, add_effects, delete_effects,
                                set(), option, option_vars,
                                pickfromtop_sampler)
        nsrts.add(pickfromtop_nsrt)

        # PickFromSide
        obj = Variable("?obj", obj_type)
        robot = Variable("?robot", robot_type)
        parameters = [obj, robot]
        option_vars = [robot, obj]
        option = Pick
        preconditions = {
            LiftedAtom(GripperOpen, [robot]),
            LiftedAtom(OnTable, [obj])
        }
        if env_name == "repeated_nextto_painting":
            preconditions.add(LiftedAtom(NextTo, [robot, obj]))
        add_effects = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(HoldingSide, [obj])
        }
        delete_effects = {LiftedAtom(GripperOpen, [robot])}

        def pickfromside_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs  # unused
            return np.array([0.0], dtype=np.float32)

        pickfromside_nsrt = NSRT("PickFromSide", parameters,
                                 preconditions, add_effects, delete_effects,
                                 set(), option, option_vars,
                                 pickfromside_sampler)
        nsrts.add(pickfromside_nsrt)

        # Wash
        obj = Variable("?obj", obj_type)
        robot = Variable("?robot", robot_type)
        parameters = [obj, robot]
        option_vars = [robot]
        option = Wash
        preconditions = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(IsDry, [obj]),
            LiftedAtom(IsDirty, [obj])
        }
        if env_name == "repeated_nextto_painting":
            preconditions.add(LiftedAtom(NextTo, [robot, obj]))
        add_effects = {LiftedAtom(IsWet, [obj]), LiftedAtom(IsClean, [obj])}
        delete_effects = {LiftedAtom(IsDry, [obj]), LiftedAtom(IsDirty, [obj])}

        wash_nsrt = NSRT("Wash", parameters, preconditions, add_effects,
                         delete_effects, set(), option, option_vars,
                         null_sampler)
        nsrts.add(wash_nsrt)

        # Dry
        obj = Variable("?obj", obj_type)
        robot = Variable("?robot", robot_type)
        parameters = [obj, robot]
        option_vars = [robot]
        option = Dry
        preconditions = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(IsWet, [obj]),
        }
        if env_name == "repeated_nextto_painting":
            preconditions.add(LiftedAtom(NextTo, [robot, obj]))
        add_effects = {LiftedAtom(IsDry, [obj])}
        delete_effects = {LiftedAtom(IsWet, [obj])}

        dry_nsrt = NSRT("Dry",
                        parameters, preconditions, add_effects, delete_effects,
                        set(), option, option_vars, null_sampler)
        nsrts.add(dry_nsrt)

        # PaintToBox
        obj = Variable("?obj", obj_type)
        box = Variable("?box", box_type)
        robot = Variable("?robot", robot_type)
        parameters = [obj, box, robot]
        option_vars = [robot]
        option = Paint
        preconditions = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(IsDry, [obj]),
            LiftedAtom(IsClean, [obj])
        }
        if env_name == "repeated_nextto_painting":
            preconditions.add(LiftedAtom(NextTo, [robot, obj]))
        add_effects = {LiftedAtom(IsBoxColor, [obj, box])}
        delete_effects = set()

        def painttobox_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del goal, rng  # unused
            box_color = state.get(objs[1], "color")
            return np.array([box_color], dtype=np.float32)

        painttobox_nsrt = NSRT("PaintToBox", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, painttobox_sampler)
        nsrts.add(painttobox_nsrt)

        # PaintToShelf
        obj = Variable("?obj", obj_type)
        shelf = Variable("?shelf", shelf_type)
        robot = Variable("?robot", robot_type)
        parameters = [obj, shelf, robot]
        option_vars = [robot]
        option = Paint
        preconditions = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(IsDry, [obj]),
            LiftedAtom(IsClean, [obj])
        }
        if env_name == "repeated_nextto_painting":
            preconditions.add(LiftedAtom(NextTo, [robot, obj]))
        add_effects = {LiftedAtom(IsShelfColor, [obj, shelf])}
        delete_effects = set()

        def painttoshelf_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            del goal, rng  # unused
            shelf_color = state.get(objs[1], "color")
            return np.array([shelf_color], dtype=np.float32)

        painttoshelf_nsrt = NSRT("PaintToShelf", parameters,
                                 preconditions, add_effects, delete_effects,
                                 set(), option, option_vars,
                                 painttoshelf_sampler)
        nsrts.add(painttoshelf_nsrt)

        # PlaceInBox
        obj = Variable("?obj", obj_type)
        box = Variable("?box", box_type)
        robot = Variable("?robot", robot_type)
        parameters = [obj, box, robot]
        option_vars = [robot]
        option = Place
        preconditions = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(HoldingTop, [obj]),
        }
        if env_name == "repeated_nextto_painting":
            preconditions.add(LiftedAtom(NextToBox, [robot, box]))
            preconditions.add(LiftedAtom(NextTo, [robot, obj]))
        add_effects = {
            LiftedAtom(InBox, [obj, box]),
            LiftedAtom(GripperOpen, [robot]),
            LiftedAtom(NotOnTable, [obj]),
        }
        delete_effects = {
            LiftedAtom(HoldingTop, [obj]),
            LiftedAtom(Holding, [obj]),
            LiftedAtom(OnTable, [obj]),
        }
        if env_name == "repeated_nextto_painting":
            # (Not)OnTable is affected by moving, not placing, in rnt_painting.
            # So we remove it from the add and delete effects here.
            add_effects.remove(LiftedAtom(NotOnTable, [obj]))
            delete_effects.remove(LiftedAtom(OnTable, [obj]))

        def placeinbox_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del goal  # unused
            x = state.get(objs[0], "pose_x")
            if env_name == "painting":
                y = rng.uniform(PaintingEnv.box_lb, PaintingEnv.box_ub)
                z = state.get(objs[0], "pose_z")
            elif env_name == "repeated_nextto_painting":
                y = rng.uniform(RepeatedNextToPaintingEnv.box_lb,
                                RepeatedNextToPaintingEnv.box_ub)
                z = RepeatedNextToPaintingEnv.obj_z
            return np.array([x, y, z], dtype=np.float32)

        placeinbox_nsrt = NSRT("PlaceInBox", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, placeinbox_sampler)
        nsrts.add(placeinbox_nsrt)

        # PlaceInShelf
        obj = Variable("?obj", obj_type)
        shelf = Variable("?shelf", shelf_type)
        robot = Variable("?robot", robot_type)
        parameters = [obj, shelf, robot]
        option_vars = [robot]
        option = Place
        preconditions = {
            LiftedAtom(Holding, [obj]),
            LiftedAtom(HoldingSide, [obj]),
        }
        if env_name == "repeated_nextto_painting":
            preconditions.add(LiftedAtom(NextToShelf, [robot, shelf]))
            preconditions.add(LiftedAtom(NextTo, [robot, obj]))
        add_effects = {
            LiftedAtom(InShelf, [obj, shelf]),
            LiftedAtom(GripperOpen, [robot]),
            LiftedAtom(NotOnTable, [obj]),
        }
        delete_effects = {
            LiftedAtom(HoldingSide, [obj]),
            LiftedAtom(Holding, [obj]),
            LiftedAtom(OnTable, [obj]),
        }
        if env_name == "repeated_nextto_painting":
            # (Not)OnTable is affected by moving, not placing, in rnt_painting.
            # So we remove it from the add and delete effects here.
            add_effects.remove(LiftedAtom(NotOnTable, [obj]))
            delete_effects.remove(LiftedAtom(OnTable, [obj]))

        def placeinshelf_sampler(state: State, goal: Set[GroundAtom],
                                 rng: np.random.Generator,
                                 objs: Sequence[Object]) -> Array:
            del goal  # unused
            x = state.get(objs[0], "pose_x")
            if env_name == "painting":
                y = rng.uniform(PaintingEnv.shelf_lb, PaintingEnv.shelf_ub)
                z = state.get(objs[0], "pose_z")
            elif env_name == "repeated_nextto_painting":
                y = rng.uniform(RepeatedNextToPaintingEnv.shelf_lb,
                                RepeatedNextToPaintingEnv.shelf_ub)
                z = RepeatedNextToPaintingEnv.obj_z
            return np.array([x, y, z], dtype=np.float32)

        placeinshelf_nsrt = NSRT("PlaceInShelf", parameters,
                                 preconditions, add_effects, delete_effects,
                                 set(), option, option_vars,
                                 placeinshelf_sampler)
        nsrts.add(placeinshelf_nsrt)

        # OpenLid
        lid = Variable("?lid", lid_type)
        robot = Variable("?robot", robot_type)
        parameters = [lid, robot]
        option_vars = [robot, lid]
        option = OpenLid
        preconditions = {LiftedAtom(GripperOpen, [robot])}
        add_effects = {LiftedAtom(IsOpen, [lid])}
        delete_effects = set()

        openlid_nsrt = NSRT("OpenLid", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            null_sampler)
        nsrts.add(openlid_nsrt)

        # PlaceOnTable
        for HoldingSideOrTop in [HoldingSide, HoldingTop]:
            obj = Variable("?obj", obj_type)
            robot = Variable("?robot", robot_type)
            parameters = [obj, robot]
            option_vars = [robot]
            option = Place
            if env_name == "painting":
                # The environment is a little weird: the object is technically
                # already OnTable when we go to place it on the table, because
                # of how the classifier is implemented.
                preconditions = {
                    LiftedAtom(Holding, [obj]),
                    LiftedAtom(OnTable, [obj]),
                    LiftedAtom(HoldingSideOrTop, [obj]),
                }
                add_effects = {
                    LiftedAtom(GripperOpen, [robot]),
                }
                delete_effects = {
                    LiftedAtom(Holding, [obj]),
                    LiftedAtom(HoldingSideOrTop, [obj]),
                }
            elif env_name == "repeated_nextto_painting":
                preconditions = {
                    LiftedAtom(Holding, [obj]),
                    LiftedAtom(NextTo, [robot, obj]),
                    LiftedAtom(NextToTable, [robot]),
                    LiftedAtom(HoldingSideOrTop, [obj]),
                }
                add_effects = {
                    LiftedAtom(GripperOpen, [robot]),
                    LiftedAtom(OnTable, [obj]),
                }
                delete_effects = {
                    LiftedAtom(Holding, [obj]),
                    LiftedAtom(HoldingSideOrTop, [obj]),
                    LiftedAtom(NotOnTable, [obj]),
                }

            def placeontable_sampler(state: State, goal: Set[GroundAtom],
                                     rng: np.random.Generator,
                                     objs: Sequence[Object]) -> Array:
                del goal  # unused
                x = state.get(objs[0], "pose_x")
                if env_name == "painting":
                    # Always release the object where it is, to avoid the
                    # possibility of collisions with other objects.
                    y = state.get(objs[0], "pose_y")
                    z = state.get(objs[0], "pose_z")
                elif env_name == "repeated_nextto_painting":
                    # Release the object at a randomly-chosen position on
                    # the table such that it is NextTo the robot.
                    robot_y = state.get(objs[1], "pose_y")
                    table_lb = RepeatedNextToPaintingEnv.table_lb
                    table_ub = RepeatedNextToPaintingEnv.table_ub
                    y = state.get(objs[0], "pose_y")
                    z = state.get(objs[0], "pose_z")
                    if table_lb < robot_y < table_ub:
                        nextto_thresh = RepeatedNextToPaintingEnv.nextto_thresh
                        y_sample_lb = max(table_lb, robot_y - nextto_thresh)
                        y_sample_ub = min(table_ub, robot_y + nextto_thresh)
                        y = rng.uniform(y_sample_lb, y_sample_ub)
                        z = RepeatedNextToPaintingEnv.obj_z
                return np.array([x, y, z], dtype=np.float32)

            placeontable_nsrt = NSRT(
                f"PlaceOnTableFrom{HoldingSideOrTop.name}", parameters,
                preconditions, add_effects, delete_effects, set(), option,
                option_vars, placeontable_sampler)
            nsrts.add(placeontable_nsrt)

        if env_name == "repeated_nextto_painting":

            def moveto_sampler(state: State, goal: Set[GroundAtom],
                               _rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
                del goal  # unused
                y = state.get(objs[1], "pose_y")
                return np.array([y], dtype=np.float32)

            # MoveToObj
            robot = Variable("?robot", robot_type)
            targetobj = Variable("?targetobj", obj_type)
            parameters = [robot, targetobj]
            option_vars = [robot, targetobj]
            option = MoveToObj
            preconditions = {
                LiftedAtom(GripperOpen, [robot]),
                LiftedAtom(OnTable, [targetobj]),
            }
            add_effects = {
                LiftedAtom(NextTo, [robot, targetobj]),
                LiftedAtom(NextToTable, [robot])
            }
            delete_effects = set()
            # Moving could have us end up NextTo other objects, and
            # can turn off being next to the box or the shelf.
            ignore_effects = {NextTo, NextToBox, NextToShelf}

            movetoobj_nsrt = NSRT("MoveToObj", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, moveto_sampler)
            nsrts.add(movetoobj_nsrt)

            # MoveToBox
            robot = Variable("?robot", robot_type)
            targetbox = Variable("?targetbox", box_type)
            obj = Variable("?obj", obj_type)
            parameters = [robot, targetbox, obj]
            option_vars = [robot, targetbox]
            option = MoveToBox
            preconditions = {
                LiftedAtom(NextTo, [robot, obj]),
                LiftedAtom(Holding, [obj]),
            }
            add_effects = {
                LiftedAtom(NextToBox, [robot, targetbox]),
                LiftedAtom(NextTo, [robot, obj]),
                LiftedAtom(NotOnTable, [obj])
            }
            delete_effects = {
                LiftedAtom(NextToTable, [robot]),
                LiftedAtom(OnTable, [obj])
            }
            # Moving could have us end up NextTo other objects.
            ignore_effects = {NextTo}
            movetobox_nsrt = NSRT("MoveToBox", parameters, preconditions,
                                  add_effects, delete_effects, ignore_effects,
                                  option, option_vars, moveto_sampler)
            nsrts.add(movetobox_nsrt)

            # MoveToShelf
            robot = Variable("?robot", robot_type)
            targetshelf = Variable("?targetshelf", shelf_type)
            obj = Variable("?obj", obj_type)
            parameters = [robot, targetshelf, obj]
            option_vars = [robot, targetshelf]
            option = MoveToShelf
            preconditions = {
                LiftedAtom(NextTo, [robot, obj]),
                LiftedAtom(Holding, [obj]),
            }
            add_effects = {
                LiftedAtom(NextToShelf, [robot, targetshelf]),
                LiftedAtom(NextTo, [robot, obj]),
                LiftedAtom(NotOnTable, [obj])
            }
            delete_effects = {
                LiftedAtom(NextToTable, [robot]),
                LiftedAtom(OnTable, [obj])
            }
            # Moving could have us end up NextTo other objects.
            ignore_effects = {NextTo}
            movetoshelf_nsrt = NSRT("MoveToShelf", parameters, preconditions,
                                    add_effects, delete_effects,
                                    ignore_effects, option, option_vars,
                                    moveto_sampler)
            nsrts.add(movetoshelf_nsrt)

        return nsrts
