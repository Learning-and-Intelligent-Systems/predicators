"""Ground-truth NSRTs for the playroom environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.playroom import PlayroomEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, \
    ParameterizedOption, Predicate, State, Type, Variable


class PlayroomGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the playroom environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {
            "playroom", "playroom_simple", "playroom_hard",
            "playroom_simple_clear"
        }

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type],
                  predicates: Dict[str, Predicate],
                  options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Types
        block_type = types["block"]
        robot_type = types["robot"]
        dial_type = types["dial"]

        # Predicates
        On = predicates["On"]
        OnTable = predicates["OnTable"]
        GripperOpen = predicates["GripperOpen"]
        Holding = predicates["Holding"]
        Clear = predicates["Clear"]
        NextToTable = predicates["NextToTable"]
        NextToDial = predicates["NextToDial"]
        LightOn = predicates["LightOn"]
        LightOff = predicates["LightOff"]

        # Options
        Pick = options["Pick"]
        Stack = options["Stack"]
        PutOnTable = options["PutOnTable"]
        TurnOnDial = options["TurnOnDial"]
        TurnOffDial = options["TurnOffDial"]

        # Additional types, predicates, and options
        if env_name in ("playroom_simple", "playroom_simple_clear"):
            MoveTableToDial = options["MoveTableToDial"]

        else:  # playroom or playroom_hard
            door_type = types["door"]
            region_type = types["region"]

            NextToDoor = predicates["NextToDoor"]
            InRegion = predicates["InRegion"]
            Borders = predicates["Borders"]
            Connects = predicates["Connects"]
            IsBoringRoom = predicates["IsBoringRoom"]
            IsPlayroom = predicates["IsPlayroom"]
            IsBoringRoomDoor = predicates["IsBoringRoomDoor"]
            IsPlayroomDoor = predicates["IsPlayroomDoor"]
            DoorOpen = predicates["DoorOpen"]
            DoorClosed = predicates["DoorClosed"]

            MoveToDoor = options["MoveToDoor"]
            MoveDoorToTable = options["MoveDoorToTable"]
            MoveDoorToDial = options["MoveDoorToDial"]
            OpenDoor = options["OpenDoor"]
            CloseDoor = options["CloseDoor"]

        nsrts = set()

        # PickFromTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [robot, block]
        option_vars = [robot, block]
        option = Pick
        preconditions = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot]),
            LiftedAtom(NextToTable, [robot])
        }
        add_effects = {LiftedAtom(Holding, [block])}
        delete_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }

        def pickfromtable_sampler(state: State, goal: Set[GroundAtom],
                                  rng: np.random.Generator,
                                  objs: Sequence[Object]) -> Array:
            del goal, rng  # unused
            assert len(objs) == 2
            _, block = objs
            assert block.is_instance(block_type)
            # find rotation of robot that faces the table
            x, y = state.get(block, "pose_x"), state.get(block, "pose_y")
            cls = PlayroomEnv
            table_x = (cls.table_x_lb + cls.table_x_ub) / 2
            table_y = (cls.table_y_lb + cls.table_y_ub) / 2
            rotation = np.arctan2(table_y - y, table_x - x) / np.pi
            return np.array([rotation], dtype=np.float32)

        pickfromtable_nsrt = NSRT("PickFromTable", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars,
                                  pickfromtable_sampler)
        nsrts.add(pickfromtable_nsrt)

        # Unstack
        block = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, otherblock, robot]
        option_vars = [robot, block]
        option = Pick
        preconditions = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot]),
            LiftedAtom(NextToTable, [robot])
        }
        add_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }
        delete_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }

        def unstack_sampler(state: State, goal: Set[GroundAtom],
                            rng: np.random.Generator,
                            objs: Sequence[Object]) -> Array:
            del goal, rng  # unused
            assert len(objs) == 3
            block, _, _ = objs
            assert block.is_instance(block_type)
            # find rotation of robot that faces the table
            x, y = state.get(block, "pose_x"), state.get(block, "pose_y")
            cls = PlayroomEnv
            table_x = (cls.table_x_lb + cls.table_x_ub) / 2
            table_y = (cls.table_y_lb + cls.table_y_ub) / 2
            rotation = np.arctan2(table_y - y, table_x - x) / np.pi
            return np.array([rotation], dtype=np.float32)

        unstack_nsrt = NSRT("Unstack", parameters, preconditions, add_effects,
                            delete_effects, set(), option, option_vars,
                            unstack_sampler)
        nsrts.add(unstack_nsrt)

        # Stack
        block = Variable("?block", block_type)
        otherblock = Variable("?otherblock", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, otherblock, robot]
        option_vars = [robot, otherblock]
        option = Stack
        preconditions = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock]),
            LiftedAtom(NextToTable, [robot])
        }
        add_effects = {
            LiftedAtom(On, [block, otherblock]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(Clear, [otherblock])
        }

        def stack_sampler(state: State, goal: Set[GroundAtom],
                          rng: np.random.Generator,
                          objs: Sequence[Object]) -> Array:
            del goal, rng  # unused
            assert len(objs) == 3
            _, otherblock, _ = objs
            assert otherblock.is_instance(block_type)
            # find rotation of robot that faces the table
            x, y = state.get(otherblock,
                             "pose_x"), state.get(otherblock, "pose_y")
            cls = PlayroomEnv
            table_x = (cls.table_x_lb + cls.table_x_ub) / 2
            table_y = (cls.table_y_lb + cls.table_y_ub) / 2
            rotation = np.arctan2(table_y - y, table_x - x) / np.pi
            return np.array([rotation], dtype=np.float32)

        stack_nsrt = NSRT("Stack", parameters, preconditions, add_effects,
                          delete_effects, set(), option, option_vars,
                          stack_sampler)
        nsrts.add(stack_nsrt)

        # PutOnTable
        block = Variable("?block", block_type)
        robot = Variable("?robot", robot_type)
        parameters = [block, robot]
        option_vars = [robot]
        option = PutOnTable
        preconditions = {
            LiftedAtom(Holding, [block]),
            LiftedAtom(NextToTable, [robot])
        }
        add_effects = {
            LiftedAtom(OnTable, [block]),
            LiftedAtom(Clear, [block]),
            LiftedAtom(GripperOpen, [robot])
        }
        delete_effects = {LiftedAtom(Holding, [block])}

        def putontable_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal, objs  # unused
            x = rng.uniform()
            y = rng.uniform()
            # find rotation of robot that faces the table
            cls = PlayroomEnv
            table_x = (cls.table_x_lb + cls.table_x_ub) / 2
            table_y = (cls.table_y_lb + cls.table_y_ub) / 2
            rotation = np.arctan2(table_y - y, table_x - x) / np.pi
            return np.array([x, y, rotation], dtype=np.float32)

        putontable_nsrt = NSRT("PutOnTable", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, putontable_sampler)
        nsrts.add(putontable_nsrt)

        # TurnOnDial
        robot = Variable("?robot", robot_type)
        dial = Variable("?dial", dial_type)
        parameters = [robot, dial]
        option_vars = [robot, dial]
        option = TurnOnDial
        preconditions = {
            LiftedAtom(NextToDial, [robot, dial]),
            LiftedAtom(LightOff, [dial]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(LightOn, [dial])}
        delete_effects = {LiftedAtom(LightOff, [dial])}

        def toggledial_sampler(state: State, goal: Set[GroundAtom],
                               rng: np.random.Generator,
                               objs: Sequence[Object]) -> Array:
            del state, goal, rng, objs  # unused
            return np.array([-0.2, 0, 0, 0.0], dtype=np.float32)

        turnondial_nsrt = NSRT("TurnOnDial", parameters, preconditions,
                               add_effects, delete_effects, set(), option,
                               option_vars, toggledial_sampler)
        nsrts.add(turnondial_nsrt)

        # TurnOffDial
        robot = Variable("?robot", robot_type)
        dial = Variable("?dial", dial_type)
        parameters = [robot, dial]
        option_vars = [robot, dial]
        option = TurnOffDial
        preconditions = {
            LiftedAtom(NextToDial, [robot, dial]),
            LiftedAtom(LightOn, [dial]),
            LiftedAtom(GripperOpen, [robot])
        }
        add_effects = {LiftedAtom(LightOff, [dial])}
        delete_effects = {LiftedAtom(LightOn, [dial])}
        turnoffdial_nsrt = NSRT("TurnOffDial", parameters,
                                preconditions, add_effects, delete_effects,
                                set(), option, option_vars, toggledial_sampler)
        nsrts.add(turnoffdial_nsrt)

        if env_name in ("playroom_simple", "playroom_simple_clear"):
            # MoveTableToDial
            robot = Variable("?robot", robot_type)
            dial = Variable("?dial", dial_type)
            parameters = [robot, dial]
            option_vars = [robot, dial]
            option = MoveTableToDial
            preconditions = {LiftedAtom(NextToTable, [robot])}
            add_effects = {LiftedAtom(NextToDial, [robot, dial])}
            delete_effects = {LiftedAtom(NextToTable, [robot])}

            def movetabletodial_sampler(state: State, goal: Set[GroundAtom],
                                        rng: np.random.Generator,
                                        objs: Sequence[Object]) -> Array:
                del state, goal, rng, objs  # unused
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

            movetabletodial_nsrt = NSRT("MoveTableToDial", parameters,
                                        preconditions, add_effects,
                                        delete_effects, set(), option,
                                        option_vars, movetabletodial_sampler)
            nsrts.add(movetabletodial_nsrt)
        else:  # playroom or playroom_hard
            # AdvanceThroughDoor
            robot = Variable("?robot", robot_type)
            door = Variable("?door", door_type)
            from_region = Variable("?from", region_type)
            to_region = Variable("?to", region_type)
            parameters = [robot, door, from_region, to_region]
            option_vars = [robot, from_region, door]
            option = MoveToDoor
            preconditions = {
                LiftedAtom(InRegion, [robot, from_region]),
                LiftedAtom(Connects, [door, from_region, to_region]),
                LiftedAtom(DoorOpen, [door]),
                LiftedAtom(NextToDoor, [robot, door])
            }
            add_effects = {LiftedAtom(InRegion, [robot, to_region])}
            delete_effects = {LiftedAtom(InRegion, [robot, from_region])}

            def advancethroughdoor_sampler(state: State, goal: Set[GroundAtom],
                                           rng: np.random.Generator,
                                           objs: Sequence[Object]) -> Array:
                del goal, rng  # unused
                assert len(objs) == 4
                robot, door, _, _ = objs
                assert robot.is_instance(robot_type)
                assert door.is_instance(door_type)
                if state.get(robot, "pose_x") < state.get(door, "pose_x"):
                    return np.array([0.2, 0.0, 0.0], dtype=np.float32)
                return np.array([-0.2, 0.0, -1.0], dtype=np.float32)

            advancethroughdoor_nsrt = NSRT("AdvanceThroughDoor", parameters,
                                           preconditions, add_effects,
                                           delete_effects, set(), option,
                                           option_vars,
                                           advancethroughdoor_sampler)
            nsrts.add(advancethroughdoor_nsrt)

            # MoveTableToDoor
            robot = Variable("?robot", robot_type)
            door = Variable("?door", door_type)
            region = Variable("?region", region_type)
            parameters = [robot, door, region]
            option_vars = [robot, region, door]
            option = MoveToDoor
            preconditions = {
                LiftedAtom(IsBoringRoom, [region]),
                LiftedAtom(InRegion, [robot, region]),
                LiftedAtom(NextToTable, [robot]),
                LiftedAtom(IsBoringRoomDoor, [door])
            }
            add_effects = {LiftedAtom(NextToDoor, [robot, door])}
            delete_effects = {LiftedAtom(NextToTable, [robot])}

            def movetabletodoor_sampler(state: State, goal: Set[GroundAtom],
                                        rng: np.random.Generator,
                                        objs: Sequence[Object]) -> Array:
                del state, goal, rng, objs  # unused
                return np.array([-0.2, 0.0, 0.0], dtype=np.float32)

            movetabletodoor_nsrt = NSRT("MoveTableToDoor", parameters,
                                        preconditions, add_effects,
                                        delete_effects, set(), option,
                                        option_vars, movetabletodoor_sampler)
            nsrts.add(movetabletodoor_nsrt)

            # MoveDoorToTable
            robot = Variable("?robot", robot_type)
            door = Variable("?door", door_type)
            region = Variable("?region", region_type)
            parameters = [robot, door, region]
            option_vars = [robot, region]
            option = MoveDoorToTable
            preconditions = {
                LiftedAtom(IsBoringRoom, [region]),
                LiftedAtom(InRegion, [robot, region]),
                LiftedAtom(NextToDoor, [robot, door]),
                LiftedAtom(IsBoringRoomDoor, [door])
            }
            add_effects = {LiftedAtom(NextToTable, [robot])}
            delete_effects = {LiftedAtom(NextToDoor, [robot, door])}

            def movedoortotable_sampler(state: State, goal: Set[GroundAtom],
                                        rng: np.random.Generator,
                                        objs: Sequence[Object]) -> Array:
                del state, goal, rng, objs  # unused
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

            movedoortotable_nsrt = NSRT("MoveDoorToTable", parameters,
                                        preconditions, add_effects,
                                        delete_effects, set(), option,
                                        option_vars, movedoortotable_sampler)
            nsrts.add(movedoortotable_nsrt)

            # MoveDoorToDoor
            robot = Variable("?robot", robot_type)
            fromdoor = Variable("?fromdoor", door_type)
            todoor = Variable("?todoor", door_type)
            region = Variable("?region", region_type)
            parameters = [robot, fromdoor, todoor, region]
            option_vars = [robot, region, todoor]
            option = MoveToDoor
            preconditions = {
                LiftedAtom(Borders, [fromdoor, region, todoor]),
                LiftedAtom(InRegion, [robot, region]),
                LiftedAtom(NextToDoor, [robot, fromdoor])
            }
            add_effects = {LiftedAtom(NextToDoor, [robot, todoor])}
            delete_effects = {LiftedAtom(NextToDoor, [robot, fromdoor])}

            def movedoortodoor_sampler(state: State, goal: Set[GroundAtom],
                                       rng: np.random.Generator,
                                       objs: Sequence[Object]) -> Array:
                del goal, rng  # unused
                assert len(objs) == 4
                _, fromdoor, todoor, _ = objs
                assert fromdoor.is_instance(door_type)
                assert todoor.is_instance(door_type)
                if state.get(fromdoor, "pose_x") < state.get(todoor, "pose_x"):
                    return np.array([-0.1, 0.0, 0.0], dtype=np.float32)
                return np.array([0.1, 0.0, -1.0], dtype=np.float32)

            movedoortodoor_nsrt = NSRT("MoveDoorToDoor", parameters,
                                       preconditions, add_effects,
                                       delete_effects, set(), option,
                                       option_vars, movedoortodoor_sampler)
            nsrts.add(movedoortodoor_nsrt)

            # MoveDoorToDial
            robot = Variable("?robot", robot_type)
            door = Variable("?door", door_type)
            dial = Variable("?dial", dial_type)
            region = Variable("?region", region_type)
            parameters = [robot, door, dial, region]
            option_vars = [robot, region, dial]
            option = MoveDoorToDial
            preconditions = {
                LiftedAtom(IsPlayroom, [region]),
                LiftedAtom(InRegion, [robot, region]),
                LiftedAtom(IsPlayroomDoor, [door]),
                LiftedAtom(NextToDoor, [robot, door])
            }
            add_effects = {LiftedAtom(NextToDial, [robot, dial])}
            delete_effects = {LiftedAtom(NextToDoor, [robot, door])}

            def movedoortodial_sampler(state: State, goal: Set[GroundAtom],
                                       rng: np.random.Generator,
                                       objs: Sequence[Object]) -> Array:
                del state, goal, rng, objs  # unused
                return np.array([0.0, 0.0, 0.0], dtype=np.float32)

            movedoortodial_nsrt = NSRT("MoveDoorToDial", parameters,
                                       preconditions, add_effects,
                                       delete_effects, set(), option,
                                       option_vars, movedoortodial_sampler)
            nsrts.add(movedoortodial_nsrt)

            # MoveDialToDoor
            robot = Variable("?robot", robot_type)
            dial = Variable("?dial", dial_type)
            door = Variable("?door", door_type)
            region = Variable("?region", region_type)
            parameters = [robot, dial, door, region]
            option_vars = [robot, region, door]
            option = MoveToDoor
            preconditions = {
                LiftedAtom(IsPlayroom, [region]),
                LiftedAtom(InRegion, [robot, region]),
                LiftedAtom(IsPlayroomDoor, [door]),
                LiftedAtom(NextToDial, [robot, dial])
            }
            add_effects = {LiftedAtom(NextToDoor, [robot, door])}
            delete_effects = {LiftedAtom(NextToDial, [robot, dial])}

            def movedialtodoor_sampler(state: State, goal: Set[GroundAtom],
                                       rng: np.random.Generator,
                                       objs: Sequence[Object]) -> Array:
                del state, goal, rng, objs  # unused
                return np.array([0.1, 0.0, -1.0], dtype=np.float32)

            movedialtodoor_nsrt = NSRT("MoveDialToDoor", parameters,
                                       preconditions, add_effects,
                                       delete_effects, set(), option,
                                       option_vars, movedialtodoor_sampler)
            nsrts.add(movedialtodoor_nsrt)

            # OpenDoor
            robot = Variable("?robot", robot_type)
            door = Variable("?door", door_type)
            parameters = [robot, door]
            option_vars = [robot, door]
            option = OpenDoor
            preconditions = {
                LiftedAtom(NextToDoor, [robot, door]),
                LiftedAtom(DoorClosed, [door]),
                LiftedAtom(GripperOpen, [robot])
            }
            add_effects = {LiftedAtom(DoorOpen, [door])}
            delete_effects = {LiftedAtom(DoorClosed, [door])}

            def toggledoor_sampler(state: State, goal: Set[GroundAtom],
                                   rng: np.random.Generator,
                                   objs: Sequence[Object]) -> Array:
                del goal, rng  # unused
                assert len(objs) == 2
                robot, door = objs
                assert robot.is_instance(robot_type)
                assert door.is_instance(door_type)
                x, door_x = state.get(robot,
                                      "pose_x"), state.get(door, "pose_x")
                rotation = 0.0 if x < door_x else -1.0
                dx = -0.2 if x < door_x else 0.2
                return np.array([dx, 0, 0, rotation], dtype=np.float32)

            opendoor_nsrt = NSRT("OpenDoor", parameters,
                                 preconditions, add_effects, delete_effects,
                                 set(), option, option_vars,
                                 toggledoor_sampler)
            nsrts.add(opendoor_nsrt)

            # CloseDoor
            robot = Variable("?robot", robot_type)
            door = Variable("?door", door_type)
            parameters = [robot, door]
            option_vars = [robot, door]
            option = CloseDoor
            preconditions = {
                LiftedAtom(NextToDoor, [robot, door]),
                LiftedAtom(DoorOpen, [door]),
                LiftedAtom(GripperOpen, [robot])
            }
            add_effects = {LiftedAtom(DoorClosed, [door])}
            delete_effects = {LiftedAtom(DoorOpen, [door])}
            closedoor_nsrt = NSRT("CloseDoor", parameters,
                                  preconditions, add_effects, delete_effects,
                                  set(), option, option_vars,
                                  toggledoor_sampler)
            nsrts.add(closedoor_nsrt)

        return nsrts
