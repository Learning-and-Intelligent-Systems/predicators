"""Ground-truth NSRTs for the Kitchen environment."""

from typing import Dict, Sequence, Set

import numpy as np

from predicators.envs.kitchen import KitchenEnv
from predicators.ground_truth_models import GroundTruthNSRTFactory
from predicators.settings import CFG
from predicators.structs import NSRT, Array, GroundAtom, LiftedAtom, Object, ParameterizedOption, Predicate, State, Type, Variable


class RoboKitchenGroundTruthNSRTFactory(GroundTruthNSRTFactory):
    """Ground-truth NSRTs for the RoboKitchen environment."""

    @classmethod
    def get_env_names(cls) -> Set[str]:
        return {"robo_kitchen"}

    @staticmethod
    def get_nsrts(env_name: str, types: Dict[str, Type], predicates: Dict[str, Predicate], options: Dict[str, ParameterizedOption]) -> Set[NSRT]:

        # Types
        gripper_type = types["gripper_type"]
        handle_type = types["handle_type"]
        hinge_type = types["hinge_type"]

        # Objects
        gripper = Variable("?gripper", gripper_type)
        handle = Variable("?handle", handle_type)
        hinge = Variable("?hinge_door", hinge_type)

        # Options
        DummyOption = options["DummyOption"]

        # Predicates
        ReadyGrabHandle = predicates["ReadyGrabHandle"]
        GripperClosed = predicates["GripperClosed"]
        GripperOpen = predicates["GripperOpen"]
        HingeClosed = predicates["HingeClosed"]
        HingeOpen = predicates["HingeOpen"]
        InContact = predicates["InContact"]

        nsrts = set()

        # OpenGripper
        parameters = [gripper]
        preconditions = set()
        add_effects = {LiftedAtom(GripperOpen, [gripper])}
        delete_effects = {LiftedAtom(GripperClosed, [gripper])}
        ignore_effects = set()
        option = DummyOption
        option_vars = []

        def open_gripper_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)
        
        open_gripper_nsrt = NSRT(
            "OpenGripper",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            open_gripper_sampler,
        )
        nsrts.add(open_gripper_nsrt)

        # MoveToHandle
        parameters = [gripper, handle]
        preconditions = {LiftedAtom(GripperOpen, [gripper])}
        add_effects = {LiftedAtom(ReadyGrabHandle, [gripper, handle])}
        delete_effects = set()
        ignore_effects = set()
        option = DummyOption
        option_vars = []
        
        def move_to_handle_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)
        
        move_to_handle_nsrt = NSRT(
            "MoveToHandle",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            move_to_handle_sampler,
        )
        nsrts.add(move_to_handle_nsrt)

        # GrabHandle
        parameters = [gripper, handle]
        preconditions = {LiftedAtom(ReadyGrabHandle, [gripper, handle]), LiftedAtom(GripperOpen, [gripper])}
        add_effects = {LiftedAtom(GripperClosed, [gripper]), LiftedAtom(InContact, [gripper, handle])}
        delete_effects = {LiftedAtom(GripperOpen, [gripper])}
        ignore_effects = set()
        option = DummyOption
        option_vars = []

        def grab_handle_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)

        grab_handle_nsrt = NSRT(
            "GrabHandle",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            grab_handle_sampler,
        )
        nsrts.add(grab_handle_nsrt)

        # PullOpenDoor
        parameters = [gripper, handle, hinge]
        preconditions = {LiftedAtom(HingeClosed, [hinge]), LiftedAtom(ReadyGrabHandle, [gripper, handle]), LiftedAtom(GripperClosed, [gripper]), LiftedAtom(InContact, [gripper, handle])}
        add_effects = {LiftedAtom(HingeOpen, [hinge])}
        delete_effects = {LiftedAtom(HingeClosed, [hinge])}
        ignore_effects = set()
        option = DummyOption
        option_vars = []
        
        def pull_open_door_sampler(state: State, memory: dict, objects: Sequence[Object], params: Array) -> Array:
            return np.array([0], dtype=np.float32)
        
        pull_open_door_nsrt = NSRT(
            "PullOpenDoor",
            parameters,
            preconditions,
            add_effects,
            delete_effects,
            ignore_effects,
            option,
            option_vars,
            pull_open_door_sampler,
        )
        nsrts.add(pull_open_door_nsrt)

        # nsrts.add(push_close_hinge_door_nsrt)

        return nsrts
