"""A TAMP approach that uses hand-specified operators.

The approach is aware of the initial predicates and options.
Predicates that are not in the initial predicates are excluded from
the ground truth operators. If an operator's option is not included,
that operator will not be generated at all.
"""

from typing import List, Sequence, Set
import numpy as np
from predicators.src.approaches import TAMPApproach
from predicators.src.envs import get_env_instance
from predicators.src.structs import Operator, Predicate, State, \
    ParameterizedOption, Variable, Type, LiftedAtom, Object, Array
from predicators.src.settings import CFG


class OracleApproach(TAMPApproach):
    """A TAMP approach that uses hand-specified operators.
    """
    @property
    def is_learning_based(self) -> bool:
        return False

    def _get_current_operators(self) -> Set[Operator]:
        return get_gt_ops(self._initial_predicates, self._initial_options)


def get_gt_ops(predicates: Set[Predicate],
               options: Set[ParameterizedOption]) -> Set[Operator]:
    """Create ground truth operators for an env.
    """
    if CFG.env == "cover":
        ops = _get_cover_gt_ops(options_are_typed=False)
    elif CFG.env == "cover_typed":
        ops = _get_cover_gt_ops(options_are_typed=True)
    elif CFG.env == "cluttered_table":
        ops = _get_cluttered_table_gt_ops()
    elif CFG.env == "behavior":
        ops = _get_behavior_gt_ops()
    else:
        raise NotImplementedError("Ground truth operators not implemented")
    # Filter out excluded predicates/options
    final_ops = set()
    for op in ops:
        if op.option not in options:
            continue
        op = op.filter_predicates(predicates)
        final_ops.add(op)
    return final_ops


def _get_from_env_by_names(env_name: str, names: Sequence[str],
                           env_attr: str) -> List:
    """Helper for loading types, predicates, and options by name.
    """
    env = get_env_instance(env_name)
    name_to_env_obj = {}
    for o in getattr(env, env_attr):
        name_to_env_obj[o.name] = o
    assert set(name_to_env_obj).issuperset(set(names))
    return [name_to_env_obj[name] for name in names]


def _get_types_by_names(env_name: str,
                        names: Sequence[str]) -> List[Type]:
    """Load types from an env given their names.
    """
    return _get_from_env_by_names(env_name, names, "types")


def _get_predicates_by_names(env_name: str,
                             names: Sequence[str]) -> List[Predicate]:
    """Load predicates from an env given their names.
    """
    return _get_from_env_by_names(env_name, names, "predicates")


def _get_options_by_names(env_name: str,
                          names: Sequence[str]) -> List[ParameterizedOption]:
    """Load parameterized options from an env given their names.
    """
    return _get_from_env_by_names(env_name, names, "options")


def _get_cover_gt_ops(options_are_typed: bool) -> Set[Operator]:
    """Create ground truth operators for CoverEnv.
    """
    block_type, target_type = _get_types_by_names("cover", ["block", "target"])

    IsBlock, IsTarget, Covers, HandEmpty, Holding = \
        _get_predicates_by_names("cover", ["IsBlock", "IsTarget", "Covers",
                                           "HandEmpty", "Holding"])

    if options_are_typed:
        Pick, Place = _get_options_by_names("cover_typed", ["Pick", "Place"])
    else:
        PickPlace, = _get_options_by_names("cover", ["PickPlace"])

    operators = set()

    # Pick
    block = Variable("?block", block_type)
    parameters = [block]
    if options_are_typed:
        option_vars = [block]
        option = Pick
    else:
        option_vars = []
        option = PickPlace
    preconditions = {LiftedAtom(IsBlock, [block]), LiftedAtom(HandEmpty, [])}
    add_effects = {LiftedAtom(Holding, [block])}
    delete_effects = {LiftedAtom(HandEmpty, [])}
    def pick_sampler(state: State, rng: np.random.Generator,
                     objs: Sequence[Object]) -> Array:
        assert len(objs) == 1
        b = objs[0]
        assert b.type == block_type
        if options_are_typed:
            lb = float(-state.get(b, "width")/2)  # relative positioning only
            ub = float(state.get(b, "width")/2)  # relative positioning only
        else:
            lb = float(state.get(b, "pose") - state.get(b, "width")/2)
            lb = max(lb, 0.0)
            ub = float(state.get(b, "pose") + state.get(b, "width")/2)
            ub = min(ub, 1.0)
        return np.array(rng.uniform(lb, ub, size=(1,)), dtype=np.float32)
    pick_operator = Operator("Pick", parameters, preconditions,
                             add_effects, delete_effects, option,
                             option_vars, pick_sampler)
    operators.add(pick_operator)

    # Place
    target = Variable("?target", target_type)
    parameters = [block, target]
    if options_are_typed:
        option_vars = [target]
        option = Place
    else:
        option_vars = []
        option = PickPlace
    preconditions = {LiftedAtom(IsBlock, [block]),
                     LiftedAtom(IsTarget, [target]),
                     LiftedAtom(Holding, [block])}
    add_effects = {LiftedAtom(HandEmpty, []),
                   LiftedAtom(Covers, [block, target])}
    delete_effects = {LiftedAtom(Holding, [block])}
    def place_sampler(state: State, rng: np.random.Generator,
                      objs: Sequence[Object]) -> Array:
        assert len(objs) == 2
        t = objs[1]
        assert t.type == target_type
        lb = float(state.get(t, "pose") - state.get(t, "width")/10)
        lb = max(lb, 0.0)
        ub = float(state.get(t, "pose") + state.get(t, "width")/10)
        ub = min(ub, 1.0)
        return np.array(rng.uniform(lb, ub, size=(1,)), dtype=np.float32)
    place_operator = Operator("Place", parameters, preconditions,
                              add_effects, delete_effects, option,
                              option_vars, place_sampler)
    operators.add(place_operator)

    return operators


def _get_cluttered_table_gt_ops() -> Set[Operator]:
    """Create ground truth operators for ClutteredTableEnv.
    """
    can_type, = _get_types_by_names("cluttered_table", ["can"])

    HandEmpty, Holding = _get_predicates_by_names(
        "cluttered_table", ["HandEmpty", "Holding"])

    Grasp, Dump = _get_options_by_names("cluttered_table", ["Grasp", "Dump"])

    operators = set()

    # Grasp
    can = Variable("?can", can_type)
    parameters = [can]
    option_vars = [can]
    option = Grasp
    preconditions = {LiftedAtom(HandEmpty, [])}
    add_effects = {LiftedAtom(Holding, [can])}
    delete_effects = {LiftedAtom(HandEmpty, [])}
    def grasp_sampler(state: State, rng: np.random.Generator,
                      objs: Sequence[Object]) -> Array:
        assert len(objs) == 1
        can = objs[0]
        end_x = state.get(can, "pose_x")
        end_y = state.get(can, "pose_y")
        start_x, start_y = rng.uniform(0.0, 1.0, size=2)  # start from anywhere
        return np.array([start_x, start_y, end_x, end_y], dtype=np.float32)
    grasp_operator = Operator("Grasp", parameters, preconditions,
                              add_effects, delete_effects, option,
                              option_vars, grasp_sampler)
    operators.add(grasp_operator)

    # Dump
    can = Variable("?can", can_type)
    parameters = [can]
    option_vars = []
    option = Dump
    preconditions = {LiftedAtom(Holding, [can])}
    add_effects = {LiftedAtom(HandEmpty, [])}
    delete_effects = {LiftedAtom(Holding, [can])}
    dump_operator = Operator("Dump", parameters, preconditions, add_effects,
                             delete_effects, option, option_vars,
                             lambda s, r, o: np.array([], dtype=np.float32))
    operators.add(dump_operator)

    return operators


def _get_behavior_gt_ops() -> Set[Operator]:
    """Create ground truth operators for BehaviorEnv.
    """
    # Currently specific to re-shelving task

    type_names = [
        "book.n.02",
        "table.n.02",
        "shelf.n.01",
        "floor.n.01",
        "agent.n.01",
    ]

    book_type, table_type, shelf_type, floor_type, agent_type = \
        _get_types_by_names("behavior", type_names)

    NavigateTo, Pick, PlaceOnTop = _get_options_by_names("behavior",
        ["NavigateTo", "Pick", "PlaceOnTop"])

    operators = set()

    # Navigate to book from nowhere
    book = Variable("?book", book_type)
    parameters = [book]
    option_vars = [book]
    option = NavigateTo
    preconditions = set()
    add_effects = {_create_behavior_atom("nextto", [book_type, agent_type])}
    delete_effects = set()
    operator = Operator("NavigateToBook", parameters, preconditions, add_effects,
                        delete_effects, option, option_vars,
                        # TODO: create sampler
                        lambda s, r, o: np.array([], dtype=np.float32))
    operators.add(operator)

    # TODO more operators


    """
    Nishanth & Willie's PDDL operators

    (:action navigate_to
        :parameters (?objto - object ?agent - agent.n.01)
        :precondition (not (nextto ?objto ?agent))
        :effect (and (nextto ?objto ?agent) 
                        (when 
                            (exists 
                                (?objfrom - object) 
                                (nextto ?objfrom ?agent)
                            )
                            (not (nextto ?objfrom ?agent))
                        ) 
                )
    )

    (:action grasp
        :parameters (?obj - object ?agent - agent.n.01)
        :precondition (and (not (holding ?obj))
                            (not (handsfull ?agent)) 
                            (nextto ?obj ?agent))
        :effect (and (holding ?obj) 
                        (handsfull ?agent))
    )
    
    (:action place_ontop ; place object 1 onto object 2
        :parameters (?obj2 - object ?agent - agent.n.01 ?obj1 - object)
        :precondition (and (holding ?obj1) 
                            (nextto ?obj2 ?agent))
        :effect (and (ontop ?obj1 ?obj2) 
                        (not (holding ?obj1)) 
                        (not (handsfull ?agent)))
    )

    (:action place_inside ; place object 1 inside object 2
        :parameters (?obj2 - object ?agent - agent.n.01 ?obj1 - object)
        :precondition (and (holding ?obj1) 
                            (nextto ?obj2 ?agent) 
                            (open ?obj2))
        :effect (and (inside ?obj1 ?obj2) 
                        (not (holding ?obj1)) 
                        (not (handsfull ?agent)))
    )
    """

    return operators
