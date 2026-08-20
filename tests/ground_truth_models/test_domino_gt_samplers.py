"""Tests for the domino ground-truth grid-free per-skill samplers.

Exercises the ``ParameterizedSampler``-signature samplers exposed by
``PyBulletDominoGroundTruthSamplerFactory`` (in domino/processes.py).
The ``Place`` sampler is checked against the *real* ``InFront`` /
``Upright`` classifiers (called via a lightweight stub ``self`` so no
PyBullet env is built) — a placement it returns for an ``InFront``
subgoal must actually satisfy that subgoal.
"""

# pylint: disable=unused-import

import numpy as np
from gym.spaces import Box

from predicators import utils  # noqa: F401  (settles import order)
from predicators.envs.pybullet_domino.components.domino_component import \
    DominoComponent
from predicators.ground_truth_models import get_gt_samplers
from predicators.ground_truth_models.domino.processes import \
    _place_option_sampler
from predicators.structs import GroundAtom, Object, Predicate, State, Type

# Domino feature layout (matches the env's domino type).
_domino_type = Type("domino",
                    ["x", "y", "z", "yaw", "roll", "r", "g", "b", "is_held"])
_robot_type = Type("robot", ["x"])

# Place option's continuous-parameter box: (target_x, target_y, release_z,
# target_yaw).
_PLACE_BOX = Box(low=np.array([0.4, 1.1, 0.5, -np.pi], dtype=np.float32),
                 high=np.array([1.1, 1.6, 0.6, np.pi], dtype=np.float32))


class _ClassifierStub:
    """Stub exposing the constants the InFront/Upright classifiers read."""
    pos_gap = 0.098
    domino_width = 0.07
    domino_roll_threshold = np.deg2rad(5)


_stub = _ClassifierStub()
_InFront = Predicate("InFront", [_domino_type, _domino_type],
                     lambda s, o: DominoComponent._InFront_holds(_stub, s, o))  # type: ignore[arg-type]  # pylint: disable=protected-access
_Upright = Predicate("Upright", [_domino_type],
                     lambda s, o: DominoComponent._Upright_holds(_stub, s, o))  # type: ignore[arg-type]  # pylint: disable=protected-access


def _domino(name, x, y, yaw, is_held=0.0, rgb=(0.5, 0.5, 0.5)):
    feats = {
        "x": x,
        "y": y,
        "z": 0.475,
        "yaw": yaw,
        "roll": 0.0,
        "r": rgb[0],
        "g": rgb[1],
        "b": rgb[2],
        "is_held": is_held,
    }
    obj = Object(name, _domino_type)
    return obj, feats


def _make_state(objs_and_feats):
    data = {}
    for obj, feats in objs_and_feats:
        data[obj] = np.array([feats[f] for f in _domino_type.feature_names],
                             dtype=np.float32)
    return State(data)


def test_factory_exposes_place_pick_push():
    """The domino factory registers grid-free samplers for all 3 skills."""
    samplers = get_gt_samplers("pybullet_domino")
    assert samplers is not None
    assert set(samplers) == {"Pick", "Push", "Place"}


def test_place_sampler_satisfies_infront_subgoal():
    """Placement for InFront(held, ref) actually makes InFront hold."""
    robot = Object("robot", _robot_type)
    # Reference domino_0 at a cardinal facing (yaw=0); held domino_1 parked
    # elsewhere (its current pose is irrelevant — the sampler computes a new
    # placement from the subgoal).
    d0, f0 = _domino("domino_0", x=0.8, y=1.3, yaw=0.0)
    d1, f1 = _domino("domino_1", x=0.5, y=1.5, yaw=0.0, is_held=1.0)
    state = _make_state([(d0, f0), (d1, f1)])
    state.data[robot] = np.array([0.0], dtype=np.float32)

    subgoal = {GroundAtom(_InFront, [d1, d0]), GroundAtom(_Upright, [d1])}
    rng = np.random.default_rng(0)
    params = _place_option_sampler(state, subgoal, rng, [robot])

    assert params.shape == (4, )
    assert np.all(params >= _PLACE_BOX.low - 1e-6)
    assert np.all(params <= _PLACE_BOX.high + 1e-6)

    # Apply the placement and confirm the subgoal now holds.
    placed = state.copy()
    placed.set(d1, "x", float(params[0]))
    placed.set(d1, "y", float(params[1]))
    placed.set(d1, "yaw", float(params[3]))
    placed.set(d1, "roll", 0.0)
    placed.set(d1, "is_held", 0.0)
    assert GroundAtom(_InFront, [d1, d0]).holds(placed)
    assert GroundAtom(_Upright, [d1]).holds(placed)
    # The exact pose is no longer pinned: with a single subgoal the sampler
    # randomizes among the tied-best straight / +-45 turn placements (see
    # test_place_sampler_randomizes_turn_offset). All that is guaranteed is
    # that the drawn placement satisfies the subgoal, checked above.


def test_place_sampler_randomizes_turn_offset():
    """The sampler explores straight and +-45 turn placements across draws.

    A single InFront subgoal is satisfied equally by a straight
    placement and by a +-45 turn, so if the sampler always returned the
    same one, backtracking that re-draws an upstream Place could never
    turn a chain that needs a bend. Every draw must still satisfy the
    subgoal.
    """
    robot = Object("robot", _robot_type)
    d0, f0 = _domino("domino_0", x=0.8, y=1.3, yaw=0.0)
    d1, f1 = _domino("domino_1", x=0.5, y=1.5, yaw=0.0, is_held=1.0)
    state = _make_state([(d0, f0), (d1, f1)])
    state.data[robot] = np.array([0.0], dtype=np.float32)
    subgoal = {GroundAtom(_InFront, [d1, d0]), GroundAtom(_Upright, [d1])}

    saw_straight = False
    saw_turn = False
    for seed in range(40):
        params = _place_option_sampler(state, subgoal,
                                       np.random.default_rng(seed), [robot])
        placed = state.copy()
        placed.set(d1, "x", float(params[0]))
        placed.set(d1, "y", float(params[1]))
        placed.set(d1, "yaw", float(params[3]))
        placed.set(d1, "roll", 0.0)
        placed.set(d1, "is_held", 0.0)
        # Whatever offset was drawn, the subgoal must hold.
        assert GroundAtom(_InFront, [d1, d0]).holds(placed)
        turn = abs(utils.wrap_angle(float(params[3])))
        if turn < np.radians(10):
            saw_straight = True
        elif abs(turn - np.pi / 4) < np.radians(10):
            saw_turn = True
    assert saw_straight, "sampler never produced a straight placement"
    assert saw_turn, "sampler never produced a +-45 turn placement"


def test_place_sampler_prefers_target_bridgeable_first_placement():
    """When a purple target is visible, tie-break toward a completable chain.

    In the seed-0 test layout, every first placement of domino_1
    satisfies ``InFront(domino_1, domino_0)`` locally, but only the
    +45-degree placement leaves a one-domino bridge point that can also
    connect to the purple target.
    """
    robot = Object("robot", _robot_type)
    d0, f0 = _domino("domino_0", x=0.9146, y=1.2534, yaw=0.0)
    d1, f1 = _domino("domino_1", x=0.47, y=1.2975, yaw=0.0, is_held=1.0)
    d2, f2 = _domino("domino_2", x=0.575, y=1.2975, yaw=0.0)
    d3, f3 = _domino("domino_3",
                     x=0.7225,
                     y=1.3609,
                     yaw=np.pi / 2,
                     rgb=(0.85, 0.7, 0.85))
    state = _make_state([(d0, f0), (d1, f1), (d2, f2), (d3, f3)])
    state.data[robot] = np.array([0.0], dtype=np.float32)
    subgoal = {GroundAtom(_InFront, [d1, d0]), GroundAtom(_Upright, [d1])}

    params = _place_option_sampler(state, subgoal, np.random.default_rng(0),
                                   [robot])

    assert np.allclose(params[:2], [0.88985, 1.32665], atol=1e-3)
    assert np.isclose(float(params[2]), 0.58)
    assert abs(utils.wrap_angle(float(params[3]) - np.pi / 4)) < 1e-3


def test_place_sampler_chain_between_two_references():
    """A two-InFront subgoal lands the held domino on the shared chain
    point."""
    robot = Object("robot", _robot_type)
    # Collinear chain along +y at pos_gap spacing: d1 -- (d2 held) -- d3.
    gap = 0.098
    d1, f1 = _domino("domino_1", x=0.8, y=1.30, yaw=0.0)
    d3, f3 = _domino("domino_3", x=0.8, y=1.30 + 2 * gap, yaw=0.0)
    d2, f2 = _domino("domino_2", x=0.5, y=1.5, yaw=0.0, is_held=1.0)
    state = _make_state([(d1, f1), (d2, f2), (d3, f3)])
    state.data[robot] = np.array([0.0], dtype=np.float32)

    subgoal = {
        GroundAtom(_InFront, [d2, d1]),
        GroundAtom(_InFront, [d3, d2]),
        GroundAtom(_Upright, [d2]),
    }
    params = _place_option_sampler(state, subgoal, np.random.default_rng(0),
                                   [robot])
    placed = state.copy()
    placed.set(d2, "x", float(params[0]))
    placed.set(d2, "y", float(params[1]))
    placed.set(d2, "yaw", float(params[3]))
    placed.set(d2, "roll", 0.0)
    placed.set(d2, "is_held", 0.0)
    # Both InFront atoms satisfied at once (the shared midpoint).
    assert GroundAtom(_InFront, [d2, d1]).holds(placed)
    assert GroundAtom(_InFront, [d3, d2]).holds(placed)


def test_place_sampler_raises_without_held_domino():
    """No held domino => raise so refinement falls back to uniform."""
    robot = Object("robot", _robot_type)
    d0, f0 = _domino("domino_0", x=0.8, y=1.3, yaw=0.0)
    state = _make_state([(d0, f0)])
    state.data[robot] = np.array([0.0], dtype=np.float32)
    subgoal = {GroundAtom(_Upright, [d0])}
    try:
        _place_option_sampler(state, subgoal, np.random.default_rng(0),
                              [robot])
        assert False, "expected ValueError"
    except ValueError:
        pass
