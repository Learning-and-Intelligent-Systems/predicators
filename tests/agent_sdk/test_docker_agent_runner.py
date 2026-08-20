"""Tests for stale Object hash repair after cross-process unpickling.

Characterizes ``_rehash_objects_after_unpickle``: Object caches
``_hash = hash(str(self))`` in a cached_property, and PYTHONHASHSEED
randomization makes those cached values stale in a new process. The
tests simulate that by planting a bogus ``_hash`` before building the
``State.data`` dict, so the dict's stored entry hashes are stale, which
is exactly what a cross-seed dill roundtrip produces (verified E2E).

Characterized behavior (current, not aspirational):

- The rehash clears every reachable Object's cached ``_hash``/``_str``,
  so stale and freshly created equal Objects hash identically again.
- The ``state.data = dict(state.data)`` rebuild does NOT re-key stale
  entries: CPython's ``dict(d)`` copies each entry's stored hash without
  recomputing it, so lookups into pre-existing dicts remain broken (a
  dict comprehension would repair them). Tests below pin this down.

Only ``_rehash_objects_after_unpickle`` is under test; ``main()`` and
``_run_query`` need Docker and the SDK.
"""
# pylint: disable=protected-access
from types import SimpleNamespace

import numpy as np
import pytest

from predicators.agent_sdk.docker_agent_runner import \
    _rehash_objects_after_unpickle
from predicators.structs import Action, GroundAtom, LowLevelTrajectory, \
    Object, Predicate, State, Task, Type

_block_type = Type("block", ["x"])
_OnTable = Predicate("OnTable", [_block_type], lambda s, o: True)


def _make_state(obj, value=0.0):
    return State({obj: np.array([value], dtype=np.float32)})


def _corrupt_hash(obj):
    """Plant a stale cached hash, as if pickled under another hash seed."""
    true_hash = hash(obj)  # Populate the cached_property.
    # +1 guarantees a different bucket index modulo any power-of-two
    # table size, so corrupted lookups miss deterministically.
    obj.__dict__["_hash"] = true_hash + 1
    return true_hash


def _stale_state(name="block0"):
    """Build a state whose dict entries are stored under a stale hash.

    Corrupting before insertion mirrors unpickling: dill inserts keys
    while their cached (old-process) ``_hash`` is in effect.
    """
    obj = Object(name, _block_type)
    _corrupt_hash(obj)
    state = _make_state(obj)
    return obj, state


def test_stale_hash_breaks_fresh_object_lookup():
    """Precondition: a fresh equal Object cannot find the stale key."""
    stale_obj, state = _stale_state()
    fresh_obj = Object("block0", _block_type)
    assert fresh_obj == stale_obj
    assert hash(fresh_obj) != hash(stale_obj)
    with pytest.raises(KeyError):
        _ = state.data[fresh_obj]
    # The stale instance itself still works: its cached hash matches
    # the hash stored at insertion.
    assert state.data[stale_obj] is not None


def test_rehash_clears_hash_caches_on_all_ctx_surfaces():
    """All reachable Objects re-hash equal to fresh ones after rehash."""
    train_obj, train_state = _stale_state("train_block")
    train_task = Task(train_state, {GroundAtom(_OnTable, [train_obj])})

    cur_obj, cur_state = _stale_state("cur_block")
    cur_task = Task(cur_state, {GroundAtom(_OnTable, [cur_obj])})

    example_obj, example_state = _stale_state("example_block")

    traj_obj, traj_state0 = _stale_state("traj_block")
    # traj_obj still carries the stale cache, so this second dict is
    # keyed under it too, like a second unpickled state.
    traj_state1 = _make_state(traj_obj, value=1.0)
    traj = LowLevelTrajectory([traj_state0, traj_state1],
                              [Action(np.zeros(1, dtype=np.float32))])

    ctx = SimpleNamespace(train_tasks=[train_task],
                          current_task=cur_task,
                          example_state=example_state,
                          offline_trajectories=[traj],
                          online_trajectories=[])
    _rehash_objects_after_unpickle(ctx)

    for name, obj in (("train_block", train_obj), ("cur_block", cur_obj),
                      ("example_block", example_obj), ("traj_block",
                                                       traj_obj)):
        fresh = Object(name, _block_type)
        assert hash(obj) == hash(fresh), f"cache not cleared for {name}"
    # Goal-atom objects were cleared too (same instances here, but the
    # atom path is walked independently of the state path).
    goal_obj = next(iter(cur_task.goal)).objects[0]
    assert hash(goal_obj) == hash(Object("cur_block", _block_type))


def test_rehash_rekeys_stale_entries():
    """The rebuild re-keys entries under the repaired hashes.

    The comprehension in ``_process_state`` is load-bearing: ``dict(d)``
    would copy each entry's stored hash without calling ``__hash__``,
    leaving lookups broken even after the caches are cleared.
    """
    stale_obj, state = _stale_state()
    ctx = SimpleNamespace(train_tasks=[Task(state, set())], current_task=None)
    _rehash_objects_after_unpickle(ctx)

    fresh_obj = Object("block0", _block_type)
    assert hash(stale_obj) == hash(fresh_obj)  # Caches repaired...
    assert fresh_obj in state.data  # ...and the table re-keyed.
    assert state.data[fresh_obj][0] == pytest.approx(0.0)
    assert stale_obj in state.data


def test_rehash_makes_newly_built_dicts_consistent():
    """Dicts re-keyed by hand after the rehash serve fresh lookups."""
    stale_obj, state = _stale_state()
    ctx = SimpleNamespace(train_tasks=[Task(state, set())], current_task=None)
    _rehash_objects_after_unpickle(ctx)
    # Re-inserting under the repaired hashes (what a comprehension or
    # any post-rehash construction does) restores fresh-object lookups.
    # pylint: disable-next=unnecessary-comprehension
    rekeyed = {obj: val for obj, val in state.data.items()}
    fresh_obj = Object("block0", _block_type)
    assert fresh_obj in rekeyed
    assert rekeyed[fresh_obj][0] == pytest.approx(0.0)
    assert stale_obj in rekeyed


def test_rehash_processes_environment_task_init_obs():
    """Task-likes exposing init_obs (EnvironmentTask shape) are walked."""
    stale_obj, state = _stale_state("obs_block")
    env_task = SimpleNamespace(init_obs=state, goal_description=None)
    ctx = SimpleNamespace(train_tasks=[env_task], current_task=None)
    _rehash_objects_after_unpickle(ctx)
    assert hash(stale_obj) == hash(Object("obs_block", _block_type))


def test_rehash_handles_minimal_ctx_and_none_current_task():
    """A ctx with no tasks, states, or trajectories is a no-op."""
    ctx = SimpleNamespace(current_task=None)
    _rehash_objects_after_unpickle(ctx)  # Should not raise.


def test_rehash_is_idempotent_on_hash_caches():
    """Running the rehash twice keeps hashes consistent."""
    stale_obj, state = _stale_state()
    task = Task(state, set())
    ctx = SimpleNamespace(train_tasks=[task], current_task=task)
    _rehash_objects_after_unpickle(ctx)
    _rehash_objects_after_unpickle(ctx)
    assert hash(stale_obj) == hash(Object("block0", _block_type))
