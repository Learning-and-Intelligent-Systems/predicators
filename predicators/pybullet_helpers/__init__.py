"""This implementation is heavily based on the pybullet-planning repository by
Caelan Garrett (https://github.com/caelan/pybullet-planning/).

In addition, the structure is loosely based off the pb_robot repository
by Rachel Holladay (https://github.com/rachelholladay/pb_robot).
"""
from typing import Any, Callable, TypeVar

import pybullet as p

_T = TypeVar("_T")


def retry_pybullet_call(fn: Callable[..., _T],
                        *args: Any,
                        retries: int = 5,
                        **kwargs: Any) -> _T:
    """Call a PyBullet API with retries on transient shared-memory errors.

    Bullet's GUI server communicates with the client over shared memory
    and occasionally drops a packet under load (especially on macOS
    Metal), surfacing as ``pybullet.error`` ("Error receiving ...", "...
    failed."). These are transient — an immediate retry typically
    succeeds.
    """
    last_err: BaseException = RuntimeError("unreachable")
    for _ in range(retries):
        try:
            return fn(*args, **kwargs)
        except p.error as e:  # type: ignore[attr-defined]
            last_err = e
    raise last_err
