"""RoboDisco: a Gymnasium-API wrapper for predicators' PyBullet envs.

RoboDisco (Robot Model Discovery Benchmark) exposes the 15 native
``predicators.envs.pybullet_*`` environments through a standard
``gymnasium.Env`` interface so the suite can be used as a robot
model-discovery benchmark independent of the predicators planning
framework.

Quick start::

    from predicators.envs import gymnasium_wrapper as robodisco

    robodisco.register_all_environments()
    env = robodisco.make("robodisco/Blocks-v0", render_mode="rgb_array")
    obs, info = env.reset()
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
"""

import importlib
from typing import Any, Dict, List, Optional, Set, Tuple
from typing import Type as TypingType
from typing import Union

import gymnasium
import numpy as np
from gymnasium import spaces
from numpy.typing import NDArray

from predicators.envs.pybullet_env import PyBulletEnv
from predicators.structs import Action, State

# ---------------------------------------------------------------------------
# Wrapper class
# ---------------------------------------------------------------------------


def _ensure_cfg_initialized() -> None:
    """Make sure predicators' global ``CFG`` has its default fields set.

    The planning entry points (``main.py``, tests) initialize ``CFG``
    via command-line parsing before any env is constructed.  When
    RoboDisco is used as a library, that bootstrap hasn't run, so
    required fields like ``seed`` are missing and ``BaseEnv.__init__``
    would crash.
    """
    from predicators import utils  # pylint: disable=import-outside-toplevel
    from predicators.settings import \
        CFG  # pylint: disable=import-outside-toplevel
    if not hasattr(CFG, "seed"):
        utils.reset_config()


def _resolve_cls(
    env_cls: Union[str, TypingType[PyBulletEnv]], ) -> TypingType[PyBulletEnv]:
    """Resolve an env class from a string entry point such as
    ``predicators.envs.pybullet_blocks:PyBulletBlocksEnv``."""
    if isinstance(env_cls, str):
        module_path, cls_name = env_cls.rsplit(":", 1)
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)
    return env_cls


class RoboDiscoEnv(gymnasium.Env):
    """Wraps a predicators ``PyBulletEnv`` as a standard ``gymnasium.Env``.

    Observation: flattened object features as a ``Box`` space, with
    ``sim_features`` (e.g. PyBullet body ids) excluded.
    Action: robot joint action space, forwarded from the underlying env
    (converted from old ``gym.spaces.Box`` to ``gymnasium.spaces.Box``).
    Reward: sparse +1 when all goal predicates are satisfied, 0 otherwise.
    """

    metadata = {"render_modes": ["rgb_array", "human"], "render_fps": 20}

    def __init__(
        self,
        env_cls: Union[str, TypingType[PyBulletEnv]],
        render_mode: Optional[str] = None,
        cfg_overrides: Optional[Dict[str, Any]] = None,
        **env_kwargs: Any,
    ) -> None:
        super().__init__()
        self.render_mode = render_mode
        use_gui = render_mode == "human"

        _ensure_cfg_initialized()
        if cfg_overrides:
            from predicators import \
                utils  # pylint: disable=import-outside-toplevel
            utils.update_config(cfg_overrides)
        resolved_cls = _resolve_cls(env_cls)
        self._env = resolved_cls(use_gui=use_gui, **env_kwargs)

        old_as = self._env.action_space
        self.action_space = spaces.Box(low=old_as.low,
                                       high=old_as.high,
                                       dtype=np.float32)

        self._train_or_test = "train"
        self._task_idx = 0
        sample_obs = self._env.reset(self._train_or_test, self._task_idx)
        assert isinstance(sample_obs, State)
        self._obs_objects = sorted(sample_obs.data.keys(),
                                   key=lambda o: o.name)
        self._obs_features = self._build_feature_list()
        obs_dim = sum(len(feats) for feats in self._obs_features)
        self.observation_space = spaces.Box(low=-np.inf,
                                            high=np.inf,
                                            shape=(obs_dim, ),
                                            dtype=np.float32)

        self._max_episode_length = 500
        self._step_count = 0

    def _build_feature_list(self) -> List[List[str]]:
        """Get ordered non-sim feature names per object."""
        feature_lists: List[List[str]] = []
        for obj in self._obs_objects:
            feats = [
                f for f in obj.type.feature_names
                if f not in obj.type.sim_features
            ]
            feature_lists.append(feats)
        return feature_lists

    def _state_to_obs(self, state: State) -> NDArray:
        """Flatten a State into a 1-D numpy observation."""
        parts: List[float] = []
        for obj, feats in zip(self._obs_objects, self._obs_features):
            for f in feats:
                parts.append(state.get(obj, f))
        return np.array(parts, dtype=np.float32)

    def _get_info(self, state: State) -> Dict[str, Any]:
        return {
            "state": state,
            "goal_reached": self._env.goal_reached(),
        }

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[NDArray, Dict[str, Any]]:
        if seed is not None:
            self._env._set_seed(seed)  # pylint: disable=protected-access
        if options:
            self._train_or_test = options.get("train_or_test",
                                              self._train_or_test)
            self._task_idx = options.get("task_idx", self._task_idx)

        obs = self._env.reset(self._train_or_test, self._task_idx)
        assert isinstance(obs, State)
        self._step_count = 0
        return self._state_to_obs(obs), self._get_info(obs)

    def step(
            self, action: NDArray
    ) -> Tuple[NDArray, float, bool, bool, Dict[str, Any]]:
        action_obj = Action(np.array(action, dtype=np.float32))
        obs = self._env.step(action_obj)
        assert isinstance(obs, State)

        goal_reached = self._env.goal_reached()
        reward = 1.0 if goal_reached else 0.0
        terminated = goal_reached
        self._step_count += 1
        truncated = self._step_count >= self._max_episode_length

        return (
            self._state_to_obs(obs),
            reward,
            terminated,
            truncated,
            self._get_info(obs),
        )

    def render(self) -> Any:  # type: ignore[override]
        if self.render_mode == "rgb_array":
            frames = self._env.render()
            if frames:
                return np.asarray(frames[0], dtype=np.uint8)
        return None

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# Environment registry
# ---------------------------------------------------------------------------

_ENTRY_POINT = "predicators.envs.gymnasium_wrapper:RoboDiscoEnv"

# (gymnasium env id, predicators env class entry point).
_ENV_REGISTRY: List[Tuple[str, str]] = [
    ("robodisco/Ants-v0", "predicators.envs.pybullet_ants:PyBulletAntsEnv"),
    ("robodisco/Balance-v0",
     "predicators.envs.pybullet_balance:PyBulletBalanceEnv"),
    ("robodisco/Barrier-v0",
     "predicators.envs.pybullet_barrier:PyBulletBarrierEnv"),
    ("robodisco/Blocks-v0",
     "predicators.envs.pybullet_blocks:PyBulletBlocksEnv"),
    ("robodisco/Boil-v0", "predicators.envs.pybullet_boil:PyBulletBoilEnv"),
    ("robodisco/Circuit-v0",
     "predicators.envs.pybullet_circuit:PyBulletCircuitEnv"),
    ("robodisco/Coffee-v0",
     "predicators.envs.pybullet_coffee:PyBulletCoffeeEnv"),
    ("robodisco/Cover-v0", "predicators.envs.pybullet_cover:PyBulletCoverEnv"),
    ("robodisco/Domino-v0",
     "predicators.envs.pybullet_domino.env:PyBulletDominoEnv"),
    ("robodisco/Fan-v0", "predicators.envs.pybullet_fan:PyBulletFanEnv"),
    ("robodisco/Float-v0", "predicators.envs.pybullet_float:PyBulletFloatEnv"),
    ("robodisco/Grow-v0", "predicators.envs.pybullet_grow:PyBulletGrowEnv"),
    ("robodisco/Laser-v0", "predicators.envs.pybullet_laser:PyBulletLaserEnv"),
    ("robodisco/MagicBin-v0",
     "predicators.envs.pybullet_magic_bin:PyBulletMagicBinEnv"),
    ("robodisco/Switch-v0",
     "predicators.envs.pybullet_switch:PyBulletSwitchEnv"),
]

_REGISTERED = False


def register_all_environments() -> None:
    """Register every RoboDisco environment with gymnasium.

    Safe to call multiple times (idempotent).
    """
    global _REGISTERED  # pylint: disable=global-statement
    if _REGISTERED:
        return
    _REGISTERED = True
    for env_id, env_cls in _ENV_REGISTRY:
        gymnasium.register(
            id=env_id,
            entry_point=_ENTRY_POINT,
            kwargs={"env_cls": env_cls},
        )


def make(env_id: str, **kwargs: Any) -> gymnasium.Env:
    """Create a RoboDisco gymnasium environment by id.

    Automatically calls :func:`register_all_environments` if not done
    yet.
    """
    register_all_environments()
    return gymnasium.make(env_id, **kwargs)


def get_all_env_ids() -> Set[str]:
    """Return the set of all registered RoboDisco environment ids."""
    register_all_environments()
    return {eid for eid, _ in _ENV_REGISTRY}
