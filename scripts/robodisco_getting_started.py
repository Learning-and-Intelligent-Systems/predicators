"""Smoke-test that mirrors the RoboDisco getting-started notebook.

Run::

    PYTHONHASHSEED=0 python scripts/robodisco_getting_started.py
"""

from pathlib import Path
from typing import Any, List, Optional, cast

import numpy as np
from numpy.typing import NDArray
from PIL import Image

from predicators import utils
from predicators.envs import gymnasium_wrapper as robodisco
from predicators.structs import State

OUT_DIR = Path(__file__).resolve().parent / "robodisco_getting_started_output"
OUT_DIR.mkdir(exist_ok=True)

# Apply small task counts so the smoke test runs quickly.
utils.reset_config({"num_train_tasks": 1, "num_test_tasks": 1})

# ── 1. Discover environments ─────────────────────────────────────────────

robodisco.register_all_environments()
env_ids = sorted(robodisco.get_all_env_ids())
print(f"[1/6] Found {len(env_ids)} environments")
assert len(env_ids) == 15, f"Expected 15 environments, got {len(env_ids)}"
for eid in env_ids:
    print(f"       {eid}")

# ── 2. Create environment ────────────────────────────────────────────────

env = robodisco.make("robodisco/Blocks-v0", render_mode="rgb_array")
obs, info = env.reset()
print("\n[2/6] Created robodisco/Blocks-v0")
assert isinstance(obs, np.ndarray)
assert obs.shape == env.observation_space.shape

# ── 3. Observation and action spaces ─────────────────────────────────────

obs_shape = env.observation_space.shape
act_shape = env.action_space.shape
assert obs_shape is not None and act_shape is not None
print(f"\n[3/6] Observation shape: {obs_shape}")
print(f"       Action shape:      {act_shape}")
assert len(obs_shape) == 1
assert obs_shape[0] > 0
assert len(act_shape) >= 1

action = env.action_space.sample()
obs, reward, terminated, truncated, info = env.step(action)
assert isinstance(reward, float)
assert isinstance(terminated, bool)
assert isinstance(truncated, bool)
print(f"       Step OK — reward={reward}, terminated={terminated}")

# ── 4. Structured state in info dict ─────────────────────────────────────

state = info["state"]
assert isinstance(state, State), f"Expected State, got {type(state)}"
assert "goal_reached" in info
assert isinstance(info["goal_reached"], bool)
print(f"\n[4/6] Structured state OK — goal_reached={info['goal_reached']}")
print(state.pretty_str())

# ── 5. Rendering ─────────────────────────────────────────────────────────

render_out: Optional[Any] = env.render()
assert render_out is not None, "render() returned None — rendering is broken"
frame: NDArray = cast(NDArray, render_out)
assert frame.ndim == 3, f"Expected 3D image array, got shape {frame.shape}"
assert frame.shape[2] == 3, f"Expected RGB (3 channels), got {frame.shape[2]}"
img_path = OUT_DIR / "blocks_initial.png"
Image.fromarray(frame).save(img_path)  # type: ignore[no-untyped-call]
print(f"\n[5/6] Render OK — frame shape {frame.shape}, saved to {img_path}")

# ── 6. Multi-step rollout with rendering ─────────────────────────────────

obs, info = env.reset()
frames: List[NDArray] = []
rollout_frame: Optional[Any] = env.render()
if rollout_frame is not None:
    frames.append(cast(NDArray, rollout_frame))

for _ in range(50):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    rollout_frame = env.render()
    if rollout_frame is not None:
        frames.append(cast(NDArray, rollout_frame))
    if terminated or truncated:
        break

assert len(frames) > 0, "No frames captured during rollout"

gif_path = OUT_DIR / "blocks_rollout.gif"
pil_frames = [
    Image.fromarray(f)  # type: ignore[no-untyped-call]
    for f in frames
]
pil_frames[0].save(
    gif_path,
    format="GIF",
    save_all=True,
    append_images=pil_frames[1:],
    duration=100,
    loop=0,
)
print(
    f"\n[6/6] Rollout OK — collected {len(frames)} frames, saved to {gif_path}"
)

env.close()

# ── 7. Reset every registered env (non-fatal) ────────────────────────────

print("\n[7/7] Resetting every registered env (non-fatal report):")
ok, fail = [], []
for eid in env_ids:
    try:
        e = robodisco.make(eid)
        e.reset()
        e.close()
        ok.append(eid)
        print(f"       OK   {eid}")
    except Exception as exc:  # pylint: disable=broad-except
        fail.append((eid, type(exc).__name__, str(exc)[:80]))
        print(f"       FAIL {eid}  {type(exc).__name__}: {str(exc)[:80]}")

print(f"\nSummary: {len(ok)}/{len(env_ids)} envs reset cleanly.")
if fail:
    print("Failing envs (require additional CFG to instantiate):")
    for eid, kind, msg in fail:
        print(f"  {eid}: {kind}: {msg}")

print("\nAll required checks passed!")
