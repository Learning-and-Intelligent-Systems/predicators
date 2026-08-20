"""Scene rendering and state-manipulation helpers."""
import contextlib
import logging
import os
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np

from predicators.agent_sdk.config import ToolSurfaceConfig
from predicators.agent_sdk.tools.context import ToolContext
from predicators.settings import CFG
from predicators.structs import State


def render_scene_image(ctx: ToolContext,
                       step_label: str) -> Optional[Dict[str, Any]]:
    """Render a scene image from the pybullet env and return as content block.

    Returns an image content block dict, or None if rendering is not
    available.  Also saves the image to ``ctx.image_save_dir`` if set.
    """
    return render_pybullet_image(ctx, step_label)


@contextlib.contextmanager
def agent_render_resolution() -> Iterator[None]:
    """Scoped camera-resolution cap for agent-facing scene renders.

    While inside the block, pybullet_camera_width/height are scaled so
    the longest side is agent_sdk_image_max_px (0 disables; never
    upscales). Every image the agent views stays in its conversation for
    the rest of the session, so pixel count directly drives per-turn
    cost - and rendering at the capped size is cheaper than rendering
    full-res and resampling. Videos render outside this scope and keep
    the full camera resolution.
    """
    max_px = ToolSurfaceConfig.from_cfg().image_max_px
    old_w = CFG.pybullet_camera_width
    old_h = CFG.pybullet_camera_height
    if not max_px or max(old_w, old_h) <= max_px:
        yield
        return
    scale = max_px / max(old_w, old_h)
    CFG.pybullet_camera_width = max(1, round(old_w * scale))
    CFG.pybullet_camera_height = max(1, round(old_h * scale))
    try:
        yield
    finally:
        CFG.pybullet_camera_width = old_w
        CFG.pybullet_camera_height = old_h


def render_pybullet_image(
    ctx: ToolContext,
    step_label: str,
    state: Optional[State] = None,
) -> Optional[Dict[str, Any]]:
    """Render a pybullet scene image and return as content block.

    If *state* is provided, the env is reset to that state before
    rendering. Returns an image content block dict, or None if rendering
    is not available. Also saves the image to ``ctx.image_save_dir`` if
    set.
    """
    if ctx.env is None:
        return None
    try:
        # pylint: disable=import-outside-toplevel
        from predicators.envs.pybullet_env import PyBulletEnv
        if not isinstance(ctx.env, PyBulletEnv):
            return None
    except ImportError:
        return None

    try:
        # pylint: disable=import-outside-toplevel
        import base64
        import io

        from PIL import Image as PILImage

        if state is not None:
            ctx.env._set_state(state)  # pylint: disable=protected-access

        with agent_render_resolution():
            video = ctx.env.render()
        if not video:
            return None
        rgb_array = np.asarray(video[0], dtype=np.uint8)
        img = PILImage.fromarray(rgb_array)  # type: ignore[no-untyped-call]

        # Save to sandbox if possible
        saved_path: Optional[str] = None
        if ctx.image_save_dir:
            os.makedirs(ctx.image_save_dir, exist_ok=True)
            safe_label = step_label.replace(" ", "_").replace("/", "_")
            task_tag = (f"_task{ctx.test_task_idx:03d}"
                        if ctx.test_task_idx is not None else "")
            filename = (f"iter{ctx.iteration_id:03d}"
                        f"{task_tag}"
                        f"_test{ctx.test_call_id:03d}"
                        f"_{safe_label}.png")
            saved_path = os.path.join(ctx.image_save_dir, filename)
            img.save(saved_path)
            logging.info("Saved scene image to %s", saved_path)

        # Encode as base64 for inline return
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.standard_b64encode(buf.getvalue()).decode("ascii")
        block: Dict[str, Any] = {
            "type": "image",
            "data": b64,
            "mimeType": "image/png"
        }
        if saved_path:
            block["saved_path"] = saved_path
        return block
    except Exception as e:  # pylint: disable=broad-except
        logging.warning("Failed to render scene image: %s", e)
        return None


def draw_pybullet_annotation(annotation: Dict[str, Any],
                             physics_client_id: int) -> List[int]:
    """Draw a single annotation as a temporary visual body in PyBullet.

    Uses createVisualShape + createMultiBody so annotations render in
    getCameraImage (unlike addUserDebugLine which only shows in GUI).
    Returns a list of body IDs for cleanup via removeBody.
    """
    import pybullet as p  # pylint: disable=import-outside-toplevel

    body_ids: List[int] = []
    ann_type = annotation["type"]
    color = annotation.get("color", [1, 0, 0])
    rgba = list(color) + [1.0] if len(color) == 3 else list(color)

    if ann_type == "marker":
        pos = annotation["position"]
        size = annotation.get("size", 0.015)
        vis = p.createVisualShape(p.GEOM_SPHERE,
                                  radius=size,
                                  rgbaColor=rgba,
                                  physicsClientId=physics_client_id)
        body = p.createMultiBody(baseVisualShapeIndex=vis,
                                 basePosition=pos,
                                 physicsClientId=physics_client_id)
        body_ids.append(body)

    elif ann_type == "line":
        from_pt = np.array(annotation["from"], dtype=float)
        to_pt = np.array(annotation["to"], dtype=float)
        diff = to_pt - from_pt
        length = float(np.linalg.norm(diff))
        if length < 1e-6:
            return body_ids
        radius = annotation.get("size", 0.005)
        mid = ((from_pt + to_pt) / 2).tolist()
        # Align cylinder z-axis with line direction
        direction = diff / length
        # Quaternion from [0,0,1] to direction
        up = np.array([0.0, 0.0, 1.0])
        cross = np.cross(up, direction)
        cross_norm = float(np.linalg.norm(cross))
        dot = float(np.dot(up, direction))
        if cross_norm < 1e-6:
            quat = [0, 0, 0, 1] if dot > 0 else [1, 0, 0, 0]
        else:
            cross /= cross_norm
            angle = np.arctan2(cross_norm, dot)
            half = angle / 2
            s = np.sin(half)
            quat = [cross[0] * s, cross[1] * s, cross[2] * s, np.cos(half)]
        vis = p.createVisualShape(p.GEOM_CYLINDER,
                                  radius=radius,
                                  length=length,
                                  rgbaColor=rgba,
                                  physicsClientId=physics_client_id)
        body = p.createMultiBody(baseVisualShapeIndex=vis,
                                 basePosition=mid,
                                 baseOrientation=quat,
                                 physicsClientId=physics_client_id)
        body_ids.append(body)

    elif ann_type == "rectangle":
        min_c = annotation["min_corner"]
        max_c = annotation["max_corner"]
        z = min_c[2]
        radius = annotation.get("size", 0.005)
        corners = [
            [min_c[0], min_c[1], z],
            [max_c[0], min_c[1], z],
            [max_c[0], max_c[1], z],
            [min_c[0], max_c[1], z],
        ]
        for i in range(4):
            edge = {
                "type": "line",
                "from": corners[i],
                "to": corners[(i + 1) % 4],
                "color": color,
                "size": radius,
            }
            body_ids.extend(draw_pybullet_annotation(edge, physics_client_id))

    return body_ids


def format_object_poses(state: State) -> str:
    """Format object positions from state for diagnostic output."""
    pose_lines = []
    for obj in sorted(state, key=str):
        feats = obj.type.feature_names
        parts = [f"{obj.name}:{obj.type.name}"]
        for f in ("x", "y", "z"):
            if f in feats:
                parts.append(f"{f}={state.get(obj, f):.3f}")
        for f in ("rot", "yaw"):
            if f in feats:
                parts.append(f"{f}={state.get(obj, f):.3f}")
        if "is_held" in feats:
            parts.append(f"held={int(state.get(obj, 'is_held'))}")
        if len(parts) > 1:  # has at least one spatial feature
            pose_lines.append("  " + " ".join(parts))
    return "\n".join(pose_lines)


def apply_state_modifications(
        state: State,
        modifications: List[Dict[str, Any]]) -> Tuple[State, List[str], str]:
    """Apply ``[{object, features}]`` overrides to a copy of ``state``.

    Used by ``BeliefProbe.reset`` (the ``sim`` probe) to stage
    hypothetical states. Returns ``(modified_state, summaries, error)``;
    ``error`` is ``""`` on success.
    """
    modified_state = state.copy()
    obj_lookup = {o.name: o for o in modified_state}
    summaries: List[str] = []
    for mod in modifications:
        obj_name = mod.get("object", "")
        features = mod.get("features", {})
        if obj_name not in obj_lookup:
            available = sorted(obj_lookup.keys())
            return modified_state, summaries, (f"Unknown object '{obj_name}'. "
                                               f"Available: {available}")
        obj = obj_lookup[obj_name]
        for feat_name, value in features.items():
            try:
                modified_state.set(obj, feat_name, value)
                summaries.append(f"  {obj_name}.{feat_name} = {value}")
            except Exception as e:  # pylint: disable=broad-except
                return modified_state, summaries, (
                    f"Failed to set {obj_name}.{feat_name}: {e}")
    return modified_state, summaries, ""
