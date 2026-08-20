"""Generate procedural textures for the PyBullet studio room visuals.

Produces a warm matte wall paint under
``predicators/envs/assets/urdf/textures/``. The texture is committed so
renders stay deterministic; tweak ``WALL_BASE`` and re-run to recolor the
walls::

    python scripts/generate_room_textures.py
"""
import os

import numpy as np
from PIL import Image

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "predicators", "envs",
                       "assets", "urdf", "textures")

# Warm off-white matte paint.
WALL_BASE = np.array([0.90, 0.88, 0.84])


def _tileable_field(height: int, width: int, rng: np.random.Generator,
                    n_waves: int) -> np.ndarray:
    """A smoothly varying field in [-1, 1] that tiles seamlessly.

    Built from integer-frequency sine gratings, so it wraps with no
    visible seam when the texture repeats.
    """
    ys = np.linspace(0, 2 * np.pi, height, endpoint=False)
    xs = np.linspace(0, 2 * np.pi, width, endpoint=False)
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")
    field = np.zeros((height, width))
    for _ in range(n_waves):
        freq_x = int(rng.integers(0, 4))
        freq_y = int(rng.integers(0, 4))
        field += rng.uniform(0.3,
                             1.0) * np.sin(freq_x * grid_x + freq_y * grid_y +
                                           rng.uniform(0, 2 * np.pi))
    return field / (np.abs(field).max() + 1e-9)


def make_wall(size: int, rng: np.random.Generator) -> np.ndarray:
    """Render a clean warm matte wall paint as a uint8 RGB array."""
    img = np.ones((size, size, 3)) * WALL_BASE
    img *= (1.0 + 0.022 * _tileable_field(size, size, rng, 5))[..., None]
    img += rng.normal(0, 0.006, img.shape)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def main() -> None:
    """Generate and save the room textures."""
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(7)
    wall = make_wall(512, rng)
    wall_img = Image.fromarray(wall)  # type: ignore[no-untyped-call]
    wall_img.save(os.path.join(OUT_DIR, "wall.png"))
    print("Wrote wall.png to", os.path.normpath(OUT_DIR))


if __name__ == "__main__":
    main()
