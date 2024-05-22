"""Utility for creating 2D grasp map arrays for spot environments."""
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from predicators import utils


def _main(obj_name: str,
          brush_size: int,
          matplotlib_backend: str,
          restart: bool = False,
          map_size: int = 100) -> None:

    matplotlib.use(matplotlib_backend)

    arr = np.zeros((map_size, map_size), dtype=np.uint8)
    object_outfile = utils.get_env_asset_path(
        f"grasp_maps/{obj_name}-object.npy", assert_exists=False)
    if not restart and Path(object_outfile).exists():
        outfile = utils.get_env_asset_path(f"grasp_maps/{obj_name}-grasps.npy",
                                           assert_exists=False)
        drawing_grasps = True
    else:
        outfile = object_outfile
        drawing_grasps = False

    # Create the plot
    fig, ax = plt.subplots(1, 1)
    if drawing_grasps:
        obj_map = np.load(object_outfile)
        ax.imshow(obj_map, cmap="gray", vmin=0, vmax=1)
        cmap = "coolwarm_r"
    else:
        cmap = "gray"
    im = ax.imshow(arr, cmap=cmap, vmin=0, vmax=1, alpha=0.5)
    if drawing_grasps:
        ax.set_title(f"Select valid grasp point for {obj_name}")
    else:
        ax.set_title(f"Draw an object map for {obj_name}")

    def _onmove(event: matplotlib.backend_bases.Event) -> None:
        if event.button != 1 or event.xdata is None:
            return
        clk_c, clk_r = int(event.xdata), int(event.ydata)
        for r in range(clk_r - brush_size, clk_r + brush_size):
            for c in range(clk_c - brush_size, clk_c + brush_size):
                if 0 <= r < arr.shape[0] and 0 <= c < arr.shape[1]:
                    arr[r, c] = True
        im.set_data(arr)
        fig.canvas.draw()

    def _onclose(event: matplotlib.backend_bases.Event) -> None:
        del event  # not used
        np.save(outfile, arr)
        print(f"Wrote out grasp map to {outfile}")

    fig.canvas.mpl_connect('motion_notify_event', _onmove)
    fig.canvas.mpl_connect('close_event', _onclose)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, type=str)
    parser.add_argument("--brush_size", default=5, type=int)
    parser.add_argument("--restart", action="store_true")
    parser.add_argument("--matplotlib_backend", default="MacOSX", type=str)
    args = parser.parse_args()
    _main(args.name, args.brush_size, args.matplotlib_backend, args.restart)
