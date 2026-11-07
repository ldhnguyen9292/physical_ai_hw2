import argparse
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import paths
from scene import PROJECTION, load_scan, save_map, world_to_pixel, SEMANTIC_COLORS


def main():
    parser = argparse.ArgumentParser(
        description="Clean the scan into the one map all ten cases are planned on.")
    parser.add_argument("--part1-output", default=paths.PART1_OUTPUT,
                        help=f"Folder to write {paths.MAP_NAME} into "
                             f"(default: {paths.PART1_OUTPUT}).")
    args = parser.parse_args()

    coords, colors = load_scan()
    scanned = len(coords)

    # TODO
    # START
    # Remove the points that are not obstacles.
    # Assign the points you keep back to both coords and colors. They are saved
    # together, so they must stay the same length.
    #
    #   coords: shape (N, 3), world metres, y-up.
    #   colors: shape (N, 3), RGB in [0, 255]. See semantic_color for the table.
    #
    # Hints: You can start by dropping the ceiling and the floor points, and drop stray points that are not
    # really obstacles.
    # END

    # Remove floor, ceiling, mat, vent, and unknown points based on semantic color
    NOT_OBSTACLES = ['floor', 'ceiling', 'mat', 'vent']

    # Keep only points painted with a listed class that is an obstacle.
    # Anything else (e.g. unlabelled [0,0,0]) is a stray point.
    keep_colors = np.array([rgb for name, rgb in SEMANTIC_COLORS.items()
                            if name not in NOT_OBSTACLES])
    mask = (colors[:, None, :] == keep_colors[None, :, :]).all(axis=2).any(axis=1)

    # Remove points above the robot (~1.5 m tall, floor at y = -1.56)
    mask &= coords[:, 1] < 0.0
    coords = coords[mask]
    colors = colors[mask]

    # Convert the remaining points to the pixel space. Which you will plan path on in main.py.
    x_img, z_img = world_to_pixel(coords[:, 0], coords[:, 2], PROJECTION)

    os.makedirs(args.part1_output, exist_ok=True)
    destination = os.path.join(args.part1_output, paths.MAP_NAME)
    save_map(destination, x_img, z_img, colors)
    print(f"kept {len(x_img)} of {scanned} points -> {destination}")


if __name__ == "__main__":
    main()
