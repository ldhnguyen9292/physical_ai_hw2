"""The world <-> pixel projection, this scene's semantic classes, and
reading the scan and the map.

    world -> pixel :  floor((world - origin) / resolution)
    pixel -> world :  origin + pixel * resolution
"""
import json
from typing import NamedTuple, Tuple

import numpy as np

import paths

Pixel = Tuple[int, int]
WorldPoint = Tuple[float, float]


class MapProjection(NamedTuple):
    """The world <-> pixel mapping, in occupancy-grid terms."""
    resolution: float          # metres per pixel
    origin: WorldPoint         # world coordinate of pixel (0, 0)
    size: int                  # the map is size x size pixels


PROJECTION = MapProjection(
    resolution=0.029782485701229846,
    origin=(-5.8313583476480435, -4.949014030717765),
    size=500,
)


MAP_ARRAYS = ("x_img", "z_img", "colors")


def _load_semantic_colors(path: str = paths.SEMANTIC_CLASSES):
    with open(path) as handle:
        return {name: tuple(rgb) for name, rgb in json.load(handle).items()}


SEMANTIC_COLORS = _load_semantic_colors()


def semantic_color(object_name: str) -> Tuple[int, int, int]:
    """The RGB a semantic category is painted with in the scan."""
    try:
        return SEMANTIC_COLORS[object_name.lower()]
    except (AttributeError, KeyError):
        raise ValueError(f"Unknown semantic object: {object_name}. "
                         f"Available: {sorted(SEMANTIC_COLORS)}")


def load_scan(point_path: str = paths.POINT_CLOUD_DATA,
              color_path: str = paths.COLOR_DATA):
    """The raw scan as (coordinates in metres, colours)."""
    metres_per_unit = 10000.0 / 255.0   # the .npy coordinates are stored scaled
    return np.load(point_path) * metres_per_unit, np.load(color_path)


def world_to_pixel(x, z, projection: MapProjection = PROJECTION):
    """World metres to pixel indices, for scalars or whole arrays.

    Floor in both directions, never round: rounding one way and truncating the
    other puts the same world point in two different cells.
    """
    px = np.floor((np.asarray(x) - projection.origin[0]) / projection.resolution)
    pz = np.floor((np.asarray(z) - projection.origin[1]) / projection.resolution)
    if px.ndim == 0:
        return int(px), int(pz)
    return px.astype(int), pz.astype(int)


def pixel_to_world(px, pz, projection: MapProjection = PROJECTION):
    """Pixel indices back to world metres, for scalars or whole arrays.

    The planner works in pixels and the record is in world metres, so every
    submission needs this direction.
    """
    x = projection.origin[0] + np.asarray(px) * projection.resolution
    z = projection.origin[1] + np.asarray(pz) * projection.resolution
    if x.ndim == 0:
        return float(x), float(z)
    return x, z


def save_map(path: str, x_img: np.ndarray, z_img: np.ndarray,
             colors: np.ndarray) -> None:
    """Write the cleaned obstacle map the planner planned on."""
    np.savez(path, **dict(zip(MAP_ARRAYS, (x_img, z_img, colors))))


def load_map(path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read the cleaned obstacle map back as (x_img, z_img, colors)."""
    data = np.load(path)
    return tuple(data[name] for name in MAP_ARRAYS)
