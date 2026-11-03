import math
from typing import List, Set

import cv2
import numpy as np
from rdflib import RDF, Graph

import record_reader
from record_reader import HW2
from scene import PROJECTION, MapProjection, world_to_pixel
from testcase import PersonalSpace, TestCase, WorldPoint

# BGR, because OpenCV is. Chosen so the three kinds of edge are told apart at
# a glance.
REJECTED = (155, 155, 155)      # grey: dark enough to read, no hue to confuse with the map
KEPT = (0, 0, 0)
PATH = (0, 0, 255)
START = (255, 255, 0)
GOAL = (0, 0, 255)
REGION_OPEN = (120, 190, 120)       # the record claims the goal is inside it
REGION_CLOSED = (150, 150, 240)
PERSON = (200, 120, 200)
UNDERLAY_ALPHA = 0.35


def render_map(x_img: np.ndarray, z_img: np.ndarray, colors: np.ndarray,
               projection: MapProjection = PROJECTION) -> np.ndarray:
    """The cleaned obstacle map, painted onto a blank canvas."""
    image = np.full((projection.size, projection.size, 3), 255, dtype=np.uint8)
    image[z_img, x_img] = colors[:, ::-1]       # the scan is RGB, the canvas BGR
    return image


def personal_space_outline(person: PersonalSpace,
                           samples: int = 96) -> List[WorldPoint]:
    """The boundary of the egg around a person, in world metres."""
    facing = math.radians(person.phi)
    outline = []
    for index in range(samples):
        angle = 2.0 * math.pi * index / samples
        cosine, sine = math.cos(angle), math.sin(angle)
        reach = person.f if cosine >= 0.0 else person.b
        radius = 1.0 / math.sqrt((cosine / reach) ** 2 + (sine / person.s) ** 2)
        outline.append((person.position[0] + radius * math.cos(angle + facing),
                        person.position[1] + radius * math.sin(angle + facing)))
    return outline


def _shade(image: np.ndarray, overlay: np.ndarray, polygon: List[WorldPoint],
           color, projection: MapProjection) -> None:
    """Fill one social space on the underlay and outline it on the image."""
    pixels = np.array([world_to_pixel(x, z, projection) for x, z in polygon],
                      dtype=np.int32)
    cv2.fillPoly(overlay, [pixels], color)
    cv2.polylines(image, [pixels], True, color, 1)


def try_draw_run(graph, map_img, case, kept, path_edges,
                 projection: MapProjection = PROJECTION):
    """draw_run, or None when the record is too broken to draw.

    Worth attempting even when the claimed path is not a chain, which is when
    seeing the tree over the map helps most. A record that failed the structural
    shapes may be missing the coordinates the drawing needs.
    """
    try:
        return draw_run(graph, map_img, case, kept, path_edges, projection)
    # Rendering is diagnostic only. A malformed coordinate or an OpenCV error
    # must not prevent the case report (or the remaining batch) from being made.
    except Exception:
        return None


def draw_run(graph: Graph, map_img: np.ndarray, case: TestCase, kept: Set,
             path_edges: Set, projection: MapProjection = PROJECTION) -> np.ndarray:
    """The whole recorded run: rejected candidates grey, the kept tree black, the
    claimed path red, over the case's social spaces."""
    image = map_img.copy()
    overlay = image.copy()

    claims = dict(record_reader.region_claims(graph))
    for space in case.restricted_zones:
        # Coloured by what the record claims, not by the test case, so a wrong
        # claim shows as the wrong colour over the right polygon.
        _shade(image, overlay, space.polygon,
               REGION_OPEN if claims.get(space.id) else REGION_CLOSED, projection)
    for person in case.people:
        _shade(image, overlay, personal_space_outline(person), PERSON, projection)
        cv2.circle(image, world_to_pixel(*person.position, projection), 2, PERSON, -1)
    cv2.addWeighted(overlay, UNDERLAY_ALPHA, image, 1 - UNDERLAY_ALPHA, 0, dst=image)

    candidates = set(graph.subjects(RDF.type, HW2.Edge))
    for edges, color, width in ((candidates - kept, REJECTED, 1),
                                (kept - path_edges, KEPT, 1),
                                (path_edges, PATH, 2)):
        for edge in edges:
            start, end = (record_reader.node_world(graph, graph.value(edge, link))
                          for link in (HW2.hasStartNode, HW2.hasEndNode))
            cv2.line(image, world_to_pixel(*start, projection),
                     world_to_pixel(*end, projection), color, width)

    run = record_reader.run_of(graph)
    for link, color in ((HW2.hasRootNode, START), (HW2.hasGoalNode, GOAL)):
        point = record_reader.node_world(graph, graph.value(run, link))
        cv2.circle(image, world_to_pixel(*point, projection), 5, color, -1)
    return image
