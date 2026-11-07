import argparse
import math
import os
import random
import sys

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from pyshacl import validate
from rdflib import RDF, XSD, Graph, Literal, Namespace

import paths
from scene import PROJECTION, load_map, pixel_to_world, semantic_color
from testcase import select_cases

HW2 = Namespace("http://taica.course/hw2/ontology#")

THETA = 0.15              # obstacle clearance; must match sh:minInclusive in ontology/rules.ttl
GOAL_CLEARANCE = 0.25     # the goal must be a bit freer than THETA
GOAL_REACH = 0.7          # goal at most this far from the object (grader allows 1.0 m)
STEP = 0.3                # RRT step: a new node is at most this far from the tree
GOAL_BIAS = 0.1           # how often the random point is the goal itself
MAX_ITERS = 5000          # RRT iterations per case
OPEN_FLOOR = 0.25         # random points are drawn only where clearance >= this
MIN_OBJECT_POINTS = 15    # smaller clusters of a goal class are mislabelled stray points
OBJECT_CELL = 0.10        # grid cell (metres) used to group goal points into objects
REGION_STEP = 0.002       # spacing of the points tested against zones and personal spaces


# 1. Geometry (handout Section 2.2)
def edge_clearance(a, b, obstacles_xz, obstacle_tree):
    # F1, exact: distance from the segment a-b to the nearest obstacle point.
    # The midpoint is on the segment, so the answer is <= its nearest distance r;
    # any obstacle that close to the segment lies within |ab|/2 + r of the midpoint.
    mid = (a + b) / 2
    r, _ = obstacle_tree.query(mid)
    near = obstacles_xz[obstacle_tree.query_ball_point(mid, np.linalg.norm(b - a) / 2 + r)]
    ab = b - a
    t = np.clip((near - a) @ ab / max(ab @ ab, 1e-12), 0.0, 1.0)
    return float(np.linalg.norm(near - (a + t[:, None] * ab), axis=1).min())


def points_on_edge(a, b):
    # Points along the edge a -> b, every REGION_STEP metres, shape (n, 2)
    n = max(2, int(math.ceil(np.linalg.norm(b - a) / REGION_STEP)) + 1)
    return a + np.linspace(0.0, 1.0, n)[:, None] * (b - a)


def in_polygon(x, z, polygon):
    # Ray casting, vectorised: a ray towards +x crosses the sides an odd number
    # of times when the point is inside. x, z are numbers or arrays.
    inside = np.zeros(np.shape(x), dtype=bool)
    for (x1, z1), (x2, z2) in zip(polygon, polygon[1:] + polygon[:1]):
        crosses = (z1 > z) != (z2 > z)
        with np.errstate(divide="ignore", invalid="ignore"):
            x_cross = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
        inside ^= crosses & (x < x_cross)
    return inside


def in_personal_space(x, z, person):
    # The egg formula of testcase.py. x, z are numbers or arrays.
    phi = math.radians(person.phi)
    dx, dz = x - person.position[0], z - person.position[1]
    alpha = dx * math.cos(phi) + dz * math.sin(phi)     # forward offset
    beta = -dx * math.sin(phi) + dz * math.cos(phi)     # lateral offset
    reach = np.where(alpha >= 0, person.f, person.b)
    return (alpha / reach) ** 2 + (beta / person.s) ** 2 <= 1.0


def regions_entered(a, b, case):
    # F2 / F3: ids of the zones and of the personal spaces any edge point lies in
    x, z = points_on_edge(a, b).T
    zones = [zn.id for zn in case.restricted_zones if in_polygon(x, z, zn.polygon).any()]
    people = [p.id for p in case.people if in_personal_space(x, z, p).any()]
    return zones, people


# 2. The map as a grid of cells (used to choose the goal and the random points)
def all_map_cells(obstacle_tree):
    # Centre of every pixel of the 500 x 500 map, and its clearance
    size = PROJECTION.size
    px, pz = np.meshgrid(np.arange(size), np.arange(size))
    cx, cz = pixel_to_world(px.ravel() + 0.5, pz.ravel() + 0.5)
    clearance, _ = obstacle_tree.query(np.column_stack((cx, cz)))
    return cx, cz, clearance


def cells_outside_people(case, cx, cz):
    outside = np.ones(len(cx), dtype=bool)
    for person in case.people:
        outside &= ~in_personal_space(cx, cz, person)
    return outside


def cells_reachable_from_start(case, free, cx, cz):
    # Flood fill the free cells and keep the region that holds the start, so
    # nothing behind a wall or inside a closed area is used
    size = PROJECTION.size
    region, _ = ndimage.label(free.reshape(size, size), structure=np.ones((3, 3)))
    region = region.ravel()
    start_region = region[np.argmin(np.hypot(cx - case.start_world[0], cz - case.start_world[1]))]
    return (region == start_region) & (start_region > 0)


# 3. RDF record
def add_node(g, node_id, p):
    g.add((HW2[node_id], RDF.type, HW2.Node))
    g.add((HW2[node_id], HW2.worldX, Literal(float(p[0]), datatype=XSD.double)))
    g.add((HW2[node_id], HW2.worldZ, Literal(float(p[1]), datatype=XSD.double)))


def add_edge(g, edge_id, start_id, end_id, clearance, zones, people):
    e = HW2[edge_id]
    g.add((e, RDF.type, HW2.Edge))
    g.add((e, HW2.hasStartNode, HW2[start_id]))
    g.add((e, HW2.hasEndNode, HW2[end_id]))
    g.add((e, HW2.obstacleClearance, Literal(clearance, datatype=XSD.double)))
    for zone_id in zones:
        g.add((e, HW2.entersRestrictedZone, HW2[zone_id]))
    for person_id in people:
        g.add((e, HW2.entersPersonalSpace, HW2[person_id]))


def new_record(case, goal):
    # The run, its root and goal nodes, every zone (with containsGoal) and every person
    record, regions = Graph(), Graph()
    record.bind("hw2", HW2)
    run = HW2[f"run_{case.id}"]
    record.add((run, RDF.type, HW2.PlanningRun))
    record.add((run, HW2.goalObject, Literal(case.goal_object, datatype=XSD.string)))
    record.add((run, HW2.hasRootNode, HW2.node_root))
    record.add((run, HW2.hasGoalNode, HW2.node_goal))
    add_node(record, "node_root", case.start_world)
    add_node(record, "node_goal", goal)
    for zone in case.restricted_zones:
        regions.add((HW2[zone.id], RDF.type, HW2.RestrictedZone))
        regions.add((HW2[zone.id], HW2.containsGoal, Literal(bool(in_polygon(*goal, zone.polygon)))))
    for person in case.people:
        regions.add((HW2[person.id], RDF.type, HW2.PersonalSpace))
    record += regions
    return record, regions


# 4. RRT
def open_floor_cells(case, cells):
    # Where the RRT draws its random points: open floor (clearance >= OPEN_FLOOR)
    # reachable from the start, outside personal spaces
    cx, cz, clearance = cells
    outside = cells_outside_people(case, cx, cz)
    reachable = cells_reachable_from_start(case, (clearance >= THETA) & outside, cx, cz)
    chosen = reachable & (clearance >= OPEN_FLOOR)
    if not chosen.any():
        chosen = reachable if reachable.any() else np.ones(len(cx), dtype=bool)
    return np.column_stack((cx[chosen], cz[chosen]))


def plan_case(case, goal, obstacles_xz, obstacle_tree, random_cells, rules, rng):
    """Grow an RRT from the start to the goal. Returns (record, reached, n_edges, n_tree)."""
    record, regions = new_record(case, goal)
    nodes = np.empty((MAX_ITERS + 1, 2))        # coordinates of the accepted tree nodes
    nodes[0] = case.start_world
    ids = ["node_root"]
    parent = {}                                 # node id -> (edge id, parent node id)
    n_edges = 0

    def try_edge(start_id, a, end_id, b):
        # Measure the edge, ask SHACL (edge + zones only), record it either way
        nonlocal n_edges
        n_edges += 1
        edge_id = f"edge_{n_edges}"
        clearance = edge_clearance(a, b, obstacles_xz, obstacle_tree)
        zones, people = regions_entered(a, b, case)
        small = Graph()
        add_edge(small, edge_id, start_id, end_id, clearance, zones, people)
        accepted, _, _ = validate(small + regions, shacl_graph=rules)
        add_edge(record, edge_id, start_id, end_id, clearance, zones, people)
        if accepted:
            record.add((HW2[edge_id], RDF.type, HW2.RRTTreeEdge))
            parent[end_id] = (edge_id, start_id)
        return accepted

    half = PROJECTION.resolution / 2
    reached = False
    for it in range(1, MAX_ITERS + 1):
        # Step 1: random point p (sometimes the goal itself)
        if rng.random() < GOAL_BIAS:
            p = goal
        else:
            p = random_cells[rng.randrange(len(random_cells))] + np.array(
                [rng.uniform(-half, half), rng.uniform(-half, half)])

        # Step 2: q = the tree node closest to p
        i = int(np.argmin(np.linalg.norm(nodes[:len(ids)] - p, axis=1)))
        q, d = nodes[i], np.linalg.norm(p - nodes[i])
        if d < 1e-6:
            continue
        if p is goal and d <= STEP:             # goal within one step: try it directly
            if try_edge(ids[i], q, "node_goal", goal):
                reached = True
                break
            continue

        # Step 3: new node at most STEP from q towards p; Step 4: SHACL decides
        new = q + (p - q) * min(1.0, STEP / d)
        new_id = f"node_{it}"
        add_node(record, new_id, new)
        if not try_edge(ids[i], q, new_id, new):
            continue
        nodes[len(ids)] = new
        ids.append(new_id)

        # Step 6: close to the goal -> try to connect straight to it
        if np.linalg.norm(goal - new) <= STEP and try_edge(new_id, new, "node_goal", goal):
            reached = True
            break

    # Walk back from the goal to the root: those edges are the path
    node_id = "node_goal" if reached else "node_root"
    while node_id != "node_root":
        edge_id, node_id = parent[node_id]
        record.add((HW2[edge_id], RDF.type, HW2.PathEdge))
    return record, reached, n_edges, len(ids)


# 5. Goal selection (called by main for every case)
def real_object_points(points):
    # Group the points of one class into objects (touching cells of OBJECT_CELL
    # metres) and keep the objects with at least MIN_OBJECT_POINTS points.
    # A cluster of a few points is a mislabelled stray point, not an object.
    gx = np.floor((points[:, 0] - points[:, 0].min()) / OBJECT_CELL).astype(int)
    gz = np.floor((points[:, 1] - points[:, 1].min()) / OBJECT_CELL).astype(int)
    grid = np.zeros((gx.max() + 1, gz.max() + 1), dtype=bool)
    grid[gx, gz] = True
    objects, _ = ndimage.label(grid, structure=np.ones((3, 3)))
    object_of_point = objects[gx, gz]
    object_size = np.bincount(object_of_point)
    keep = object_size[object_of_point] >= MIN_OBJECT_POINTS
    if not keep.any():                          # nothing big enough: keep the largest
        keep = object_of_point == np.argmax(object_size[1:]) + 1
    return points[keep]


def choose_goal(case, obstacles_xz, colors, cells):
    """A free point near the goal object that the robot can reach from the start.
    Falls back to less ideal points; None only if the object is not in the map."""
    cx, cz, clearance = cells
    is_goal_class = np.all(colors == semantic_color(case.goal_object), axis=1)
    if not is_goal_class.any():                 # object not in the map at all
        return None

    # a. Distance from every cell to the real goal objects
    object_tree = cKDTree(real_object_points(obstacles_xz[is_goal_class]))
    to_object, _ = object_tree.query(np.column_stack((cx, cz)))

    # b. Cells where the robot may stop: free, close to the object, not in a personal space
    outside = cells_outside_people(case, cx, cz)
    can_stop = (clearance >= GOAL_CLEARANCE) & (to_object <= GOAL_REACH) & outside

    # c. Prefer the ones reachable from the start, trying wide corridors first
    #    and narrowing down to THETA, so the goal has the widest way in
    for width in (0.40, 0.35, 0.30, 0.25, 0.20, THETA):
        reachable = can_stop & cells_reachable_from_start(case, (clearance >= width) & outside, cx, cz)
        if reachable.any():
            can_stop = reachable
            break
    if not can_stop.any():                      # fallback: any free cell, nearest the object
        can_stop = (clearance >= THETA) & outside

    # d. Closest to the object; ties -> closest to the start
    to_start = np.hypot(cx - case.start_world[0], cz - case.start_world[1])
    idx = np.flatnonzero(can_stop)
    best = idx[np.lexsort((to_start[idx], to_object[idx]))[0]]
    return np.array([cx[best], cz[best]])


def main():
    parser = argparse.ArgumentParser(
        description="Plan each test case under the rules, and record every "
                    "candidate the search considered.")
    parser.add_argument("--case", default=None,
                        help="Test case id to plan (default: every case).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seeds each case's sampler, so a run can be repeated "
                             "(default: 0).")
    parser.add_argument("--part1-output", default=paths.PART1_OUTPUT,
                        help=f"Folder holding {paths.MAP_NAME}, and where each case's "
                             f"{paths.RECORD_NAME} is written "
                             f"(default: {paths.PART1_OUTPUT}).")
    args = parser.parse_args()

    map_path = os.path.join(args.part1_output, paths.MAP_NAME)
    if not os.path.exists(map_path):
        print(f"{map_path} not found -- run process-map first.")
        sys.exit(1)

    # Shared by every case: the map (obstacle pixels -> world metres), its
    # KD-tree, the clearance of every map cell, and the SHACL rules
    x_img, z_img, colors = load_map(map_path)
    obstacles_xz = np.column_stack(pixel_to_world(x_img, z_img))
    obstacle_tree = cKDTree(obstacles_xz)
    cells = all_map_cells(obstacle_tree)
    rules = Graph().parse(paths.ONTOLOGY_RULES, format="turtle")

    for case in select_cases(paths.TESTCASES, args.case):
        # Seeded per case, so planning one case alone repeats the batch exactly.
        rng = random.Random(f"{args.seed}:{case.id}")
        os.makedirs(os.path.join(args.part1_output, case.id), exist_ok=True)
        record_path = os.path.join(args.part1_output, case.id, paths.RECORD_NAME)
        if os.path.exists(record_path):
            os.remove(record_path)
        # TODO
        # START
        # Load the map and plan the path using RRT and write its record to record_path.
        # Draw every random choice from `rng` so the run reproduces.
        # Write the record even when the search finds no path.
        #
        # ---------------------------------- Hints ----------------------------------
        # For every candidate edge, record in the RDF graph:
        #   - minimum obstacle clearance (metres)   -> hw2:obstacleClearance
        #   - each restricted zone it enters        -> hw2:entersRestrictedZone
        #   - each personal space it enters         -> hw2:entersPersonalSpace
        #   - for every restricted zone, whether it contains the goal
        #                                           -> hw2:containsGoal
        # Then check the candidate against your ontology/rules.ttl with SHACL:
        #   - pyshacl.validate(data_graph, shacl_graph=rules_graph)
        #   - validate a small graph (the edge plus the zones it refers to), not the
        #     whole record
        #   - SHACL constraints such as sh:datatype are sensitive to the datatype of a
        #     literal. In Turtle, an unquoted 0.25 denotes an xsd:decimal, whereas
        #     "0.25"^^xsd:double denotes an xsd:double.
        # END

        # Every case gets a record, even when no path is found
        goal = choose_goal(case, obstacles_xz, colors, cells)
        if goal is None:
            # The object is not in the map: record the run (goal node at the
            # start), but claim no path
            record, _ = new_record(case, np.array(case.start_world, dtype=float))
            print(f"{case.id}: no {case.goal_object} in the map, no path planned")
        else:
            record, reached, n_edges, n_tree = plan_case(
                case, goal, obstacles_xz, obstacle_tree, open_floor_cells(case, cells), rules, rng)
            print(f"{case.id} {case.goal_object}: goal=({goal[0]:.2f}, {goal[1]:.2f}) "
                  f"reached={reached} candidates={n_edges} tree_nodes={n_tree}")
        record.serialize(destination=record_path, format="turtle")

        print(f"{case.id} -> {record_path}" if os.path.exists(record_path)
              else f"{case.id}: nothing was written to {record_path}")


if __name__ == "__main__":
    main()
