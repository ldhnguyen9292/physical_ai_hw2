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

THETA = 0.15            # must match sh:minInclusive in ontology/rules.ttl
GOAL_CLEARANCE = 0.25   # the goal must be a bit freer than THETA
GOAL_REACH = 0.7        # grader allows 1.0 m from the object; keep a margin
STEP = 0.3              # RRT step size (metres)
GOAL_BIAS = 0.1         # how often the sampler picks the goal itself
MAX_ITERS = 3000        # RRT iterations per case


# Geometry
def sample_edge(a, b, step=0.002):
    # Points along the edge a -> b, every `step` metres, shape (n, 2)
    n = max(2, int(math.ceil(math.dist(a, b) / step)) + 1)
    t = np.linspace(0.0, 1.0, n)[:, None]
    return np.asarray(a, dtype=float) + t * (np.asarray(b, dtype=float) - np.asarray(a, dtype=float))


def edge_clearance(a, b, obstacle_tree):
    # Smallest distance from any point of the edge to any obstacle
    dist, _ = obstacle_tree.query(sample_edge(a, b, step=0.01))
    return float(dist.min())


def point_in_polygon(p, polygon):
    # Ray casting: a ray from p towards +x crosses the sides an odd number of times
    # when p is inside
    x, z = p
    inside = False
    for i in range(len(polygon)):
        x1, z1 = polygon[i]
        x2, z2 = polygon[(i + 1) % len(polygon)]
        if (z1 > z) != (z2 > z):
            x_cross = x1 + (z - z1) * (x2 - x1) / (z2 - z1)
            if x < x_cross:
                inside = not inside
    return inside


def point_in_personal_space(p, person):
    # The formula in testcase.py / handout Section 2.2.
    # p can be one point (x, z) or two arrays (xs, zs).
    phi = math.radians(person.phi)
    dx, dz = p[0] - person.position[0], p[1] - person.position[1]
    alpha = dx * math.cos(phi) + dz * math.sin(phi)     # forward offset
    beta = -dx * math.sin(phi) + dz * math.cos(phi)     # lateral offset
    reach = np.where(alpha >= 0, person.f, person.b)
    return (alpha / reach) ** 2 + (beta / person.s) ** 2 <= 1.0


def edge_enters_zone(a, b, polygon):
    return any(point_in_polygon(p, polygon) for p in sample_edge(a, b))


def edge_enters_personal_space(a, b, person):
    points = sample_edge(a, b)
    return bool(np.any(point_in_personal_space((points[:, 0], points[:, 1]), person)))


# Goal selection
def choose_goal(case, obstacles_xz, obstacles_tree, colors):
    # 1. Points of the goal object (same colour filter as Task 1)
    target = np.all(colors == semantic_color(case.goal_object), axis=1)
    if not target.any():
        return None
    goal_tree = cKDTree(obstacles_xz[target])

    # 2. Candidates: the centre of every pixel of the 500 x 500 map
    size = PROJECTION.size
    px, pz = np.meshgrid(np.arange(size), np.arange(size))
    cx, cz = pixel_to_world(px.ravel() + 0.5, pz.ravel() + 0.5)
    cands = np.column_stack((cx, cz))

    # 3. For every candidate: distance to any obstacle, and to the goal object
    clear, _ = obstacles_tree.query(cands)
    dgoal, _ = goal_tree.query(cands)

    # 4. Cells outside every personal space
    outside_people = np.ones(len(cands), dtype=bool)
    for person in case.people:
        outside_people &= ~point_in_personal_space((cx, cz), person)
    start_cell = np.argmin(np.hypot(cx - case.start_world[0], cz - case.start_world[1]))
    near_object = (clear >= GOAL_CLEARANCE) & (dgoal <= GOAL_REACH) & outside_people

    # 5. Keep only cells connected to the start, so the goal is not chosen on the
    #    other side of a wall or inside a closed area. Try wide corridors first
    #    (free = clearance >= width) and narrow down to THETA, so the goal is the
    #    one with the widest way in: easier for the RRT, safer for the robot.
    ok = near_object
    for width in (0.40, 0.35, 0.30, 0.25, 0.20, THETA):
        free = (clear >= width) & outside_people
        labels, _ = ndimage.label(free.reshape(size, size), structure=np.ones((3, 3)))
        labels = labels.ravel()
        reachable = near_object & (labels == labels[start_cell]) & (labels[start_cell] > 0)
        if reachable.any():
            ok = reachable
            break
    if not ok.any():
        return None

    # 6. Closest to the object; ties -> closest to the start
    dstart = np.hypot(cx - case.start_world[0], cz - case.start_world[1])
    idx = np.flatnonzero(ok)
    best = idx[np.lexsort((dstart[idx], dgoal[idx]))[0]]
    return (float(cx[best]), float(cz[best])), float(clear[best]), float(dgoal[best])


# RDF
def add_node(g, node_id, p):
    g.add((HW2[node_id], RDF.type, HW2.Node))
    g.add((HW2[node_id], HW2.worldX, Literal(float(p[0]), datatype=XSD.double)))
    g.add((HW2[node_id], HW2.worldZ, Literal(float(p[1]), datatype=XSD.double)))


def add_regions(g, case, goal_xz):
    # Every zone (with containsGoal) and every person of the case
    for zone in case.restricted_zones:
        g.add((HW2[zone.id], RDF.type, HW2.RestrictedZone))
        g.add((HW2[zone.id], HW2.containsGoal,
               Literal(point_in_polygon(goal_xz, zone.polygon))))
    for person in case.people:
        g.add((HW2[person.id], RDF.type, HW2.PersonalSpace))


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


# RRT
def plan_case(case, goal_xz, tree, bounds, rules, rng):
    """Grow an RRT from the start to goal_xz and return the RDF record."""
    record = Graph()
    record.bind("hw2", HW2)

    run = HW2[f"run_{case.id}"]
    record.add((run, RDF.type, HW2.PlanningRun))
    record.add((run, HW2.goalObject, Literal(case.goal_object, datatype=XSD.string)))
    record.add((run, HW2.hasRootNode, HW2.node_root))
    record.add((run, HW2.hasGoalNode, HW2.node_goal))
    add_node(record, "node_root", case.start_world)
    add_node(record, "node_goal", goal_xz)

    # Zones and people, needed by every edge check
    regions = Graph()
    add_regions(regions, case, goal_xz)
    record += regions

    tree_nodes = [("node_root", tuple(case.start_world))]   # nodes accepted into the tree
    parent = {}                                             # node id -> (edge id, parent node id)
    counter = {"node": 0, "edge": 0}

    def try_edge(start_id, a, end_id, b):
        # Measure one candidate edge, check it with SHACL, record it either way
        counter["edge"] += 1
        edge_id = f"edge_{counter['edge']}"
        clearance = edge_clearance(a, b, tree)
        zones = [z.id for z in case.restricted_zones if edge_enters_zone(a, b, z.polygon)]
        people = [p.id for p in case.people if edge_enters_personal_space(a, b, p)]

        # Validate a small graph: this edge plus the zones it refers to
        small = Graph()
        add_edge(small, edge_id, start_id, end_id, clearance, zones, people)
        small += regions
        conforms, _, _ = validate(small, shacl_graph=rules)

        add_edge(record, edge_id, start_id, end_id, clearance, zones, people)
        if conforms:
            record.add((HW2[edge_id], RDF.type, HW2.RRTTreeEdge))
            parent[end_id] = (edge_id, start_id)
        return conforms

    reached = False
    for _ in range(MAX_ITERS):
        # 1. Sample a point (sometimes the goal itself)
        if rng.random() < GOAL_BIAS:
            sample = goal_xz
        else:
            sample = (rng.uniform(bounds[0], bounds[1]), rng.uniform(bounds[2], bounds[3]))

        # 2. Nearest tree node
        near_id, near = min(tree_nodes, key=lambda n: math.dist(n[1], sample))
        d = math.dist(near, sample)
        if d < 1e-6:
            continue

        # 3. The goal is within one step: try to reach it directly
        if sample is goal_xz and d <= STEP:
            if try_edge(near_id, near, "node_goal", goal_xz):
                reached = True
                break
            continue

        # 4. Otherwise steer at most STEP towards the sample
        k = min(1.0, STEP / d)
        new = (near[0] + k * (sample[0] - near[0]), near[1] + k * (sample[1] - near[1]))
        counter["node"] += 1
        new_id = f"node_{counter['node']}"
        add_node(record, new_id, new)

        # 5. Keep the new node only if SHACL accepts the edge
        if not try_edge(near_id, near, new_id, new):
            continue
        tree_nodes.append((new_id, new))

        # 6. Close to the goal: try to connect straight to it
        if math.dist(new, goal_xz) <= STEP and try_edge(new_id, new, "node_goal", goal_xz):
            reached = True
            break

    # 7. Mark the root -> goal chain as PathEdge
    if reached:
        node_id = "node_goal"
        while node_id != "node_root":
            edge_id, node_id = parent[node_id]
            record.add((HW2[edge_id], RDF.type, HW2.PathEdge))

    return record, reached, counter["edge"], len(tree_nodes)


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

        # Load the map (obstacle pixels -> world metres) and the SHACL rules
        x_img, z_img, colors = load_map(map_path)
        x, z = pixel_to_world(x_img, z_img)
        obstacles_xz = np.column_stack((x, z))
        tree = cKDTree(obstacles_xz)
        bounds = (x.min(), x.max(), z.min(), z.max())
        rules = Graph().parse(paths.ONTOLOGY_RULES, format="turtle")

        goal = choose_goal(case, obstacles_xz, tree, colors)
        if goal is None:
            print(f"{case.id}: no free point found near any {case.goal_object}")
        else:
            record, reached, n_edges, n_tree = plan_case(case, goal[0], tree, bounds, rules, rng)
            record.serialize(destination=record_path, format="turtle")
            print(f"{case.id} {case.goal_object}: goal=({goal[0][0]:.2f}, {goal[0][1]:.2f}) "
                  f"reached={reached} candidates={n_edges} tree_nodes={n_tree}")

        print(f"{case.id} -> {record_path}" if os.path.exists(record_path)
              else f"{case.id}: nothing was written to {record_path}")


if __name__ == "__main__":
    main()
