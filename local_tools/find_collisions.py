"""Where does the robot collide? (debug aid, not part of the submission)

Drives each claimed path exactly like part2_verification/navigator.drive_path,
but prints the position of every step counted as a collision. Needs Habitat-Sim
(linux-64 / osx-arm64). Usage from the repo root:
  pixi run -e habitat python local_tools/find_collisions.py --case tc_02 --case tc_05
"""
import argparse
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "part2_verification")]

import paths
import testcase
import check_structure
import navigator
import record_reader
from navigator import MOVE_M, TURN_DEG, _agent_xz, _spawn


def drive_and_log(world_path):
    sim, agent = _spawn(*world_path[0], want_sensors=False)
    hits = []
    try:
        heading = math.pi
        for leg, (target_x, target_z) in enumerate(world_path[1:], start=1):
            x, z = _agent_xz(agent)
            dx, dz = target_x - x, target_z - z
            angle = (math.atan2(dx, dz) - heading + math.pi) % (2 * math.pi) - math.pi
            turns = int(abs(math.degrees(angle)) / TURN_DEG)
            for _ in range(turns):
                sim.step("turn_left" if angle > 0 else "turn_right")
            heading += math.copysign(math.radians(turns * TURN_DEG), angle)
            for _ in range(int(math.hypot(dx, dz) / MOVE_M)):
                before = _agent_xz(agent)
                sim.step("move_forward")
                after = _agent_xz(agent)
                if math.dist(before, after) < MOVE_M / 2:
                    pos = agent.get_state().position
                    wall = sim.pathfinder.distance_to_closest_obstacle(pos, 2.0)
                    hits.append((leg, world_path[leg - 1], (target_x, target_z), after, wall))
    finally:
        sim.close()
    return hits


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", action="append", help="case id (repeatable; default: all)")
    args = parser.parse_args()
    structure = record_reader.load_structure()
    for case in testcase.load_testcases(paths.TESTCASES):
        if args.case and case.id not in args.case:
            continue
        record = record_reader.load_record(case.id, paths.PART1_OUTPUT)
        item, chain = check_structure.check(record, structure)
        if not item.passed:
            print(f"{case.id}: record structure fails, skipped")
            continue
        world_path = record_reader.world_path(record, chain)
        hits = drive_and_log(world_path)
        print(f"{case.id}: {len(hits)} collision(s) over a path of {len(world_path)} points")
        for leg, start, target, at, wall in hits:
            print(f"  leg {leg:3d}  from ({start[0]:.3f}, {start[1]:.3f}) to ({target[0]:.3f}, "
                  f"{target[1]:.3f})  stuck at ({at[0]:.3f}, {at[1]:.3f})  "
                  f"navmesh distance to obstacle {wall:.3f} m")


if __name__ == "__main__":
    main()
