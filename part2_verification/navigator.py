import math
import os
from typing import List, NamedTuple, Tuple

import cv2
import numpy as np

import paths

# Set before habitat-sim initialises, or its C++ side logs the whole GL context.
os.environ.setdefault("MAGNUM_LOG", "quiet")
os.environ.setdefault("HABITAT_SIM_LOG", "quiet")

import habitat_sim
import magnum as mn
from habitat_sim.utils.common import d3_40_colors_rgb

from testcase import WorldPoint

SCENE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                     paths.SCENE)
SENSOR_PX = 512
SENSOR_HEIGHT_M = 1.5
MOVE_M = 0.05
TURN_DEG = 1.0

NORMAL_DRAW_EVERY = 5
FAST_DRAW_EVERY = 20
ARRIVAL_LIMIT_M = 1.0
# A malformed record must not turn one grading case into an unbounded loop.
# This is far above any plausible route inside the apartment (5 km of travel).
MAX_DRIVE_STEPS = 100_000

WINDOW = "colour / depth / semantic"


class DriveResult(NamedTuple):
    """What driving a claimed path produced."""
    collisions: int
    steps: int
    arrived: bool
    distance_m: float


def _spawn(x: float, z: float,
          want_sensors: bool) -> Tuple[habitat_sim.Simulator, habitat_sim.Agent]:
    """Start the simulator with the agent standing on the navmesh at (x, z).

    The three camera sensors are what nothing but the window needs: habitat
    renders all three, at 512x512, on every single step, whether or not
    anything reads the result. Skipping them when draw_every is 0 is why a
    driven case takes a fraction of a second instead of tens of seconds.
    """
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = []
    if want_sensors:
        for uuid, kind in (("color_sensor", habitat_sim.SensorType.COLOR),
                           ("depth_sensor", habitat_sim.SensorType.DEPTH),
                           ("semantic_sensor", habitat_sim.SensorType.SEMANTIC)):
            spec = habitat_sim.CameraSensorSpec()
            spec.uuid, spec.sensor_type = uuid, kind
            spec.resolution = [SENSOR_PX, SENSOR_PX]
            spec.position = [0.0, SENSOR_HEIGHT_M, 0.0]
            spec.orientation = [0.0, 0.0, 0.0]
            spec.sensor_subtype = habitat_sim.SensorSubType.PINHOLE
            agent_cfg.sensor_specifications.append(spec)

    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=MOVE_M)),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=TURN_DEG)),
    }

    sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent_cfg]))
    try:
        agent = sim.initialize_agent(0)

        # The floor is not at y = 0 in this scene, so its height comes from the
        # navmesh.
        floor = sim.pathfinder.snap_point(mn.Vector3(x, 0.0, z))
        if not math.isfinite(floor[1]):
            raise ValueError(f"The path starts at ({x:.2f}, {z:.2f}), "
                             f"which is nowhere near the navmesh.")
        state = habitat_sim.AgentState()
        state.position = np.array([x, float(floor[1]), z])
        agent.set_state(state)
        return sim, agent
    except Exception:
        sim.close()
        raise


def _agent_xz(agent: habitat_sim.Agent) -> WorldPoint:
    position = agent.get_state().position
    return float(position[0]), float(position[2])


def _validate_path(world_path: List[WorldPoint]) -> None:
    """Reject coordinates that cannot be driven safely and in finite time."""
    if not world_path:
        raise ValueError("The claimed path has no waypoints to drive.")

    for index, point in enumerate(world_path):
        try:
            x, z = point
        except (TypeError, ValueError):
            raise ValueError(f"Waypoint {index} is not an (X, Z) coordinate.")
        if not math.isfinite(x) or not math.isfinite(z):
            raise ValueError(f"Waypoint {index} has a non-finite coordinate: "
                             f"({x}, {z}).")

    drive_steps = 0
    for start, end in zip(world_path, world_path[1:]):
        distance = math.dist(start, end)
        if not math.isfinite(distance):
            raise ValueError("A claimed path leg has a non-finite length.")
        drive_steps += int(distance / MOVE_M)
        if drive_steps > MAX_DRIVE_STEPS:
            raise ValueError(
                f"The claimed path requires more than {MAX_DRIVE_STEPS} forward "
                "steps, so the verifier will not attempt to drive it.")


def _draw(observation: dict, step: int, draw_every: int) -> None:
    """Show what the robot sees, on every draw_every-th step."""
    if not draw_every or step % draw_every:
        return
    depth = np.clip(observation["depth_sensor"] / 10.0, 0.0, 1.0) * 255.0
    panels = (cv2.cvtColor(observation["color_sensor"], cv2.COLOR_RGBA2BGR),
              cv2.cvtColor(depth.astype(np.uint8), cv2.COLOR_GRAY2BGR),
              cv2.cvtColor(d3_40_colors_rgb[observation["semantic_sensor"] % 40],
                           cv2.COLOR_RGB2BGR))
    cv2.imshow(WINDOW, np.hstack(panels))
    cv2.waitKey(1)          # never waitKey(0): it waits for a key forever


def drive_path(world_path: List[WorldPoint], goal_object: str,
               draw_every: int = NORMAL_DRAW_EVERY) -> DriveResult:
    """Drive the claimed path and report the collisions and where it ended up."""
    _validate_path(world_path)
    sim, agent = _spawn(*world_path[0], want_sensors=bool(draw_every))
    try:
        boxes = [obj.obb for obj in sim.semantic_scene.objects
                 if obj is not None and obj.category is not None
                 and obj.category.name() == goal_object]
        if not boxes:
            raise ValueError(f"The scene holds no instance of '{goal_object}'.")

        heading, collisions, steps = math.pi, 0, 0   # the agent spawns facing -Z
        for target_x, target_z in world_path[1:]:
            # Each leg is aimed from where the agent actually is, so one that
            # only partly arrived does not push the rest off course.
            x, z = _agent_xz(agent)
            dx, dz = target_x - x, target_z - z
            angle = (math.atan2(dx, dz) - heading + math.pi) % (2 * math.pi) - math.pi

            # Both counts below are truncated, never rounded: rounding lengthens
            # a leg just enough to graze scenery the truncated one clears.
            turns = int(abs(math.degrees(angle)) / TURN_DEG)
            for _ in range(turns):
                steps += 1
                _draw(sim.step("turn_left" if angle > 0 else "turn_right"),
                      steps, draw_every)
            heading += math.copysign(math.radians(turns * TURN_DEG), angle)

            for _ in range(int(math.hypot(dx, dz) / MOVE_M)):
                before = _agent_xz(agent)
                steps += 1
                _draw(sim.step("move_forward"), steps, draw_every)
                # Habitat's own collided flag also fires on a harmless brush
                # past a thinly scanned surface; getting nowhere is the test.
                if math.dist(before, _agent_xz(agent)) < MOVE_M / 2:
                    collisions += 1

        # Each box is queried at its own centre height, so the answer stays
        # horizontal: the robot stands on the floor and a worktop does not.
        x, z = _agent_xz(agent)
        on_box = [box.closest_point(mn.Vector3(x, box.center[1], z)) for box in boxes]
        distance = min(math.hypot(point[0] - x, point[2] - z) for point in on_box)
        return DriveResult(collisions, steps, distance <= ARRIVAL_LIMIT_M, distance)
    finally:
        sim.close()
        if draw_every:
            cv2.waitKey(1000)       # hold the last frame, then let go by itself
            cv2.destroyAllWindows()
            cv2.waitKey(1)
