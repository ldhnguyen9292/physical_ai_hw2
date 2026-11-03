"""The test-case format, in world-frame metres on the XZ plane."""
import json
from typing import List, NamedTuple, Tuple

WorldPoint = Tuple[float, float]


class RestrictedZone(NamedTuple):
    """A floor region designated as restricted.

    The polygon is closed: the last vertex connects to the first.
    """
    id: str
    polygon: List[WorldPoint]
    comment: str = ""


class PersonalSpace(NamedTuple):
    """An egg-shaped personal-space region around a person.

    ``phi`` is measured in degrees counterclockwise from the +X axis toward
    the +Z axis. The region extends ``f`` in the facing direction, ``b``
    behind the person, and ``s`` to either side.

    To test whether a point is inside, express its offset from the person in
    the facing and lateral directions. The point is inside when::

        (forward_offset / reach) ** 2 + (lateral_offset / s) ** 2 <= 1

    where ``reach`` is ``f`` when ``forward_offset`` is positive and ``b``
    when ``forward_offset`` is negative.
    """
    id: str
    position: WorldPoint
    phi: float
    f: float
    b: float
    s: float
    comment: str = ""



class TestCase(NamedTuple):
    """One case: where the robot starts, what it is sent to, and who and what
    is in the way."""
    id: str
    start_world: WorldPoint
    goal_object: str
    restricted_zones: List[RestrictedZone]
    people: List[PersonalSpace]
    comment: str = ""


def _parse_case(raw: dict) -> TestCase:
    return TestCase(
        id=raw["id"],
        start_world=(float(raw["start_world"][0]), float(raw["start_world"][1])),
        goal_object=raw["goal_object"],
        comment=raw.get("comment", ""),
        restricted_zones=[
            RestrictedZone(id=space["id"],
                            polygon=[(float(x), float(z)) for x, z in space["polygon"]],
                            comment=space.get("comment", ""))
            for space in raw.get("restricted_zones", [])
        ],
        people=[
            PersonalSpace(id=person["id"],
                          position=(float(person["position"][0]),
                                    float(person["position"][1])),
                          phi=float(person["phi"]),
                          f=float(person["f"]),
                          b=float(person["b"]),
                          s=float(person["s"]),
                          comment=person.get("comment", ""))
            for person in raw.get("people", [])
        ],
    )


def load_testcases(path: str) -> List[TestCase]:
    """Every test case in the file, in file order."""
    with open(path) as handle:
        return [_parse_case(case) for case in json.load(handle)["cases"]]


def select_cases(path: str, case_id: str = None) -> List[TestCase]:
    """Every test case, or just the named one."""
    cases = load_testcases(path)
    if case_id is None:
        return cases

    matching = [case for case in cases if case.id == case_id]
    if not matching:
        raise ValueError(f"Unknown test case '{case_id}'. "
                         f"Available: {', '.join(case.id for case in cases)}")
    return matching
