from typing import List

import navigator
import record_reader
from report import Item, skipped


def check(record, case, chain, draw_every) -> List[Item]:
    """Items 5 and 6, from one drive of the claimed path."""
    declared = record_reader.goal_object(record)
    if declared != case.goal_object:
        wrong = (f"The record declares goalObject '{declared}', but this case asks "
                 f"for '{case.goal_object}'.")
        return [skipped(5, wrong), Item(6, False, wrong)]

    try:
        world_path = record_reader.world_path(record, chain)
        drive = navigator.drive_path(world_path, case.goal_object, draw_every)
    except ValueError as problem:
        return [Item(5, False, str(problem)), Item(6, False, str(problem))]
    except Exception as problem:
        # Habitat also raises non-ValueError exceptions for environmental
        # failures (for example, AssertionError when the scene is incomplete).
        # Report this case and let the caller continue with the rest of a batch.
        detail = str(problem).strip() or "no further details"
        message = (f"Habitat execution failed with {type(problem).__name__}: "
                   f"{detail}")
        return [Item(5, False, message), Item(6, False, message)]

    return [
        Item(5, drive.collisions == 0,
             f"{drive.collisions} collisions over {drive.steps} steps driving the "
             f"claimed path of {len(world_path)} points (the limit is 0)."),
        Item(6, drive.arrived,
             f"Stopped {drive.distance_m:.2f} m from the nearest "
             f"{case.goal_object} (the limit is "
             f"{navigator.ARRIVAL_LIMIT_M:.2f} m)."),
    ]
