from typing import List, Tuple

import paths
import record_reader
import report
from report import Item


def check(record, structure) -> Tuple[Item, List]:
    """Item 1, and the claimed path it read, which is empty when the item fails."""
    runs = record_reader.runs(record)
    if not runs:
        return Item(1, False, [
            "The record holds no hw2:PlanningRun. The namespace must be exactly",
            f"    {record_reader.HW2}",
            "A typo there parses as valid RDF that matches nothing."]), []

    if len(runs) > 1:
        return Item(1, False, [
            f"The record holds {len(runs)} planning runs, and one record describes "
            "one run:",
            f"    {', '.join(record_reader._name(run) for run in runs)}",
            "There is no way to tell which of them the claimed path belongs to."]), []

    malformed = record_reader.violations(record, structure)
    if malformed:
        return Item(1, False, report.violation_lines(malformed) +
                    [f"{paths.ONTOLOGY_STRUCTURE} says what a record must contain."]), []

    chain, not_a_chain = record_reader.path_chain(record)
    if not_a_chain:
        detail = [f"The claimed path is not one chain from the run's root node: "
                  f"{not_a_chain}."]
        if not record_reader.typed(record, record_reader.HW2.RRTTreeEdge):
            detail.append("The record claims no tree edges either.")
        elif not chain:
            detail.append("Recording the tree of a search that found nothing is "
                          "right, but with no route there is nothing to drive.")
        return Item(1, False, detail), []

    tree = record_reader.typed(record, record_reader.HW2.RRTTreeEdge)
    return Item(1, True,
                f"A run to '{record_reader.goal_object(record)}': "
                f"{len(record_reader.typed(record, record_reader.HW2.Edge))} candidates, "
                f"{len(tree)} kept, and a claimed path of {len(chain)} edges "
                f"from the root node."), chain
