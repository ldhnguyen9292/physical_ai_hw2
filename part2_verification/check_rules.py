from typing import List

import record_reader
import report
from report import Item

REQUIRED_PATHS = ("obstacleClearance", "entersRestrictedZone",
                  "entersPersonalSpace")


def unstated_constraints(declared) -> List[str]:
    """The constraints the assignment asks for that these rules never mention."""
    SH = record_reader.SH
    stated = {str(path).rsplit("#", 1)[-1] for path in declared.objects(None, SH.path)}
    missing = [name for name in REQUIRED_PATHS if name not in stated]
    if not any(target == record_reader.HW2.Edge
               for target in declared.objects(None, SH.targetClass)):
        missing.append("no shape targets hw2:Edge")
    return missing


def check(record, declared, rules_path) -> Item:
    """Item 2."""
    missing = unstated_constraints(declared)
    if missing:
        return Item(2, False, [
            f"{rules_path} does not constrain: {', '.join(missing)}.",
            "A shape that does not exist rejects nothing, so there is nothing",
            "here for the record to be consistent with."])

    rejected = record_reader.rejected_edges(record, declared)
    tree = record_reader.typed(record, record_reader.HW2.RRTTreeEdge)
    kept_but_rejected = rejected & tree
    if kept_but_rejected:
        return Item(2, False,
                    [f"Edges typed hw2:RRTTreeEdge that {rules_path} rejects:"] +
                    report.violation_lines(
                        record_reader.violations(record, declared,
                                                 focus=kept_but_rejected)))

    return Item(2, True,
                f"All {len(tree)} kept edges satisfy {rules_path}, and the "
                f"{len(rejected)} candidates it rejects were all left out of the tree.")
