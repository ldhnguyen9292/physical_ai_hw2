import argparse
import os
import random
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import paths
from testcase import select_cases


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
    
        print(f"{case.id} -> {record_path}" if os.path.exists(record_path)
              else f"{case.id}: nothing was written to {record_path}")


if __name__ == "__main__":
    main()
