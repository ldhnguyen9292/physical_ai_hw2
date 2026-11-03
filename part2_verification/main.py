import argparse
import os
import sys

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__)))]

import cv2

import paths
import scene
import testcase

import check_drive
import check_rules
import check_structure
import navigator
import record_reader
import render
from report import (ITEM_NAMES, Item, reset_dir, skipped, write_lines,
                    write_report)

# The verifier's four items, by their marking numbers. Items 3 and 4 need the
# test-case geometry recomputed, which happens when the work is marked.
SHOWN = (1, 2, 5, 6)

FOOTER = ["-" * 72,
          "This result is not your grade for this case.",
          "",
          "The verifier reads your record. It checks that the record is well",
          f"formed, that it agrees with the rules you wrote in {paths.ONTOLOGY_RULES},",
          "and that the claimed path drives. It does not check whether the zones",
          "and people your edges recorded are the ones they really entered.",
          "",
          f"Items 3 ({ITEM_NAMES[2]}) and 4 ({ITEM_NAMES[3]}) check that, ",
          "and both are only checked during grading."]


def naming_notes(case, record) -> list:
    """Warnings, not items: test-case zones or people the record does not name by
    their id. Grading matches them by name, so a mismatch costs items 3 and 4."""
    notes = []
    for label, ids, rdf_class in (
            ("restricted zone", {zone.id for zone in case.restricted_zones},
             record_reader.HW2.RestrictedZone),
            ("person", {person.id for person in case.people},
             record_reader.HW2.PersonalSpace)):
        named = record_reader.named(record, rdf_class)
        notes += [f"Note: {label} '{name}' from testcases.json is not named in "
                  "your record." for name in sorted(ids - named)]
        notes += [f"Note: your record names a {label} '{name}' that testcases.json "
                  "does not define." for name in sorted(named - ids)]
    if notes:
        notes += ["Name each zone and person after its id in testcases.json.", ""]
    return notes


def verify(case, record, structure, declared, rules_path, draw_every):
    """The verifier's four items for one case, and the claimed path they read."""
    structural, chain = check_structure.check(record, structure)
    consistent = check_rules.check(record, declared, rules_path)
    if not structural.passed:
        why = "the record structure is not valid, so there is no route to drive."
        return [structural, consistent, skipped(5, why), skipped(6, why)], chain
    return ([structural, consistent]
            + check_drive.check(record, case, chain, draw_every)), chain


def summary_lines(results) -> list:
    """One row per case, so a failing case is visible at a glance."""
    widths = [len(ITEM_NAMES[number - 1]) + 2 for number in SHOWN]
    lines = ["".join(["case".ljust(10)] +
                     [ITEM_NAMES[number - 1].ljust(width)
                      for number, width in zip(SHOWN, widths)]).rstrip(),
             "-" * (10 + sum(widths))]
    for case_id, items in results:
        lines.append("".join(
            [case_id.ljust(10)] + [("PASS" if item.passed else "FAIL").ljust(width)
                                   for item, width in zip(items, widths)]).rstrip())

    failed = [case_id for case_id, items in results
              if not all(item.passed for item in items)]
    lines += ["", f"{len(results) - len(failed)} of {len(results)} cases pass all four."]
    if failed:
        lines.append(f"Needs attention: {', '.join(failed)}")
    return lines + [
        "",
        "Check each case's report.txt as well: it can carry warnings (lines",
        "starting with \"Note:\") that this table does not show.",
        "",
        "This is not your grade. Passing here means your record agrees with the",
        f"rules you wrote. Items 3 ({ITEM_NAMES[2]}) and 4 ({ITEM_NAMES[3]})",
        "are checked during grading, against the test case's own geometry."]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check a submitted record before handing it in.")
    parser.add_argument("--case", help="verify one case (default: all of them)")
    parser.add_argument("--no-draw", action="store_true",
                        help="drive without opening a window")
    parser.add_argument("--fast", action="store_true",
                        help="draw fewer frames while driving")
    parser.add_argument("--part1-output", default=paths.PART1_OUTPUT,
                        help="the submission: the map and one record per case")
    parser.add_argument("--rules", default=paths.ONTOLOGY_RULES,
                        help="the feasibility rules the record is held to")
    parser.add_argument("--part2-output", default=paths.PART2_OUTPUT,
                        help="where the reports and pictures go")
    args = parser.parse_args()

    map_path = os.path.join(args.part1_output, paths.MAP_NAME)
    if not os.path.exists(map_path):
        raise SystemExit(f"No map at {map_path}: run the planner first.")
    map_img = render.render_map(*scene.load_map(map_path), scene.PROJECTION)
    structure = record_reader.load_structure()
    declared = record_reader.load_rules(args.rules)
    draw_every = 0 if args.no_draw else (
        navigator.FAST_DRAW_EVERY if args.fast else navigator.NORMAL_DRAW_EVERY)

    results = []
    for case in testcase.select_cases(paths.TESTCASES, args.case):
        directory = reset_dir(args.part2_output, case.id)
        picture = None
        notes = []
        try:
            record = record_reader.load_record(case.id, args.part1_output)
        except Exception as problem:
            why = "the record could not be read."
            items = ([Item(1, False, f"The record could not be read: {problem}")]
                     + [skipped(number, why) for number in SHOWN[1:]])
        else:
            try:
                items, chain = verify(case, record, structure, declared,
                                      args.rules, draw_every)
                notes = naming_notes(case, record)
                picture = render.try_draw_run(
                    record, map_img, case,
                    record_reader.typed(record, record_reader.HW2.RRTTreeEdge),
                    set(chain), scene.PROJECTION)
            except Exception as problem:
                # A broken case is student feedback, not a reason to lose every
                # later case in a batch. Expected drive errors are converted to
                # items in check_drive; this is the last-resort case boundary.
                detail = str(problem).strip() or "no further details"
                why = "verification stopped safely for this case."
                items = ([Item(1, False,
                               f"Verifier error ({type(problem).__name__}): {detail}")]
                         + [skipped(number, why) for number in SHOWN[1:]])

        results.append((case.id, items))
        print(write_report(directory, "report.txt", case.id, items,
                           notes + FOOTER))
        if picture is not None:
            cv2.imwrite(os.path.join(directory, "run.png"), picture)
            print(os.path.join(directory, "run.png"))
    # Only a whole run gets a summary. Checking one case leaves the last full
    # run's view alone, and its own report.txt already says everything.
    if not args.case:
        print(write_lines(args.part2_output, "summary.txt", summary_lines(results)))


if __name__ == "__main__":
    main()
