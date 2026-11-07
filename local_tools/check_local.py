"""Local pre-check for machines without Habitat-Sim (e.g. Intel macOS).

Runs the verifier's items that do not need the simulator:
  1. record structure          (part2_verification/check_structure.py)
  2. consistency with rules     (part2_verification/check_rules.py)
plus the id-naming warnings and the run.png picture. Items 5 (collision-free
drive) and 6 (goal reaching) need Habitat: run `pixi run -e habitat part2` on a
linux-64 / osx-arm64 machine for those.

Not part of the submission. Usage from the repo root:
  pixi run --manifest-path macos-intel/pixi.toml python local_tools/check_local.py [--case tc_01]
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path[:0] = [ROOT, os.path.join(ROOT, "part2_verification")]

import cv2

import paths
import scene
import testcase

import check_rules
import check_structure
import record_reader
import render
from report import ITEM_NAMES, Item, reset_dir, skipped, write_report

OUT = "local_check_output"


def naming_notes(case, record):
    notes = []
    for label, ids, rdf_class in (
            ("restricted zone", {z.id for z in case.restricted_zones},
             record_reader.HW2.RestrictedZone),
            ("person", {p.id for p in case.people}, record_reader.HW2.PersonalSpace)):
        named = record_reader.named(record, rdf_class)
        notes += [f"Note: {label} '{n}' from testcases.json is not named in your record."
                  for n in sorted(ids - named)]
        notes += [f"Note: your record names a {label} '{n}' that testcases.json does not define."
                  for n in sorted(named - ids)]
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--case", help="check one case (default: all)")
    args = parser.parse_args()

    map_path = os.path.join(paths.PART1_OUTPUT, paths.MAP_NAME)
    if not os.path.exists(map_path):
        raise SystemExit(f"No map at {map_path}: run part1_planning/map_processor.py first.")
    map_img = render.render_map(*scene.load_map(map_path), scene.PROJECTION)
    structure = record_reader.load_structure()
    declared = record_reader.load_rules(paths.ONTOLOGY_RULES)

    rows = []
    for case in testcase.select_cases(paths.TESTCASES, args.case):
        directory = reset_dir(OUT, case.id)
        notes, picture = [], None
        try:
            record = record_reader.load_record(case.id, paths.PART1_OUTPUT)
        except Exception as problem:
            items = [Item(1, False, f"The record could not be read: {problem}"),
                     skipped(2, "the record could not be read.")]
        else:
            structural, chain = check_structure.check(record, structure)
            consistent = check_rules.check(record, declared, paths.ONTOLOGY_RULES)
            items = [structural, consistent]
            notes = naming_notes(case, record)
            try:
                picture = render.try_draw_run(
                    record, map_img, case,
                    record_reader.typed(record, record_reader.HW2.RRTTreeEdge),
                    set(chain), scene.PROJECTION)
            except Exception as problem:
                notes.append(f"Note: could not draw the run ({problem}).")
        write_report(directory, "report.txt", case.id, items,
                     notes + ["Items 5-6 (Habitat drive) not checked here."])
        if picture is not None:
            cv2.imwrite(os.path.join(directory, "run.png"), picture)
        rows.append((case.id, items, notes))

    print(f"{'case':8s} {ITEM_NAMES[0]:28s} {ITEM_NAMES[1]:28s} notes")
    for case_id, items, notes in rows:
        print(f"{case_id:8s} {('PASS' if items[0].passed else 'FAIL'):28s} "
              f"{('PASS' if items[1].passed else 'FAIL'):28s} {len(notes)}")
    print(f"\nDetails: {OUT}/<case>/report.txt and run.png")


if __name__ == "__main__":
    main()
