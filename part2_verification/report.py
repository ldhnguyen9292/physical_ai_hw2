import os
import shutil
from typing import List, NamedTuple, Sequence, Tuple

# The six items, in the order they are marked. Items 3 and 4 need the grader's
# rules and its own geometry, so the verifier writes 1, 2, 5 and 6 and leaves
# their numbers alone. A failed check and a lost mark then read alike.
ITEM_NAMES = ("record structure", "declared rules", "restricted zone",
              "personal space", "collision-free driving", "target reached")

WIDTH = 72


class Item(NamedTuple):
    """One numbered check: the verdict and the detail behind it."""
    number: int
    passed: bool
    detail: object          # one line, or a list of lines
    points: int = 0         # what the item is worth; set only when the work is marked

    @property
    def title(self) -> str:
        return ITEM_NAMES[self.number - 1]

    def lines(self) -> List[str]:
        """This item as report lines, with its detail indented underneath."""
        verdict = "PASS" if self.passed else "FAIL"
        if self.points:
            verdict += f" (+{self.points})"
        head = f"[{self.number}] {self.title} "
        detail = [self.detail] if isinstance(self.detail, str) else self.detail
        return ([head + "." * max(3, WIDTH - len(head) - len(verdict)) + " " + verdict]
                + [f"      {line}" for line in detail] + [""])


def skipped(number: int, why: str) -> Item:
    """An item nothing could be said about, because an earlier one failed."""
    return Item(number, False, f"Not checked: {why}")


def reset_dir(base: str, case_id: str) -> str:
    """Empty (or create) one case's folder, so it never holds two runs at once."""
    path = os.path.join(base, case_id)
    shutil.rmtree(path, ignore_errors=True)
    os.makedirs(path)
    return path


def violation_lines(violations: List[Tuple[str, List[str]]],
                    sample: int = 8) -> List[str]:
    """One block per distinct message: how many raised it, and which."""
    lines = []
    for message, names in violations:
        rest = f", and {len(names) - sample} more" if len(names) > sample else ""
        lines.append(f"- {message}")
        lines.append(f"    {len(names)} raised this: {', '.join(names[:sample])}{rest}")
    return lines


def write_lines(directory: str, name: str, lines: Sequence[str]) -> str:
    """Write lines to directory/name, and return the path written."""
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    with open(path, "w") as handle:
        handle.write("\n".join(lines).rstrip() + "\n")
    return path


def write_report(directory: str, name: str, case_id: str, items: Sequence[Item],
                 footer: Sequence[str] = ()) -> str:
    """Write one case's numbered items, and return the path written."""
    lines = [case_id, "=" * WIDTH, ""]
    for item in items:
        lines += item.lines()
    return write_lines(directory, name, lines + list(footer))
