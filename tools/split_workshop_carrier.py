#!/usr/bin/env python3
"""Rewrite the segmentation workshop's CARRIED_FILES literal as short string pieces (in-place transform).

The workshop generator (``build_segmentation_workshop.py/1``) is not in this repository, so the notebook
is maintained by anchor-checked, in-place edits (``metadata.dimer.review_revisions``, commit 0c7906b).
This transform is recorded in ``metadata.dimer.generated_from.post_generation_revisions`` instead, so the
last review revision stays the review record. It changes only the layout of the ``CARRIED_FILES = {...}``
literal: each carried file becomes a parenthesised run of implicitly concatenated string pieces, one per
source line, with long lines cut every CARRIER_PIECE characters. Python joins the pieces back into
identical text, so the carried bytes, CARRIED_HASHES, the carried source.json and generated_from.files are
unchanged. The rewrite asserts ``ast.literal_eval(new) == ast.literal_eval(old)`` before writing.

Usage (from the repository root):
    python tools/split_workshop_carrier.py           # rewrite the notebook in place
    python tools/split_workshop_carrier.py --check   # exit 1 if any notebook cell has a line over MAX_LINE
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Image_Segmentation_Workshop.ipynb"
CARRIER_PIECE = 1000
MAX_LINE = 2000
TARGET = "CARRIED_FILES"
REVISION = {
    "date": "2026-10-03",
    "reason": (
        "Source layout: no cell line over 2000 characters (the CARRIED_FILES line was 415,188 characters)"
    ),
    "base_commit": "0c7906b5d2578e1433f735c9963636d0d52bca6b",
    "cells": ["code-03"],
    "carried_files_changed": [],
    "method": (
        "tools/split_workshop_carrier.py: CARRIED_FILES re-serialised as parenthesised runs of string"
        " pieces (at most 1000 characters each); ast.literal_eval of the new literal equals the old one,"
        " so the carried text and its hashes are unchanged; generator not in this repository"
    ),
    "clean_runtime_evidence": "pending; no hosted run of this blob yet",
}


def carried_literal(files: dict[str, str]) -> str:
    """A dict literal of ``files`` with no physical line longer than about CARRIER_PIECE characters."""
    out = ["{"]
    for name, text in files.items():
        out.append(f"    {name!r}: (")
        for line in text.splitlines(keepends=True) or [""]:
            for start in range(0, max(len(line), 1), CARRIER_PIECE):
                out.append(f"        {line[start:start + CARRIER_PIECE]!r}")
        out.append("    ),")
    out.append("}")
    return "\n".join(out)


def _source(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _target_assignments(tree: ast.Module) -> list[ast.Assign]:
    return [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
        and node.targets[0].id == TARGET
    ]


def split_carrier(source: str) -> str:
    """Return ``source`` with only the CARRIED_FILES value re-serialised by ``carried_literal``."""
    nodes = _target_assignments(ast.parse(source))
    if len(nodes) != 1:
        raise SystemExit(f"expected exactly one {TARGET} assignment, found {len(nodes)}")
    value = nodes[0].value
    old = ast.literal_eval(value)
    if not isinstance(old, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in old.items()
    ):
        raise SystemExit(f"{TARGET} must be a dict of str to str")
    lines = source.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))

    def char_offset(lineno: int, col: int) -> int:
        # ast column offsets are UTF-8 byte offsets within a line; convert to character offsets.
        return offsets[lineno - 1] + len(lines[lineno - 1].encode("utf-8")[:col].decode("utf-8"))

    start = char_offset(value.lineno, value.col_offset)
    end = char_offset(value.end_lineno, value.end_col_offset)
    new_source = source[:start] + carried_literal(old) + source[end:]
    if ast.literal_eval(_target_assignments(ast.parse(new_source))[0].value) != old:
        raise SystemExit(f"{TARGET} does not round-trip")
    return new_source


def long_lines(notebook: dict) -> list[tuple[int, int]]:
    """(cell index, longest line length) for every cell with a source line over MAX_LINE characters."""
    found = []
    for index, cell in enumerate(notebook["cells"]):
        longest = max((len(line) for line in _source(cell).split("\n")), default=0)
        if longest > MAX_LINE:
            found.append((index, longest))
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help=f"exit 1 if any cell line exceeds {MAX_LINE} characters"
    )
    parser.add_argument("--notebook", type=Path, default=NOTEBOOK)
    args = parser.parse_args(argv)
    notebook = json.loads(args.notebook.read_text(encoding="utf-8"))
    if args.check:
        found = long_lines(notebook)
        for index, longest in found:
            print(f"cell {index}: line of {longest} characters (limit {MAX_LINE})", file=sys.stderr)
        print("FAIL" if found else f"OK: no cell line over {MAX_LINE} characters in {args.notebook.name}")
        return 1 if found else 0
    cells = [
        c for c in notebook["cells"] if c["cell_type"] == "code" and _source(c).startswith(f"{TARGET} = ")
    ]
    if len(cells) != 1:
        raise SystemExit(f"expected exactly one {TARGET} cell, found {len(cells)}")
    cell = cells[0]
    new_source = split_carrier(_source(cell))
    cell["source"] = new_source if isinstance(cell["source"], str) else new_source.splitlines(keepends=True)
    revisions = notebook["metadata"]["dimer"]["generated_from"].setdefault("post_generation_revisions", [])
    if REVISION not in revisions:
        revisions.append(REVISION)
    text = json.dumps(notebook, indent=1, ensure_ascii=False) + "\n"
    args.notebook.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {args.notebook.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
