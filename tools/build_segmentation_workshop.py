#!/usr/bin/env python3
"""Generate the multi-model image segmentation WORKSHOP notebook from repository sources — reconstruction.

``tutorials/DIMER_MultiModel_Image_Segmentation_Workshop.ipynb`` was first generated at base ``2b00483`` by
``build_segmentation_workshop.py/1``, whose source was never committed and could not be recovered.
Three edit rounds were then applied to the notebook directly (0c7906b review fixes, a93bf44 carrier layout,
the 2026-10-13 SEG2 round). This file is a reconstruction (2026-10-13, review finding SEG2-m4): it emits the
committed notebook byte for byte with all of those rounds folded in, so the notebook is again a build product
and CI can tell when the two drift apart.

Sources (all under version control):
- ``tools/segmentation_workshop/layout.json`` — notebook metadata and the cell list (ids, cell metadata, and
  where each cell's source comes from);
- ``tools/segmentation_workshop/cells/<id>.md|.py`` — one file per cell; the file holds the cell source plus a
  trailing newline, which the generator removes;
- the carried files of code-03, in CARRIED order: the CLIPSeg package and its manifest come from ``src/`` and
  ``weights/`` (the validator's parity map), the vendored SAM / SAM 2 files and the workshop-only files from
  ``tools/segmentation_workshop/carried/``, and ``source.json`` is derived from the digests of the others;
- ``tools/split_workshop_carrier.py`` ``carried_literal()`` — the CARRIED_FILES layout introduced in a93bf44.

The carried ``source.json`` keeps the original generator id (``build_segmentation_workshop.py/1``): it is
part of the carried bytes the notebook verifies at run time. ``metadata.dimer.generated_from.generator`` names
this file.

Usage (from the repository root):
    python tools/build_segmentation_workshop.py            # write the notebook
    python tools/build_segmentation_workshop.py --check    # exit 1 if the committed notebook differs
    python tools/build_segmentation_workshop.py --out PATH # write elsewhere
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Image_Segmentation_Workshop.ipynb"
SOURCES = Path("tools") / "segmentation_workshop"
GENERATOR_VERSION = "tools/build_segmentation_workshop.py/1r"
ORIGINAL_GENERATOR = "build_segmentation_workshop.py/1"
ORIGINAL_BASE = "2b004834303160e19ad2aef37b9ac7eeb996476c"
REPOSITORY = "kurtvalcorza/clipseg-segmentation-pipeline"
DIGESTS_PLACEHOLDER = "@carried-digests"
CARRIER_PLACEHOLDER = "@carrier"

# CARRIED_FILES order is part of the notebook bytes. None = derived (source.json); a path starting with
# "src/" or "weights/" is read from the package / snapshot (CRLF normalised, as the validator compares them);
# anything else lives under tools/segmentation_workshop/carried/.
CARRIED: tuple[tuple[str, str | None], ...] = (
    ("clipseg_reference/__init__.py", "src/clipseg_segmentation_pipeline/__init__.py"),
    ("clipseg_reference/metrics.py", "src/clipseg_segmentation_pipeline/metrics.py"),
    ("clipseg_reference/pipeline.py", "src/clipseg_segmentation_pipeline/pipeline.py"),
    ("clipseg_reference/samples.py", "src/clipseg_segmentation_pipeline/samples.py"),
    ("sam_reference.py", "carried/sam_reference.py"),
    ("weights/sam/dimer-base-manifest.json", "carried/weights/sam/dimer-base-manifest.json"),
    ("licenses/sam.txt", "carried/licenses/sam.txt"),
    ("sam2_reference.py", "carried/sam2_reference.py"),
    ("weights/sam2/dimer-base-manifest.json", "carried/weights/sam2/dimer-base-manifest.json"),
    ("licenses/sam2.txt", "carried/licenses/sam2.txt"),
    ("vendor_provenance.json", "carried/vendor_provenance.json"),
    ("weights/clipseg/dimer-base-manifest.json", "weights/clipseg-rd64-refined/dimer-base-manifest.json"),
    ("workshop.py", "carried/workshop.py"),
    ("sample.json", "carried/sample.json"),
    ("requirements.txt", "carried/requirements.txt"),
    ("source.json", None),
)


def _carrier_module(repo: Path):
    path = repo / "tools" / "split_workshop_carrier.py"
    spec = importlib.util.spec_from_file_location("split_workshop_carrier", path)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _read_text(path: Path) -> str:
    return path.read_bytes().decode("utf-8")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def carried_files(repo: Path) -> dict[str, str]:
    """The carried files of code-03, in CARRIED order, with source.json derived from the others."""
    files: dict[str, str] = {}
    for name, source in CARRIED:
        if source is None:
            continue
        if source.startswith(("src/", "weights/")):
            files[name] = _read_text(repo / source).replace("\r\n", "\n")
        else:
            files[name] = _read_text(repo / SOURCES / source)
    digests = {name: sha256(text) for name, text in files.items()}
    record = {
        "repository": REPOSITORY,
        "base_revision": ORIGINAL_BASE,
        "generator": ORIGINAL_GENERATOR,
        "files": digests,
    }
    files["source.json"] = json.dumps(record, indent=2)  # no trailing newline, as carried since 2b00483
    return {name: files[name] for name, _ in CARRIED}


def carrier_source(repo: Path, files: dict[str, str]) -> str:
    literal = _carrier_module(repo).carried_literal(files)
    hashes = {name: sha256(text) for name, text in files.items()}
    tail = _cell_file(repo, "cells/code-03.tail.py")
    return f"CARRIED_FILES = {literal}\nCARRIED_HASHES = {hashes!r}\n{tail}"


def _cell_file(repo: Path, rel: str) -> str:
    text = _read_text(repo / SOURCES / rel)
    if not text.endswith("\n") or text.endswith("\n\n"):
        raise SystemExit(f"{SOURCES / rel}: expected exactly one trailing newline (cell source + '\\n')")
    return text[:-1]


def render(repo: Path = ROOT) -> dict[str, Any]:
    layout = json.loads(_read_text(repo / SOURCES / "layout.json"))
    files = carried_files(repo)
    generated = layout["metadata"]["dimer"]["generated_from"]
    if generated.get("files") != DIGESTS_PLACEHOLDER:
        raise SystemExit(f"layout.json: generated_from.files must be {DIGESTS_PLACEHOLDER!r}")
    generated["files"] = {name: sha256(text) for name, text in files.items() if name != "source.json"}
    cells = []
    for entry in layout["cells"]:
        cell = dict(entry)
        ref = cell.pop("source")
        if ref == CARRIER_PLACEHOLDER:
            source = carrier_source(repo, files)
        elif isinstance(ref, str) and ref.startswith("@"):
            source = _cell_file(repo, ref[1:])
        else:
            raise SystemExit(f"layout.json: cell {cell.get('id')}: unknown source reference {ref!r}")
        cell["source"] = source.splitlines(keepends=True)
        cells.append(cell)
    layout["cells"] = cells
    return layout


def to_bytes(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if the committed notebook differs")
    parser.add_argument("--out", type=Path, default=None, help="write here instead of tutorials/")
    parser.add_argument("--repo", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    data = to_bytes(render(args.repo))
    target = NOTEBOOK if args.repo == ROOT else args.repo / NOTEBOOK.relative_to(ROOT)
    if args.check:
        committed = target.read_bytes() if target.is_file() else b""
        if committed != data:
            print(
                f"FAIL: {target.name} differs from the generator output"
                f" ({len(committed)} vs {len(data)} bytes);"
                " edit tools/segmentation_workshop/ and run python tools/build_segmentation_workshop.py",
                file=sys.stderr,
            )
            return 1
        print(f"OK: {target.name} is up to date ({len(data)} bytes)")
        return 0
    out = args.out or target
    out.write_bytes(data)
    print(f"wrote {out} ({len(data)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
