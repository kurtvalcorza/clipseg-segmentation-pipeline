"""SEG2-m4: the reconstructed workshop generator reproduces the committed notebook byte for byte."""

from __future__ import annotations

import ast
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
GENERATOR = ROOT / "tools" / "build_segmentation_workshop.py"
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Image_Segmentation_Workshop.ipynb"


def _load():
    spec = importlib.util.spec_from_file_location("build_segmentation_workshop", GENERATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def build():
    return _load()


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    """A minimal copy of the repository holding everything the generator reads."""
    for rel in ("tools/build_segmentation_workshop.py", "tools/split_workshop_carrier.py"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / rel, tmp_path / rel)
    shutil.copytree(ROOT / "tools" / "segmentation_workshop", tmp_path / "tools" / "segmentation_workshop")
    shutil.copytree(ROOT / "src", tmp_path / "src")
    manifest = Path("weights/clipseg-rd64-refined/dimer-base-manifest.json")
    (tmp_path / manifest).parent.mkdir(parents=True)
    shutil.copy2(ROOT / manifest, tmp_path / manifest)
    (tmp_path / "tutorials").mkdir()
    shutil.copy2(NOTEBOOK, tmp_path / "tutorials" / NOTEBOOK.name)
    return tmp_path


def test_generator_output_equals_committed_notebook(build) -> None:
    assert build.to_bytes(build.render(ROOT)) == NOTEBOOK.read_bytes()
    assert build.main(["--check"]) == 0


def test_metadata_names_the_reconstructed_generator(build) -> None:
    generated = json.loads(NOTEBOOK.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]
    assert generated["generator"] == build.GENERATOR_VERSION
    assert generated["generator"].startswith("tools/build_segmentation_workshop.py")
    assert generated["original_generator"] == build.ORIGINAL_GENERATOR
    assert generated["base_revision"] == build.ORIGINAL_BASE
    assert "reconstruction" in generated["reconstruction"] and "2b00483" in generated["reconstruction"]
    text = NOTEBOOK.read_text(encoding="utf-8")
    assert "generator not in this repository\"" not in text


def test_derived_source_json_and_digests_match_the_carrier(build) -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    cell = next(c for c in notebook["cells"] if c["id"] == "code-03")
    body = ast.parse("".join(cell["source"])).body
    carried = ast.literal_eval(body[0].value)
    files = build.carried_files(ROOT)
    assert files == carried
    assert list(files) == [name for name, _ in build.CARRIED]
    source = json.loads(files["source.json"])
    assert source["generator"] == build.ORIGINAL_GENERATOR
    assert source["files"] == notebook["metadata"]["dimer"]["generated_from"]["files"]


def test_check_fails_when_a_cell_source_drifts(build, tree: Path) -> None:
    assert build.main(["--check", "--repo", str(tree)]) == 0
    cell = tree / "tools" / "segmentation_workshop" / "cells" / "md-17.md"
    cell.write_bytes(cell.read_bytes().replace(b"\n", b" edited\n", 1))
    assert build.main(["--check", "--repo", str(tree)]) == 1


def test_check_fails_when_a_carried_package_module_drifts(build, tree: Path) -> None:
    module = tree / "src" / "clipseg_segmentation_pipeline" / "metrics.py"
    module.write_bytes(module.read_bytes() + b"# drift\n")
    assert build.main(["--check", "--repo", str(tree)]) == 1


def test_check_fails_when_the_notebook_is_hand_edited(build, tree: Path) -> None:
    target = tree / "tutorials" / NOTEBOOK.name
    target.write_bytes(target.read_bytes().replace(b'"Candidate"', b'"Release-grade"', 1))
    assert build.main(["--check", "--repo", str(tree)]) == 1


def test_sources_are_lf_only() -> None:
    for path in (ROOT / "tools" / "segmentation_workshop").rglob("*"):
        if path.is_file():
            assert b"\r" not in path.read_bytes(), path
