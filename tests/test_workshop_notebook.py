"""WORKSHOP notebook (NOTEBOOK_SPEC 2.2): carried-source integrity, package parity and negative controls."""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VALIDATOR = ROOT / "tools" / "validate_release_assets.py"
WORKSHOP = "DIMER_MultiModel_Image_Segmentation_Workshop.ipynb"


def _load_validator(root: Path):
    spec = importlib.util.spec_from_file_location("validate_release_assets_workshop", VALIDATOR)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = root
    return module


@pytest.fixture
def tree(tmp_path: Path) -> Path:
    for name in ("docs", "tutorials", "src"):
        shutil.copytree(ROOT / name, tmp_path / name, ignore=shutil.ignore_patterns("__pycache__"))
    (tmp_path / "weights" / "clipseg-rd64-refined").mkdir(parents=True)
    manifest = "weights/clipseg-rd64-refined/dimer-base-manifest.json"
    shutil.copy2(ROOT / manifest, tmp_path / manifest)
    return tmp_path


def _carried_cell(notebook: dict) -> dict:
    return next(c for c in notebook["cells"] if "".join(c["source"]).startswith("CARRIED_FILES = "))


def _edit(root: Path, mutate) -> None:
    path = root / "tutorials" / WORKSHOP
    notebook = json.loads(path.read_text(encoding="utf-8"))
    mutate(notebook)
    path.write_text(json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")


def _rewrite_carried(root: Path, name: str, old: str, new: str) -> None:
    def mutate(notebook: dict) -> None:
        cell = _carried_cell(notebook)
        text = "".join(cell["source"])
        body = ast.parse(text).body
        files = ast.literal_eval(body[0].value)
        assert old in files[name]
        files[name] = files[name].replace(old, new, 1)
        segment = ast.get_source_segment(text, body[0].value)
        cell["source"] = [text.replace(segment, repr(files), 1)]

    _edit(root, mutate)


def test_committed_workshop_passes() -> None:
    _load_validator(ROOT).validate_workshop_notebooks()


def test_carried_files_verify_and_vendor_record_matches() -> None:
    notebook = json.loads((ROOT / "tutorials" / WORKSHOP).read_text(encoding="utf-8"))
    body = ast.parse("".join(_carried_cell(notebook)["source"])).body
    files, hashes = (ast.literal_eval(node.value) for node in body[:2])
    assert {name: hashlib.sha256(text.encode()).hexdigest() for name, text in files.items()} == hashes
    vendor = json.loads(files["vendor_provenance.json"])
    # These digests were once computed on CRLF checkouts and matched no carried file.
    assert vendor["sam"]["files"]["sam_reference.py"]["sha256"] == hashes["sam_reference.py"]
    assert vendor["sam"]["files"]["sam_LICENSE"]["sha256"] == hashes["licenses/sam.txt"]
    assert vendor["sam2"]["files"]["sam2_LICENSE"]["sha256"] == hashes["licenses/sam2.txt"]


def test_control_tampered_carried_file_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _rewrite_carried(tree, "workshop.py", "import", "import  ")
    with pytest.raises(module.ValidationError, match="CARRIED_HASHES digest"):
        module.validate_workshop_notebooks()


def test_control_package_drift_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    metrics = tree / "src" / "clipseg_segmentation_pipeline" / "metrics.py"
    metrics.write_text(metrics.read_text(encoding="utf-8") + "\n# drift\n", encoding="utf-8")
    drift = "differs from src/clipseg_segmentation_pipeline/metrics.py"
    with pytest.raises(module.ValidationError, match=drift):
        module.validate_workshop_notebooks()


def test_control_stale_vendor_digest_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    notebook = json.loads((tree / "tutorials" / WORKSHOP).read_text(encoding="utf-8"))
    files = ast.literal_eval(ast.parse("".join(_carried_cell(notebook)["source"])).body[0].value)
    good = json.loads(files["vendor_provenance.json"])["sam"]["files"]["sam_LICENSE"]["sha256"]

    def mutate(nb: dict) -> None:
        cell = _carried_cell(nb)
        text = "".join(cell["source"])
        body = ast.parse(text).body
        carried, hashes = (ast.literal_eval(node.value) for node in body[:2])
        carried["vendor_provenance.json"] = carried["vendor_provenance.json"].replace(good, "0" * 64, 1)
        previous = hashes["vendor_provenance.json"]
        current = hashlib.sha256(carried["vendor_provenance.json"].encode()).hexdigest()
        hashes["vendor_provenance.json"] = current
        # Keep source.json and the metadata consistent so only the vendor record is wrong.
        carried["source.json"] = carried["source.json"].replace(previous, current)
        hashes["source.json"] = hashlib.sha256(carried["source.json"].encode()).hexdigest()
        nb["metadata"]["dimer"]["generated_from"]["files"]["vendor_provenance.json"] = current
        text = text.replace(ast.get_source_segment(text, body[0].value), repr(carried), 1)
        text = text.replace(ast.get_source_segment(text, ast.parse(text).body[1].value), repr(hashes), 1)
        cell["source"] = [text]

    _edit(tree, mutate)
    with pytest.raises(module.ValidationError, match="vendor_provenance.json digest for sam_LICENSE"):
        module.validate_workshop_notebooks()


def test_control_persisted_output_is_rejected(tree: Path) -> None:
    module = _load_validator(tree)
    _edit(tree, lambda notebook: _carried_cell(notebook).__setitem__("execution_count", 3))
    with pytest.raises(module.ValidationError, match="persists outputs"):
        module.validate_workshop_notebooks()


@pytest.mark.parametrize("key", ["notebook_profile", "notebook_mode"])
def test_control_metadata_key_is_required(tree: Path, key: str) -> None:
    module = _load_validator(tree)
    _edit(tree, lambda notebook: notebook["metadata"]["dimer"].pop(key))
    with pytest.raises(module.ValidationError, match=key):
        module.validate_workshop_notebooks()
