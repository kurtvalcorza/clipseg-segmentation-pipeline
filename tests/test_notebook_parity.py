"""NOTEBOOK_SPEC 2.0 parity tests (PAR1–PAR3) for the standalone tutorial notebook.

The notebook carries `src/<package>/pipeline.py` verbatim; these tests fail whenever the carried
cell, the inline manifest, or the inline pins diverge from the repository at HEAD.
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOLS = ROOT / "tools"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, TOOLS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
MODULE = ROOT / "src" / TEMPLATE["package"] / "pipeline.py"
MANIFEST = ROOT / "weights" / TEMPLATE["weights_key"] / "dimer-base-manifest.json"


@pytest.fixture(scope="module")
def notebook() -> dict:
    if not NOTEBOOK.exists():
        pytest.skip(f"{NOTEBOOK.name} not generated yet")
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells(notebook: dict, cell_type: str) -> list[dict]:
    return [c for c in notebook["cells"] if c["cell_type"] == cell_type]


def _source(cell: dict) -> str:
    src = cell["source"]
    return "".join(src) if isinstance(src, list) else src


def test_par1_embedded_module_equals_repository_module(notebook: dict) -> None:
    tagged = [
        c for c in _cells(notebook, "code") if c.get("metadata", {}).get("dimer", {}).get("embedded_module")
    ]
    assert len(tagged) == 1, "exactly one cell must be tagged metadata.dimer.embedded_module"
    cell = tagged[0]
    assert cell["metadata"]["dimer"]["embedded_module"] == f"src/{TEMPLATE['package']}/pipeline.py"
    expected = build.apply_rewrites(MODULE.read_text(encoding="utf-8"), REWRITES)
    drifted = "embedded module drifted from src/; regenerate the notebook"
    assert _source(cell).rstrip("\n") + "\n" == expected, drifted


REWRITES = TEMPLATE.get("rewrites", build.REWRITES)  # a template may declare its own rules (generator /2)


def test_par1_rewrite_rules_are_the_only_difference() -> None:
    module = MODULE.read_text(encoding="utf-8")
    rewritten = build.apply_rewrites(module, REWRITES)
    diff = [(a, b) for a, b in zip(module.splitlines(), rewritten.splitlines(), strict=True) if a != b]
    assert len(diff) == len(REWRITES)
    for original, replaced in diff:
        assert "__file__" in original, original
        assert "__file__" not in replaced and "standalone rewrite" in replaced, replaced


def test_par2_inline_manifest_and_pins_match_repository(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    inline = re.search(r"^MANIFEST = (\{.*?^\})$", code, re.M | re.S)
    assert inline, "model cell must carry MANIFEST = {...}"
    assert json.loads(inline.group(1)) == manifest
    pins_block = re.search(r"^PINS = \[(.*?)^\]", code, re.M | re.S)
    assert pins_block, "install cell must carry PINS = [...]"
    inline_pins = re.findall(r"'([^']+)'", pins_block.group(1))
    assert inline_pins == build._pins(ROOT)
    meta = notebook["metadata"]["dimer"]
    assert meta["standalone"] is True
    assert meta["notebook_spec"] == build.NOTEBOOK_SPEC
    assert meta["generated_from"]["module"] == f"src/{TEMPLATE['package']}/pipeline.py"
    assert meta["generated_from"]["module_sha256"] == build.load_context(ROOT, TEMPLATE)["module_sha256"]


def test_par3_generator_check_is_clean(notebook: dict) -> None:
    # The recorded revision is a provenance label carried through the check (see build_notebook.py
    # --check); content drift is what fails this comparison.
    recorded = notebook["metadata"]["dimer"]["generated_from"]["revision"]
    rendered = build.to_bytes(build.render(ROOT, TEMPLATE, recorded))
    current = NOTEBOOK.read_bytes().replace(b"\r\n", b"\n")  # autocrlf checkouts are CRLF
    assert current == rendered, "notebook is stale; run python tools/build_notebook.py"


def test_st1_primary_path_has_no_repository_dependency(notebook: dict) -> None:
    code = "\n".join(_source(c) for c in _cells(notebook, "code"))
    assert "git" not in re.findall(r"subprocess\.run\(\[([^\]]*)\]", code).__str__()
    assert f"import {TEMPLATE['package']}" not in code
    assert f"from {TEMPLATE['package']}" not in code
    assert "github.com" not in code
