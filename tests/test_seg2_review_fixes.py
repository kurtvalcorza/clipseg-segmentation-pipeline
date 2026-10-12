"""Regression tests for the 2026-10-12 review of the image-segmentation workshop (SEG2-m1..m3, m5, m6).

Like tests/test_seg_review_fixes.py, these run the notebook's own code: the carried runner ``workshop.py``
is materialised from CARRIED_FILES (and imported, or run as a child process the way the notebook runs it),
and the display helpers are executed from cell ``code-04``. Inputs are small synthetic images; no model is
loaded, so these tests check messages, exit codes and displays on CPU. They are not model or clean-runtime
evidence.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from test_seg_review_fixes import (  # noqa: F401  (the module fixture is reused by name)
    _carried,
    _cells,
    _Display,
    _helpers,
    _mask_size_mismatch,
    _write_byod,
    workshop,
)

ROOT = Path(__file__).resolve().parents[1]
RECORD_HELPERS = ("show_reload", "show_training", "show_json", "show_box_experiment")


def _code_cells() -> dict[str, str]:
    return {cid: text for cid, text in _cells().items() if cid.startswith("code-")}


# ---------------------------------------------------------------- SEG2-m1 (ENV3)


def test_environment_check_output_is_printed_by_the_kernel() -> None:
    source = _cells()["code-04"]
    calls = [
        node
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and ast.unparse(node.func) == "subprocess.run"
        and "transformers.__version__" in ast.unparse(node)
    ]
    assert len(calls) == 1
    keywords = {k.arg: ast.unparse(k.value) for k in calls[0].keywords}
    assert keywords.get("capture_output") == "True" and keywords.get("text") == "True"
    assert "check" not in keywords  # a failure prints the captured stderr before raising
    assert "print(versions.stdout.strip())" in source and "print(versions.stderr)" in source
    assert "Dependencies installed and verified" in source


# ---------------------------------------------------------------- SEG2-m2


def test_no_cell_auto_displays_a_record() -> None:
    for cid, source in _code_cells().items():
        body = ast.parse(source).body
        if body and isinstance(body[-1], ast.Expr) and isinstance(body[-1].value, ast.Call):
            assert ast.unparse(body[-1].value.func) not in RECORD_HELPERS, cid
    assert _cells()["code-12"].rstrip().endswith("parity = show_reload()")


def test_reload_display_is_one_line_that_names_the_full_record(tmp_path, capsys) -> None:
    (tmp_path / "outputs").mkdir()
    records = [
        {"id": f"r{i}", "max_abs_probability_difference": 0.0, "identical_masks": True} for i in range(92)
    ]
    parity = {"passed": True, "atol": 1e-5, "pid": 7, "metric_parity": True, "records": records}
    (tmp_path / "outputs/reload_parity.json").write_text(json.dumps(parity))
    helpers = _helpers(tmp_path)
    assert helpers["show_reload"]() == parity
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "passed: 92 records" in lines[0]
    assert lines[0].endswith(os.path.relpath(tmp_path / "outputs/reload_parity.json"))


# ---------------------------------------------------------------- SEG2-m3 (REL13)


def test_quoted_runtimes_name_their_revisions() -> None:
    opening = _cells()["md-00"]
    sentence = next(s for s in re.split(r"(?<=[.])\s+", opening) if "372 s" in s)
    assert "`a93bf44`" in sentence and "`3f94b41f0870`" in sentence and "2026-10-03" in sentence
    assert "Colab CLI" in sentence and "not re-run" in sentence
    assert "`54ae910`" in sentence and "393 s" in sentence
    assert "previous revision" not in opening


# ---------------------------------------------------------------- SEG2-m5 (KND10)


def _box_entry(expansion: float, clipped: dict, not_clipped: dict) -> dict:
    record = {
        "id": "foodseg103-val-78",
        "phrase": "biscuit",
        "iou_tight_box": 0.953,
        "iou_expanded_box": 0.002,
        "clipped_sides": [],
        "whole_image_box": False,
    }
    return {
        "expansion": expansion,
        "records": [record],
        "improved": 0,
        "unchanged": 0,
        "worsened": 1,
        "clipped_records": clipped["n"],
        "whole_image_records": 0,
        "by_clipping": {"clipped": clipped, "not_clipped": not_clipped},
    }


def test_box_experiment_shows_clipped_and_not_clipped_counts(tmp_path, monkeypatch) -> None:
    shown = _Display()
    shown.install(monkeypatch)
    (tmp_path / "outputs").mkdir()
    entries = [
        _box_entry(0.1, {"n": 9, "improved": 2, "unchanged": 0, "worsened": 7},
                   {"n": 3, "improved": 1, "unchanged": 0, "worsened": 2}),
        _box_entry(0.25, {"n": 11, "improved": 1, "unchanged": 0, "worsened": 10},
                   {"n": 1, "improved": 0, "unchanged": 0, "worsened": 1}),
    ]
    (tmp_path / "outputs/box_experiment.json").write_text(json.dumps(entries))
    _helpers(tmp_path)["show_box_experiment"]()
    table = shown.shown[-1]
    assert "Boxes clipped by an edge: improved / unchanged / worsened" in table
    assert "| 10% | 2 / 0 / 7 (of 9) | 1 / 0 / 2 (of 3) |" in table
    assert "| 25% | 1 / 0 / 10 (of 11) | 0 / 0 / 1 (of 1) |" in table
    assert "| foodseg103-val-78 | biscuit | 0.953 | 0.002 | none |" in table


def test_activity_interpretation_covers_boxes_that_are_not_clipped() -> None:
    note = _cells()["md-17"]
    assert "foodseg103-val-78" in note and "not clipped" in note and "plate" in note
    assert "enclosing object" in note and "SAM (ViT-B)" in note
    assert "not re-run in this notebook" in note and "`a93bf44`" in note  # REL13 labels on quoted numbers
    assert "narrow conclusion" in note


# ---------------------------------------------------------------- SEG2-m6 (DAT20, REL12)


@pytest.fixture
def carried_dir(tmp_path_factory) -> Path:
    directory = tmp_path_factory.mktemp("carried_runner")
    for name, text in _carried().items():
        path = directory / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    return directory


def _child_env() -> dict:
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop("PYTHONPATH", None)
    return env


def test_refused_byod_prints_one_line_through_run_stage(tmp_path, carried_dir, capsys) -> None:
    byod = _write_byod(tmp_path / "byod", _mask_size_mismatch)
    child = tmp_path / "child"
    child.mkdir()
    helpers = _helpers(carried_dir, PYTHON=Path(sys.executable), ENV=_child_env())
    assert helpers["run_stage"]("prepare", "--byod", str(byod), root=child) is False  # no RuntimeError
    shown = capsys.readouterr().out
    refusals = [line for line in shown.splitlines() if line.startswith("Refused:")]
    assert len(refusals) == 1
    assert "records[5] (id 'r05')" in refusals[0] and "run this cell again" in refusals[0]
    assert "Traceback" not in shown and "records[0]" not in shown and "ValueError" not in shown
    receipts = json.loads((child / "outputs/stage_receipts.json").read_text())
    assert receipts["stages"]["prepare"]["status"] == "refused"


def test_unreadable_manifest_is_refused_cleanly(tmp_path, carried_dir) -> None:
    byod = tmp_path / "byod"
    byod.mkdir()
    (byod / "manifest.json").write_text('{"records": [', encoding="utf-8")
    command = [sys.executable, str(carried_dir / "workshop.py"), "--root", str(tmp_path / "child"),
               "--stage", "prepare", "--byod", str(byod)]
    result = subprocess.run(command, capture_output=True, text=True, env=_child_env())
    assert result.returncode == 65
    assert result.stdout.startswith("Refused: manifest.json is not valid UTF-8 JSON") and not result.stderr


def test_genuine_prepare_failures_still_raise(tmp_path, monkeypatch, workshop) -> None:  # noqa: F811
    byod = _write_byod(tmp_path / "byod")
    argv = ["workshop.py", "--root", str(tmp_path / "run"), "--stage", "prepare", "--byod", str(byod)]
    monkeypatch.setattr(sys, "argv", argv)

    def broken(root, records, default):
        raise ValueError("not an input rule")

    monkeypatch.setattr(workshop, "persist_records", broken)
    with pytest.raises(ValueError, match="not an input rule"):
        workshop.main()
    monkeypatch.setattr(workshop, "prepare", lambda root, byod: (_ for _ in ()).throw(RuntimeError("crash")))
    with pytest.raises(RuntimeError, match="crash"):
        workshop.main()


def test_default_prepare_does_not_turn_errors_into_refusals(tmp_path, monkeypatch, workshop) -> None:  # noqa: F811
    monkeypatch.setattr(sys, "argv", ["workshop.py", "--root", str(tmp_path), "--stage", "prepare"])

    def refused_without_byod(root, byod):
        raise workshop.InputRefused("sample manifest mismatch")

    monkeypatch.setattr(workshop, "prepare", refused_without_byod)
    with pytest.raises(workshop.InputRefused):
        workshop.main()


@pytest.mark.parametrize(("child_exit", "expected"), [(65, SystemExit), (3, subprocess.CalledProcessError)])
def test_byod_stage_passes_a_refusal_on_and_keeps_failures(
    tmp_path, monkeypatch, workshop, child_exit, expected  # noqa: F811
) -> None:
    for name in ("source.json", "requirements.txt"):
        (tmp_path / name).write_text("{}")
    calls = []

    def fake_run(command, *args, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, child_exit)

    monkeypatch.setattr(workshop.subprocess, "run", fake_run)
    monkeypatch.setattr(Path, "symlink_to", lambda self, *a, **k: None)
    monkeypatch.setattr(
        sys, "argv", ["workshop.py", "--root", str(tmp_path), "--stage", "byod", "--byod", str(tmp_path)]
    )
    with pytest.raises(expected) as caught:
        workshop.main()
    assert len(calls) == 1 and calls[0][calls[0].index("--stage") + 1] == "prepare"
    if expected is SystemExit:
        assert caught.value.code == 65


def test_run_stage_reports_success_refusal_and_failure(tmp_path, capsys) -> None:
    script = "import sys\nprint('stage-line')\nsys.exit({'ok': 0, 'refused': 65}.get(sys.argv[-1], 3))\n"
    (tmp_path / "workshop.py").write_text(script)
    helpers = _helpers(tmp_path, PYTHON=Path(sys.executable), ENV=dict(os.environ))
    assert helpers["run_stage"]("ok") is True
    assert helpers["run_stage"]("refused") is False
    assert capsys.readouterr().out.count("stage-line") == 2  # the refusal's log is not repeated
    with pytest.raises(RuntimeError, match="bad failed with exit 3"):
        helpers["run_stage"]("bad")


def test_byod_cell_refuses_instead_of_raising() -> None:
    cell = _cells()["code-20"]
    assert "raise" not in cell
    assert "print('Refused: BYOD_PATH is empty." in cell
    assert "    if run_stage('byod', '--byod', str(Path(BYOD_PATH).resolve())):\n        byod_root = " in cell
    assert "`Refused:`" in _cells()["md-19"] and "run the cell again" in _cells()["md-19"]


def test_byod_cell_shows_nothing_after_a_refusal(tmp_path, monkeypatch, capsys) -> None:
    shown = _Display()
    shown.install(monkeypatch)
    namespace = {"Path": Path, "json": json, "os": os, "ROOT": tmp_path}
    namespace["run_stage"] = lambda *args, **kwargs: False
    for name in ("show_training", "show_score_table"):
        namespace[name] = lambda *a, **k: pytest.fail("results displayed after a refusal")
    source = _cells()["code-20"].replace("RUN_BYOD = False", "RUN_BYOD = True")
    source = source.replace("BYOD_PATH = ''", f"BYOD_PATH = {str(tmp_path)!r}")
    exec(compile(source, "code-20", "exec"), namespace)
    assert shown.shown == [] and capsys.readouterr().out == ""


def test_seg2_revision_is_recorded() -> None:
    meta = json.loads(
        (ROOT / "tutorials/DIMER_MultiModel_Image_Segmentation_Workshop.ipynb").read_text(encoding="utf-8")
    )["metadata"]["dimer"]
    first, last = meta["review_revisions"][-2:]
    assert last["revision"] == "0.2.1-candidate" and last["base_commit"].startswith("1e156ee")
    assert last["findings"] == ["SEG2-m1", "SEG2-m2", "SEG2-m3", "SEG2-m5", "SEG2-m6"]
    assert last["workshop_py_sha256"]["before"] == first["workshop_py_sha256"]["after"]
    assert last["workshop_py_sha256"]["after"] == meta["generated_from"]["files"]["workshop.py"]
    assert meta["release_status"] == "Candidate"


def test_carried_worker_exposes_the_refusal_contract() -> None:
    source = _carried()["workshop.py"]
    tree = ast.parse(source)
    names = {n.name for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))}
    assert {"InputRefused", "refuse"} <= names and "REFUSED_EXIT = 65" in source
    validate = source[source.index("def validate_roles"): source.index("def load_records")]
    assert "from exc" not in validate and "raise ValueError(" not in validate
