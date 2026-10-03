"""Regression tests for the 2026-10-02 review of the image-segmentation notebook (SEG-M1..M2, SEG-m1..m9).

The tests run the notebook's own code: the carried runner ``workshop.py`` is materialised from
CARRIED_FILES and imported, and the display helpers are executed from cell ``code-04``. Models are
stand-ins that return fixed masks, so these tests check bookkeeping, messages and displays on CPU;
they are not model evidence. They need numpy, Pillow and torch (CI installs CPU torch) and no SciPy,
IPython or network.
"""

from __future__ import annotations

import ast
import importlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Image_Segmentation_Workshop.ipynb"
ROLES = ["train"] * 8 + ["validation"] * 2 + ["test"] * 2 + ["activity"]
SIZE = (48, 40)  # width, height


def _notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _cells() -> dict[str, str]:
    return {c["id"]: "".join(c["source"]) for c in _notebook()["cells"]}


def _carried() -> dict[str, str]:
    text = _cells()["code-03"]
    return ast.literal_eval(ast.parse(text).body[0].value)


@pytest.fixture(scope="module")
def workshop(tmp_path_factory):
    source = tmp_path_factory.mktemp("carried")
    for name, text in _carried().items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    sys.path.insert(0, str(source))
    for name in [m for m in sys.modules if m == "workshop" or m.startswith("clipseg_reference")]:
        del sys.modules[name]
    module = importlib.import_module("workshop")
    module.CARRIED_ROOT = source
    yield module
    sys.path.remove(str(source))
    for name in [m for m in sys.modules if m == "workshop" or m.startswith("clipseg_reference")]:
        del sys.modules[name]


def _mask(index: int, *, left_edge: bool = False) -> np.ndarray:
    mask = np.zeros((SIZE[1], SIZE[0]), dtype=bool)
    x0 = 0 if left_edge else 6 + index % 3
    mask[8 + index % 4 : 30, x0 : 26 + index % 5] = True
    return mask


def _box(mask: np.ndarray) -> list[int]:
    ys, xs = np.nonzero(mask)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def _make_run(root: Path) -> list[dict]:
    """A complete prepared run (13 records, 8/2/2/1) without SciPy: images, masks, records and prompts."""
    rng = np.random.default_rng(0)
    records, prompts = [], []
    for index, role in enumerate(ROLES):
        rid = f"r{index:02d}"
        mask = _mask(index, left_edge=role == "activity")
        image_path, mask_path = root / f"data/images/{rid}.png", root / f"data/masks/{rid}.png"
        image_path.parent.mkdir(parents=True, exist_ok=True)
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)).save(image_path)
        Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
        records.append(
            {
                "id": rid,
                "image": f"data/images/{rid}.png",
                "mask": f"data/masks/{rid}.png",
                "prompt": "square",
                "role": role,
                "group": rid,
            }
        )
        ys, xs = np.nonzero(mask)
        prompts.append(
            {
                "id": rid,
                "role": role,
                "text": "square",
                "point": [int(xs[0]), int(ys[0])],
                "box": _box(mask),
                "components": 1,
            }
        )
    (root / "outputs").mkdir(parents=True, exist_ok=True)
    (root / "data/records.json").write_text(json.dumps({"records": records, "private_byod": False}))
    (root / "outputs/prompt_manifest.json").write_text(json.dumps({"records": prompts}))
    return records


class _StandInSam:
    """Three candidates (a fixed square, the whole image, nothing); predicted quality favours the square."""

    @classmethod
    def from_pretrained(cls, **_kwargs):
        return cls()

    def segment(self, image, multimask=True, **_kwargs):
        w, h = image.size
        square = np.zeros((h, w), dtype=bool)
        square[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = True
        return {
            "masks": [square, np.ones((h, w), bool), np.zeros((h, w), bool)],
            "iou_scores": [0.9, 0.5, 0.1],
        }


@pytest.fixture
def stand_ins(monkeypatch, workshop):
    import torch

    for name, cls in (
        ("sam_reference", "SAMViTSegmentationPipeline"),
        ("sam2_reference", "SAM2SegmentationPipeline"),
    ):
        module = types.ModuleType(name)
        setattr(module, cls, _StandInSam)
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(workshop, "device", lambda: "cpu")
    monkeypatch.setattr(workshop, "timed", lambda call: (call(), 0.0))
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda *a, **k: 0)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda *a, **k: "stand-in")


# ---------------------------------------------------------------- SEG-M1


def test_box_change_names_clipped_sides_and_whole_image_boxes(workshop) -> None:
    base = [0, 10, 20, 30]
    actual = workshop.expand_box(base, (40, 40), 0.25)
    change = workshop.box_change(base, actual, (40, 40), 0.25)
    assert change["requested_expansion_pixels"] == [5.0, 5.0, 5.0, 5.0]
    assert change["realised_expansion_pixels"] == [0.0, 5.0, 5.0, 5.0]
    assert change["clipped_sides"] == ["left"]
    assert change["whole_image_box"] is False
    full = workshop.box_change(
        [0, 0, 40, 40], workshop.expand_box([0, 0, 40, 40], (40, 40), 0.1), (40, 40), 0.1
    )
    assert full["clipped_sides"] == ["left", "top", "right", "bottom"] and full["whole_image_box"] is True
    tight = workshop.box_change(base, workshop.expand_box(base, (40, 40), 0.0), (40, 40), 0.0)
    assert tight["clipped_sides"] == [] and tight["realised_expansion_pixels"] == [0.0, 0.0, 0.0, 0.0]


def test_box_experiment_records_the_realised_boxes(tmp_path, workshop, stand_ins) -> None:
    _make_run(tmp_path)
    workshop.sam_stage(tmp_path, "sam2")
    entries = json.loads((tmp_path / "outputs/box_experiment.json").read_text())
    assert [e["expansion"] for e in entries] == [0.1, 0.25]
    for entry in entries:
        (record,) = entry["records"]  # one activity image, whose target touches the left edge
        for key in (
            "clipped_sides",
            "whole_image_box",
            "realised_expansion_pixels",
            "iou_tight_box",
            "iou_expanded_box",
            "phrase",
            "box_area_fraction",
        ):
            assert key in record
        assert "left" in record["clipped_sides"]
        assert entry["clipped_records"] == 1 and isinstance(entry["whole_image_records"], int)
        split = entry["by_clipping"]
        assert split["clipped"]["n"] + split["not_clipped"]["n"] == len(entry["records"])
        assert entry["improved"] + entry["unchanged"] + entry["worsened"] == len(entry["records"])
    rows = json.loads((tmp_path / "outputs/controlled_experiment.json").read_text())
    assert all(r["actual_expansion_pixels"] == r["realised_expansion_pixels"] for r in rows)
    assert [r["clipped_sides"] for r in rows if r["protocol"] == "box_expand_0.0"] == [[]]


# ---------------------------------------------------------------- SEG-m1


def _write_byod(directory: Path, mutate=None) -> Path:
    (directory / "images").mkdir(parents=True)
    (directory / "masks").mkdir(parents=True)
    rng = np.random.default_rng(1)
    records = []
    for index, role in enumerate(ROLES):
        Image.fromarray(rng.integers(0, 255, (SIZE[1], SIZE[0], 3), dtype=np.uint8)).save(
            directory / f"images/{index:02d}.png"
        )
        Image.fromarray(_mask(index).astype(np.uint8) * 255).save(directory / f"masks/{index:02d}.png")
        records.append(
            {
                "id": f"r{index:02d}",
                "image": f"images/{index:02d}.png",
                "mask": f"masks/{index:02d}.png",
                "prompt": "square",
                "role": role,
                "group_id": f"g{index:02d}",
            }
        )
    if mutate is not None:
        mutate(directory, records)
    (directory / "manifest.json").write_text(json.dumps({"records": records}))
    return directory


def _mask_size_mismatch(directory, records):
    Image.fromarray(np.full((20, 20), 255, np.uint8)).save(directory / "masks/05.png")


def _missing_prompt(directory, records):
    del records[7]["prompt"]


def _missing_mask_key(directory, records):
    del records[4]["mask"]


def _tiny_image(directory, records):
    Image.fromarray(np.zeros((8, 8, 3), np.uint8)).save(directory / "images/03.png")


def _missing_file(directory, records):
    records[9]["image"] = "images/absent.png"


@pytest.mark.parametrize(
    ("mutate", "index"),
    [
        (_mask_size_mismatch, 5),
        (_missing_prompt, 7),
        (_missing_mask_key, 4),
        (_tiny_image, 3),
        (_missing_file, 9),
    ],
)
def test_byod_errors_name_the_offending_record(tmp_path, monkeypatch, workshop, mutate, index) -> None:
    monkeypatch.setattr(
        workshop, "persist_records", lambda root, records, default: workshop.validate_roles(records)
    )
    byod = _write_byod(tmp_path / "byod", mutate)
    with pytest.raises(ValueError) as caught:
        workshop.prepare(tmp_path / "run", byod)
    message = str(caught.value)
    assert f"records[{index}] (id 'r{index:02d}')" in message
    if index != 0:
        assert "records[0]" not in message


def test_valid_byod_still_passes_validation(tmp_path, monkeypatch, workshop) -> None:
    seen = {}

    def capture(root, records, default):
        seen.update(workshop.validate_roles(records, default=default))

    monkeypatch.setattr(workshop, "persist_records", capture)
    workshop.prepare(tmp_path / "run", _write_byod(tmp_path / "byod"))
    assert {role: len(v) for role, v in seen.items()} == {
        "train": 8,
        "validation": 2,
        "test": 2,
        "activity": 1,
    }


def test_byod_without_manifest_is_explained(tmp_path, workshop) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="containing manifest.json"):
        workshop.prepare(tmp_path / "run", tmp_path / "empty")


# ---------------------------------------------------------------- SEG-m3


def _complete_run(root: Path, workshop) -> None:
    records = _make_run(root)
    workshop.sam_stage(root, "sam")
    workshop.sam_stage(root, "sam2")
    by_id = {r["id"]: r for r in records}
    baselines, frozen, reloaded, parity = [], [], [], []
    for rid, record in by_id.items():
        if record["role"] == "train":
            continue
        reference = np.asarray(Image.open(root / record["mask"])) > 0
        loaded = {
            "id": rid,
            "role": record["role"],
            "image": Image.open(root / record["image"]).convert("RGB"),
        }
        box = _box(reference)
        filled = np.zeros_like(reference)
        filled[box[1] : box[3], box[0] : box[2]] = True
        for name, mask in (("empty", np.zeros_like(reference)), ("oracle_filled_box", filled)):
            protocol = "unassisted" if name == "empty" else "oracle_box"
            baselines.append(
                {
                    "id": rid,
                    "role": record["role"],
                    "model": name,
                    "protocol": protocol,
                    **workshop.overlap(mask, reference),
                }
            )
        # The adapted model improves test image r10 and leaves test image r11 unchanged.
        shifted = np.roll(reference, 3, axis=1)
        for name, mask, rows in (
            ("clipseg_frozen", shifted, frozen),
            ("clipseg_reloaded", reference if rid == "r10" else shifted, reloaded),
        ):
            workshop.save_prediction(root, name, loaded, mask)
            rows.append(
                {
                    "id": rid,
                    "role": record["role"],
                    "model": name,
                    "protocol": "text",
                    "seconds": 0.0,
                    **workshop.overlap(mask, reference),
                }
            )
        parity.append({"id": rid, "max_abs_probability_difference": 0.0, "identical_masks": True})
    outputs = root / "outputs"
    for name, value in {
        "baselines.json": baselines,
        "clipseg_frozen.json": frozen,
        "clipseg_reloaded.json": reloaded,
        "training_history.json": {"best_epoch": 2, "history": []},
        "reload_parity.json": {
            "passed": True,
            "atol": 1e-5,
            "rtol": 0,
            "pid": 1,
            "metric_parity": True,
            "records": parity,
        },
        "stage_receipts.json": {"stages": {}},
        **{
            n: {}
            for n in (
                "prepare.json",
                "clipseg.json",
                "frozen_experiment.json",
                "selection.json",
                "sample_manifest.json",
            )
        },
    }.items():
        (outputs / name).write_text(json.dumps(value))
    for name in ("source.json", "requirements.txt"):
        (root / name).write_text((workshop.CARRIED_ROOT / name).read_text(encoding="utf-8"), encoding="utf-8")
    for model in ("clipseg", "sam", "sam2"):
        manifest = f"weights/{model}/dimer-base-manifest.json"
        (root / manifest).parent.mkdir(parents=True, exist_ok=True)
        (root / manifest).write_text((workshop.CARRIED_ROOT / manifest).read_text(encoding="utf-8"))
    (outputs / "adapter").mkdir()
    (outputs / "adapter/manifest.json").write_text("{}")
    (outputs / "adapter/adapter.safetensors").write_bytes(b"stand-in")


def test_summary_states_reload_paired_changes_and_activity(
    tmp_path, monkeypatch, workshop, stand_ins
) -> None:
    _complete_run(tmp_path, workshop)
    monkeypatch.setattr(workshop, "prompts", lambda mask: {"components": 1})
    workshop.report(tmp_path)
    summary = (tmp_path / "outputs/summary.md").read_text(encoding="utf-8")
    paired = json.loads((tmp_path / "outputs/paired_changes.json").read_text())
    assert paired["improved"] == 1 and paired["unchanged"] == 1 and paired["worsened"] == 0
    assert "Fresh reload: passed on 5 records" in summary
    assert (
        f"{paired['improved']} improved, {paired['unchanged']} unchanged, {paired['worsened']} worsened "
        f"(of {len(paired['records'])})"
    ) in summary
    assert "## SAM 2 box-precision activity" in summary
    assert "| 10% | 1 | 0 | 1 | 0 | 1 |" in summary  # one activity image, unchanged by the stand-in, clipped


# ---------------------------------------------------------------- notebook display helpers (SEG-M2, m2, m4)


class _Display:
    def __init__(self) -> None:
        self.shown: list[str] = []

    def install(self, monkeypatch) -> None:
        display = types.ModuleType("IPython.display")
        display.Markdown = lambda text: text
        display.Image = lambda filename=None: filename
        display.FileLink = lambda path: path
        display.display = lambda value: self.shown.append(str(value))
        package = types.ModuleType("IPython")
        package.display = display
        monkeypatch.setitem(sys.modules, "IPython", package)
        monkeypatch.setitem(sys.modules, "IPython.display", display)


def _helpers(root: Path, **extra) -> dict:
    import os
    import subprocess

    tree = ast.parse(_cells()["code-04"])
    namespace = {"json": json, "Path": Path, "ROOT": root, "os": os, "subprocess": subprocess, **extra}
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    exec(compile(ast.Module(functions, []), "code-04", "exec"), namespace)
    return namespace


def _score_rows(root: Path, iou: float) -> None:
    (root / "outputs").mkdir(parents=True)
    row = {
        "id": "t1",
        "role": "test",
        "tp": 1,
        "fp": 1,
        "fn": 1,
        "pixels": 9,
        "iou": iou,
        "dice": 0.5,
        "precision": 0.5,
        "recall": 0.5,
        "predicted_area_fraction": 0.2,
    }
    (root / "outputs/baselines.json").write_text(
        json.dumps(
            [
                {**row, "model": "empty", "protocol": "unassisted", "iou": 0.0, "precision": None},
                {**row, "model": "oracle_filled_box", "protocol": "oracle_box"},
            ]
        )
    )
    (root / "outputs/sam_metrics.json").write_text(json.dumps([{**row, "model": "sam", "protocol": "box"}]))
    (root / "outputs/sam2_metrics.json").write_text(json.dumps([{**row, "model": "sam2", "protocol": "box"}]))


def test_score_table_shows_filled_box_rates_and_the_requested_run(tmp_path, monkeypatch) -> None:
    shown = _Display()
    shown.install(monkeypatch)
    _score_rows(tmp_path / "default", 0.1111)
    _score_rows(tmp_path / "byod", 0.7777)
    helpers = _helpers(tmp_path / "default")
    helpers["show_score_table"](
        ["baselines.json", "sam_metrics.json", "sam2_metrics.json"],
        models={"oracle_filled_box", "sam", "sam2"},
        root=tmp_path / "byod",
    )
    table = shown.shown[-1]
    assert "Precision" in table and "Recall" in table and "Predicted area" in table
    assert "| oracle_filled_box | oracle_box |" in table and "| empty |" not in table
    assert "0.7777" in table and "0.1111" not in table
    helpers["show_score_table"](["baselines.json"], models={"empty"})
    assert "| empty | unassisted | 1 | 0.0000 | 0.5000 | n/a |" in shown.shown[-1]


def test_run_stage_repeats_the_log_only_on_failure(tmp_path, capsys) -> None:
    import os

    script = "import sys\nprint('stage-line')\nsys.exit(3 if sys.argv[-1] == 'bad' else 0)\n"
    (tmp_path / "workshop.py").write_text(script)
    helpers = _helpers(tmp_path, PYTHON=Path(sys.executable), ENV=dict(os.environ))
    helpers["run_stage"]("good")
    assert capsys.readouterr().out.count("stage-line") == 1
    with pytest.raises(RuntimeError, match="bad failed with exit 3"):
        helpers["run_stage"]("bad")
    assert capsys.readouterr().out.count("stage-line") == 2


def test_training_and_reload_displays_are_compact(tmp_path, capsys) -> None:
    (tmp_path / "outputs").mkdir()
    history = [
        {"epoch": e, "train_loss": None if e == 0 else 0.5 / e, "val": {"miou": 0.6 + e / 100}}
        for e in range(5)
    ]
    (tmp_path / "outputs/training_history.json").write_text(
        json.dumps(
            {
                "best_epoch": 4,
                "history": history,
                "optimizer_steps": 120,
                "n_trainable": 10,
                "n_total": 100,
                "trainable_names": [f"decoder.layer{i}" for i in range(40)],
                "observed_epoch_updates": [
                    {"epoch": e, "max_abs_decoder_change": e * 0.01} for e in range(5)
                ],
            }
        )
    )
    (tmp_path / "outputs/reload_parity.json").write_text(
        json.dumps(
            {
                "passed": True,
                "atol": 1e-5,
                "pid": 7,
                "metric_parity": True,
                "records": [
                    {"id": f"r{i}", "max_abs_probability_difference": 0.0, "identical_masks": True}
                    for i in range(92)
                ],
            }
        )
    )
    helpers = _helpers(tmp_path)
    helpers["show_training"]()
    lines = capsys.readouterr().out.strip().splitlines()
    assert (
        len(lines) == 1 + 5 + 1 and "Selected epoch: 4" in lines[-1] and "decoder.layer" not in "".join(lines)
    )
    helpers["show_reload"]()
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "passed: 92 records" in lines[0]


# ---------------------------------------------------------------- learner-facing source (M1, M2, m2, m4-m9)


def test_cells_display_what_their_notes_ask_for() -> None:
    cells = _cells()
    assert "show_box_experiment()" in cells["code-16"] and "show_json" not in cells["code-16"]
    assert "whole image" in cells["md-17"] and "clipped" in cells["md-15"]
    assert "summary.md" in cells["code-20"] and "root=byod_root" in cells["code-20"]
    assert "'baselines.json'" in cells["code-14"] and "oracle_filled_box" in cells["code-14"]
    assert "show_json" not in cells["code-10"] and "show_json" not in cells["code-12"]
    assert "transformers.__version__" in cells["code-04"]
    for cid in ("code-02", "code-04"):
        assert cells[cid].startswith("# @title Infrastructure")
    assert (
        "**How to use this notebook**" in cells["md-00"] and "<summary>Glossary</summary>" in cells["md-00"]
    )
    assert "Files sidebar" in cells["code-18"] and "Files sidebar" in cells["code-20"]
    assert "393 s" in cells["md-00"] and "still need fresh-Colab measurement" not in cells["md-00"]


def test_review_revision_is_recorded() -> None:
    notebook = _notebook()
    meta = notebook["metadata"]["dimer"]
    revision = meta["review_revisions"][-1]
    assert revision["revision"] == "0.2.0-candidate" and revision["base_commit"].startswith("bf3ae25")
    assert {"SEG-M1", "SEG-M2"} <= set(revision["findings"])
    code02 = next(c for c in notebook["cells"] if c["id"] == "code-02")
    assert code02["metadata"].get("cellView") == "form"
    assert revision["workshop_py_sha256"]["after"] == meta["generated_from"]["files"]["workshop.py"]
