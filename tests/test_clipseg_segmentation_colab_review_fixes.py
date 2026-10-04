"""Regression tests for the 2026-10-02 review of tutorials/clipseg_segmentation_colab.ipynb (CLS-M1..M5, CLS-m1..m5).

They execute the notebook's own cells (the carried modules and the Section 4 cell) in a plain namespace on small
synthetic image sets, so they need NumPy and Pillow only: no torch, no weights, no network.
"""
# ruff: noqa: E501  -- assertion messages and expected refusal texts are kept on one line

from __future__ import annotations

import ast
import hashlib
import importlib.util
import io
import json
import re
import sys
import types
import zipfile
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "clipseg_segmentation_colab.ipynb"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_cls_{name}", ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def notebook() -> dict:
    return json.loads(NOTEBOOK.read_text(encoding="utf-8"))


def _code(notebook: dict) -> list[str]:
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


def _markdown(notebook: dict) -> str:
    return "\n".join("".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "markdown")


def _cell(notebook: dict, marker: str) -> str:
    hits = [source for source in _code(notebook) if marker in source]
    assert len(hits) == 1, (marker, len(hits))
    return hits[0]


def _set_field(source: str, name: str, value) -> str:
    out, n = re.subn(rf"^{name} = .*?(  # @param.*)$", lambda m: f"{name} = {value!r}{m.group(1)}", source, flags=re.M)
    assert n == 1, name
    return out


@pytest.fixture()
def carried(notebook):
    """A namespace holding the three carried modules, executed from the notebook cells as the kernel would."""
    namespace: dict = {"__name__": "__main__"}
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code" and c["metadata"].get("dimer", {}).get("embedded_module")]
    assert [c["metadata"]["dimer"]["embedded_module"].rsplit("/", 1)[-1] for c in cells] == ["metrics.py", "pipeline.py", "samples.py"]
    for cell in cells:
        exec(compile("".join(cell["source"]), cell["metadata"]["dimer"]["embedded_module"], "exec"), namespace)
    import os

    namespace["os"] = os
    return namespace


def _png(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _shape(k: int, size=(96, 72)) -> tuple[Image.Image, Image.Image]:
    image = Image.new("RGB", size, (230 - k, 235, 240))
    mask = Image.new("L", size, 0)
    box = [8 + k, 8, 40 + k, 40]
    ImageDraw.Draw(image).ellipse(box, fill=(220, 40, 40))
    ImageDraw.Draw(mask).ellipse(box, fill=255)
    return image, mask


def _zip(path: Path, n: int, *, layout: str = "folders", extra: dict | None = None, rows: list[str] | None = None) -> Path:
    lines = ["file,mask,prompt"]
    with zipfile.ZipFile(path, "w") as archive:
        for k in range(n):
            image, mask = _shape(k)
            if layout == "folders":
                archive.writestr(f"set/images/{k:04d}.png", _png(image))
                archive.writestr(f"set/masks/{k:04d}.png", _png(mask))
                lines.append(f"images/{k:04d}.png,masks/{k:04d}.png,a red circle")
            else:
                archive.writestr(f"img{k:03d}.png", _png(image))
                archive.writestr(f"img{k:03d}_mask.png", _png(mask))
                lines.append(f"img{k:03d}.png,img{k:03d}_mask.png,a red circle")
        for name, payload in (extra or {}).items():
            archive.writestr(name, payload)
        table = "set/masks.csv" if layout == "folders" else "masks.csv"
        archive.writestr(table, "\n".join(rows if rows is not None else lines) + "\n")
    return path


def _section4(notebook, carried, byod_path: str, use_byod: bool = True) -> dict:
    source = _cell(notebook, "USE_BYOD = False")
    source = _set_field(_set_field(source, "USE_BYOD", use_byod), "BYOD_PATH", byod_path)
    namespace = dict(carried)
    exec(compile(source, "<section 4>", "exec"), namespace)
    return namespace


def _helpers(notebook, carried) -> dict:
    """Only the function and constant definitions of the Section 4 cell (no data is read)."""
    tree = ast.parse(_cell(notebook, "USE_BYOD = False"))
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef | ast.Import | ast.ImportFrom) or (isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id.startswith("BYOD_"))]
    namespace = dict(carried)
    exec(compile(ast.Module(body=keep, type_ignores=[]), "<section 4 helpers>", "exec"), namespace)
    return namespace


# ---------------------------------------------------------------------------------------------- CLS-M1
def test_isolated_runtime_no_in_kernel_install(notebook):
    sources = _code(notebook)
    kernel = [s for s in sources if "# dimer: kernel cell" in s]
    assert len(kernel) == 2
    install = next(s for s in kernel if "LOCK_TEXT = r" in s)
    for needed in ('"--managed-python"', '"--require-hashes"', '"--only-binary"', '":all:"', "UV_SHA256", "LOCK_SHA256"):
        assert needed in install
    assert notebook["metadata"]["dimer"]["notebook_spec"] == "2.2"
    assert notebook["metadata"]["dimer"]["generated_from"]["generator"] == "build_notebook.py/2.1"
    # The only remaining pip line is the guarded fallback in the runtime-record cell, skipped by the isolated worker.
    pip_cells = [s for s in sources if "'-m', 'pip', 'install'" in s]
    assert len(pip_cells) == 1 and "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'" in pip_cells[0]


def test_lock_pins_every_direct_dependency_with_hashes():
    build = _load("build_notebook")
    template = _load("notebook_template").TEMPLATE
    lock = (ROOT / template["lock"]).read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT, template), lock)  # raises SystemExit on a missing pin or hash
    assert build.lock_packages(lock)["pyarrow"] == "25.0.1"


def test_no_restart_text_left(notebook):
    markdown = _markdown(notebook)
    for stale in ("Restart the runtime, then rerun", "installs the pinned dependencies", "about 2 minutes of cell time", "re-run from that cell"):
        assert stale not in markdown


# ---------------------------------------------------------------------------------------------- CLS-M2
def test_sections_5_to_7_reset_before_using_the_model(notebook):
    s5 = _cell(notebook, "def reset_to_pretrained")
    s6 = _cell(notebook, "frozen_test = pipe.evaluate")
    s7 = _cell(notebook, "adapt_result = pipe.adapt(")
    assert s5.index("\nreset_to_pretrained()") < s5.index("segment_scene(pipe, 'frozen')")
    assert s6.index("reset_to_pretrained()") < s6.index("frozen_test = pipe.evaluate")
    assert s6.index("if frozen_test['adapted']:") < s6.index("print({'frozen_model_test'")
    assert s7.index("reset_to_pretrained()") < s7.index("if pipe.adapter is not None:") < s7.index("pipe.adapt(")
    assert re.search(r"^THRESHOLD = 0\.5  # @param", s5, re.M)


def test_reset_to_pretrained_reloads_only_an_adapted_model(notebook):
    tree = ast.parse(_cell(notebook, "def reset_to_pretrained"))
    func = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "reset_to_pretrained")
    loads = []

    class FakePipeline:
        def __init__(self, adapter):
            self.adapter = adapter

        @classmethod
        def from_pretrained(cls, weights_dir):
            loads.append(weights_dir)
            return cls(None)

    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None))
    namespace = {"gc": __import__("gc"), "torch": fake_torch, "ClipSegSegmentationPipeline": FakePipeline, "WEIGHTS_DIR": "w"}
    exec(compile(ast.Module(body=[func], type_ignores=[]), "<reset>", "exec"), namespace)
    namespace["pipe"] = FakePipeline(None)
    namespace["reset_to_pretrained"]()
    assert loads == []
    namespace["pipe"] = FakePipeline({"best_epoch": 4})
    namespace["reset_to_pretrained"]()
    assert loads == ["w"] and namespace["pipe"].adapter is None


# ---------------------------------------------------------------------------------------------- CLS-M3
def test_byod_minimum_is_twelve_and_matches_split_dataset(notebook, carried):
    helpers = _helpers(notebook, carried)
    assert helpers["BYOD_MIN_IMAGES"] == 12
    records = [{"id": f"r{k}", "image": _shape(k)[0], "prompt": "a red circle", "mask": _shape(k)[1]} for k in range(12)]
    sizes = {name: len(part) for name, part in carried["split_dataset"](records, seed=42).items()}
    assert sizes == {"test": 2, "validation": 2, "train": 8}
    sizes11 = {name: len(part) for name, part in carried["split_dataset"](records[:11], seed=42).items()}
    assert sizes11["train"] < carried["MIN_RECORDS"]


def test_byod_at_the_minimum_is_accepted_and_one_below_is_refused(notebook, carried, tmp_path):
    ok = _section4(notebook, carried, str(_zip(tmp_path / "twelve.zip", 12)))
    assert ok["disjoint"] == {"test": 2, "validation": 2, "train": 8}
    assert len(ok["train_records"]) == 8 and ok["data_source"] == "BYOD (twelve.zip)"
    with pytest.raises(ValueError, match=r"holds 11 records \(11 distinct images\); BYOD needs at least 12 distinct images"):
        _section4(notebook, carried, str(_zip(tmp_path / "eleven.zip", 11)))


# ---------------------------------------------------------------------------------------------- CLS-M4
def test_two_folder_layout_keeps_images_and_masks_apart(notebook, carried, tmp_path):
    helpers = _helpers(notebook, carried)
    records, report = helpers["load_byod_records"](_zip(tmp_path / "folders.zip", 12))
    assert len(records) == 12 and report["files"] == 25
    for record in records:
        assert len(record["image"].getcolors()) == 2  # the drawn photograph, not its two-level mask
        assert record["image"].tobytes() != record["mask"].convert("RGB").tobytes()
    assert records[0]["id"] == "set_images_0000"


def test_flat_layout_still_loads(notebook, carried, tmp_path):
    helpers = _helpers(notebook, carried)
    records, _report = helpers["load_byod_records"](_zip(tmp_path / "flat.zip", 3, layout="flat"))
    assert [r["id"] for r in records] == ["img000", "img001", "img002"]


def test_image_and_mask_resolving_to_one_member_is_refused(notebook, carried, tmp_path):
    helpers = _helpers(notebook, carried)
    rows = ["file,mask,prompt", "images/0000.png,images/0000.png,a red circle"]
    with pytest.raises(ValueError, match=r"the image and the mask are the same file 'set/images/0000.png'"):
        helpers["load_byod_records"](_zip(tmp_path / "same.zip", 1, rows=rows))


def test_duplicate_member_path_is_refused(notebook, carried, tmp_path):
    helpers = _helpers(notebook, carried)
    path = _zip(tmp_path / "dup.zip", 2)
    with pytest.warns(UserWarning, match="Duplicate name"), zipfile.ZipFile(path, "a") as archive:
        archive.writestr("set/images/0000.png", _png(Image.new("RGB", (96, 72))))
    with pytest.raises(ValueError, match=r"two members with the same path 'set/images/0000.png'"):
        helpers["load_byod_records"](path)


def test_stray_file_with_a_listed_name_is_refused_not_swapped_in(notebook, carried, tmp_path):
    helpers = _helpers(notebook, carried)
    stray = {"set/other/0000.png": _png(Image.new("RGB", (96, 72)))}
    with pytest.raises(ValueError, match=r"not named by any masks\.csv row, e\.g\. 'set/other/0000\.png': delete them"):
        helpers["load_byod_records"](_zip(tmp_path / "stray.zip", 2, extra=stray))


@pytest.mark.parametrize("bad", ["../escape.png", "/abs/0000.png", "C:/abs/0000.png"])
def test_paths_leaving_the_upload_are_refused(notebook, carried, tmp_path, bad):
    helpers = _helpers(notebook, carried)
    rows = ["file,mask,prompt", f"{bad},masks/0000.png,a red circle"]
    with pytest.raises(ValueError, match=r"absolute path|leaves the upload folder"):
        helpers["load_byod_records"](_zip(tmp_path / "bad.zip", 1, rows=rows))


def test_missing_file_names_the_relative_path_and_a_hint(notebook, carried, tmp_path):
    helpers = _helpers(notebook, carried)
    rows = ["file,mask,prompt", "0000.png,masks/0000.png,a red circle"]
    with pytest.raises(ValueError, match=r"names a missing file 'set/0000\.png'; the upload has \['set/images/0000\.png'"):
        helpers["load_byod_records"](_zip(tmp_path / "missing.zip", 1, rows=rows))


# ---------------------------------------------------------------------------------------------- CLS-m5
def test_macos_metadata_is_ignored(notebook, carried, tmp_path):
    helpers = _helpers(notebook, carried)
    extra = {"__MACOSX/set/images/._0000.png": b"\x00\x05\x16\x07", "set/.DS_Store": b"x", "set/images/._0001.png": b"x"}
    records, report = helpers["load_byod_records"](_zip(tmp_path / "mac.zip", 2, extra=extra))
    assert len(records) == 2 and report["ignored_os_metadata"] == 3


def test_jpeg_mask_is_refused_and_grey_masks_are_counted(notebook, carried, tmp_path):
    helpers = _helpers(notebook, carried)
    image, mask = _shape(0)
    jpeg = io.BytesIO()
    mask.save(jpeg, "JPEG")
    path = tmp_path / "jpeg.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.png", _png(image))
        archive.writestr("a_mask.jpg", jpeg.getvalue())
        archive.writestr("masks.csv", "file,mask,prompt\na.png,a_mask.jpg,a red circle\n")
    with pytest.raises(ValueError, match=r"is a JPEG; JPEG compression adds stray non-zero pixels"):
        helpers["load_byod_records"](path)
    grey = mask.copy()
    grey.putpixel((0, 0), 128)
    path = tmp_path / "grey.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.png", _png(image))
        archive.writestr("a_mask.png", _png(grey))
        archive.writestr("masks.csv", "file,mask,prompt\na.png,a_mask.png,a red circle\n")
    _records, report = helpers["load_byod_records"](path)
    assert report["non_binary_masks"] == 1 and report["non_binary_examples"] == ["a_mask.png"]


def test_byod_path_directory_needs_no_colab(notebook, carried, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "google", None)  # importing google.colab would fail
    source = tmp_path / "zipped.zip"
    _zip(source, 12)
    folder = tmp_path / "folder"
    with zipfile.ZipFile(source) as archive:
        for info in archive.infolist():
            target = folder / info.filename
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(info))
    ns = _section4(notebook, carried, str(folder / "set"))
    assert ns["byod_upload"]["kind"] == "directory" and ns["byod_upload"]["files"] == 25
    with pytest.raises(RuntimeError, match="set BYOD_PATH to a .zip or a directory holding masks.csv"):
        _section4(notebook, carried, "")


def test_empty_upload_is_refused_with_an_instruction(notebook, carried, monkeypatch):
    colab = types.ModuleType("google.colab")
    colab.files = types.SimpleNamespace(upload=lambda: {})
    google = types.ModuleType("google")
    google.colab = colab
    monkeypatch.setitem(sys.modules, "google", google)
    monkeypatch.setitem(sys.modules, "google.colab", colab)
    with pytest.raises(ValueError, match=r"Upload exactly one \.zip file \(received 0\)"):
        _section4(notebook, carried, "")


# ---------------------------------------------------------------------------------------------- CLS-m4 / m1
def test_byod_upload_digest_and_result_without_corpus(notebook, carried, tmp_path):
    path = _zip(tmp_path / "digest.zip", 12)
    ns = _section4(notebook, carried, str(path))
    assert ns["byod_upload"] == {"name": "digest.zip", "kind": "zip", "bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    s9 = _cell(notebook, "reloaded = ClipSegSegmentationPipeline.from_artifact")
    tail = s9[s9.index("if byod_upload is not None:"):]
    assert "del result_payload['corpus']" in tail and "result_payload['byod_upload'] = byod_upload" in tail
    assert tail.index("result_payload['byod_upload']") < tail.index("json.dump(result_payload")


def test_panels_draw_reference_frozen_and_adapted(notebook):
    s6 = _cell(notebook, "frozen_test = pipe.evaluate")
    s9 = _cell(notebook, "reloaded = ClipSegSegmentationPipeline.from_artifact")
    assert "frozen_example_masks = " in s6
    assert "overlay(small, fro_small, (40, 70, 200))" in s9
    assert "'panels': ['reference overlay', 'frozen overlay', 'adapted overlay']" in s9


def test_no_result_dependent_asserts_in_learner_cells(notebook):
    for cell in notebook["cells"]:
        source = "".join(cell["source"])
        if cell["cell_type"] != "code" or cell["metadata"].get("dimer", {}).get("embedded_module") or "# dimer: kernel cell" in source:
            continue
        assert not re.search(r"^\s*assert ", source, re.M), source[:60]


# ---------------------------------------------------------------------------------------------- CLS-M5 / m2 / m3
def test_guided_layer_and_corrected_numbers(notebook):
    markdown = _markdown(notebook)
    for marker in ("**Who this is for.**", "**Input → Model → Output.**", "**How to use this notebook.**", "**Roadmap:**", "## 10. Your turn — change one thing", "**Predict → Change → Run → Observe → Explain:**", "## Troubleshooting", "## Glossary", "## Conclusion (your notes)"):
        assert marker in markdown, marker
    assert markdown.count("**Predict before running:**") >= 7 and markdown.count("<summary>Check your reasoning</summary>") >= 7
    assert markdown.count("> **Infrastructure.**") == 3
    assert "0.86" not in markdown and "mean IoU of **0.947**" in markdown
    assert "{{" not in markdown and "}}" not in markdown
    hidden = [c for c in notebook["cells"] if c["metadata"].get("dimer", {}).get("embedded_module")]
    assert all(c["metadata"].get("jupyter", {}).get("source_hidden") for c in hidden)
    assert re.search(r"^ACTIVITY_THRESHOLD = 0\.3  # @param", _cell(notebook, "ACTIVITY_THRESHOLD = 0.3"), re.M)
