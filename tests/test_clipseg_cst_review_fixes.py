"""Regression tests for the 2026-10-16 review of tutorials/clipseg_segmentation_colab.ipynb (CST-m1..m8, CST-S1).

They read the generated notebook and execute its own Section 4 cell on small synthetic image sets. The target-resize
round trip uses torch (installed in CI, CPU only); every other test needs NumPy and Pillow only.
"""
# ruff: noqa: E501  -- assertion messages and expected texts are kept on one line

from __future__ import annotations

import importlib.util
import io
import json
import re
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "clipseg_segmentation_colab.ipynb"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"_cst_{name}", ROOT / "tools" / f"{name}.py")
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


# ---------------------------------------------------------------------------------------------- CST-m8
def _target_mode(notebook: dict) -> str:
    source = _cell(notebook, "def _encode(")
    modes = re.findall(r'size=\(LOGIT_SIZE, LOGIT_SIZE\), mode="([a-z-]+)"', source)
    assert len(modes) == 1, modes
    return modes[0]


def test_training_targets_are_area_averaged_not_nearest(notebook):
    assert _target_mode(notebook) == "area"
    assert 'mode="nearest"' not in _cell(notebook, "def _encode(")
    # src/ keeps its own resize (the workshop notebook carries src/ byte for byte); the tutorial's rewrite is the fix.
    template = _load("notebook_template")
    assert any("CST-m8" in rule[1] for rule in template.TEMPLATE["rewrites"])


def test_target_round_trip_is_centre_consistent(notebook):
    torch = pytest.importorskip("torch")
    mode = _target_mode(notebook)
    height, width, size = 480, 640, 352
    mask = np.zeros((height, width), dtype=np.float32)
    mask[140:340, 220:420] = 1.0  # a centred 200 x 200 square
    down = torch.nn.functional.interpolate(torch.from_numpy(mask)[None, None], size=(size, size), mode=mode)
    # the notebook's prediction path: bilinear, align_corners=False, then >= 0.5
    back = (torch.nn.functional.interpolate(down, size=(height, width), mode="bilinear", align_corners=False)[0, 0] >= 0.5).numpy()
    ys, xs = np.nonzero(back)
    y0, x0 = np.nonzero(mask)
    assert abs(xs.mean() - x0.mean()) < 0.05 and abs(ys.mean() - y0.mean()) < 0.05
    iou = np.logical_and(back, mask > 0).sum() / np.logical_or(back, mask > 0).sum()
    assert iou > 0.995


# ---------------------------------------------------------------------------------------------- CST-m7
def test_no_exactness_claim_over_the_float16_cache(notebook):
    encode = _cell(notebook, "def _encode(")
    assert "torch.float16" in encode  # the cache stays half precision (the measured effect is ~1e-5 mIoU)
    text = _markdown(notebook) + "\n" + encode
    assert not re.search(r"logits equal the full model'?s exactly", text)
    assert "cached in float16, so the decoder sees them rounded" in encode
    assert "rounded to half precision" in _markdown(notebook)


# ---------------------------------------------------------------------------------------------- CST-m1
def test_spec_2_5_metadata_kind_and_use_notice(notebook):
    meta = notebook["metadata"]["dimer"]
    assert meta["notebook_spec"] == "2.5" and meta["notebook_kind"] == "TUTORIAL"
    first_code = next(i for i, c in enumerate(notebook["cells"]) if c["cell_type"] == "code")
    before = "\n".join("".join(c["source"]) for c in notebook["cells"][:first_code])
    assert "> **⚠️ Notice.** For research, training and evaluation only." in before
    assert "without warranty of any kind" in before and "accept no liability" in before
    assert "use notice" in before.lower()
    title_end = before.index("\n")
    assert before.index("**⚠️ Notice.**") > title_end


def test_references_are_numbered_ieee_and_cited(notebook):
    text = _markdown(notebook)
    refs = text[text.index("## References"):]
    numbers = [int(n) for n in re.findall(r"^\[(\d+)\] ", refs, flags=re.M)]
    assert numbers == list(range(1, len(numbers) + 1)) and len(numbers) >= 8
    body = text[: text.index("## References")]
    cited = {int(n) for group in re.findall(r"\[(\d+(?:\], \[\d+)*)\]", body) for n in re.findall(r"\d+", group)}
    assert cited == set(numbers), (sorted(cited), numbers)
    for entry in re.findall(r"^\[\d+\] .*$", refs, flags=re.M):
        assert "https://" in entry or "ml-worker" in entry, entry


# ---------------------------------------------------------------------------------------------- CST-m2 / m3
def test_run_statement_is_run_neutral_and_quotes_are_labelled(notebook):
    text = _markdown(notebook)
    assert "no hosted run of this version is recorded yet" not in text
    assert "docs/release-verification.md" in text
    # every quote of the 2026-09-21 Kaggle run names that earlier blob's revision
    for match in re.finditer(r"2026-09-21", text):
        window = text[max(0, match.start() - 160): match.end() + 160]
        assert "2c999ae" in window or "100adc4f" in window, window
    assert "local CPU run of this version (2026-10-04)" not in text
    assert "CPU build record" in text


# ---------------------------------------------------------------------------------------------- CST-m4
def test_pretraining_overlap_is_stated(notebook):
    text = _markdown(notebook)
    assert re.search(r"pretrain[^.]{0,200}may include", text)
    assert "Recipe1M" in text and "not audited" in text


# ---------------------------------------------------------------------------------------------- CST-m6
def test_isolated_env_variable_is_documented(notebook):
    assert any("DIMER_ISOLATED_ENV" in s for s in _code(notebook))
    assert "`DIMER_ISOLATED_ENV`" in _markdown(notebook)


# ---------------------------------------------------------------------------------------------- CST-S1
def test_frozen_vs_adapted_claim_is_scoped_to_the_fixed_threshold(notebook):
    text = _markdown(notebook)
    interpretation = text[text.index("## Interpretation and limits"):]
    assert "no validation-tuned threshold was tried for the frozen model" in interpretation


# ---------------------------------------------------------------------------------------------- CST-m5
def test_router_relays_refusals_without_a_traceback(notebook):
    router = _cell(notebook, "class IsolatedCellError")
    assert "class IsolatedCellRefusal(Exception)" in router
    assert 'elif kind == "refusal":' in router and "raise IsolatedCellRefusal(message[1]) from None" in router
    assert 'if getattr(exc, "dimer_refusal", False):' in router
    # genuine failures still carry their traceback
    assert 'raise IsolatedCellError(message[2]) from None' in router and "traceback.format_exception" in router


@pytest.fixture()
def carried(notebook):
    namespace: dict = {"__name__": "__main__"}
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code" and cell["metadata"].get("dimer", {}).get("embedded_module"):
            exec(compile("".join(cell["source"]), cell["metadata"]["dimer"]["embedded_module"], "exec"), namespace)
    import os

    namespace["os"] = os
    return namespace


def _zip(path: Path, n: int) -> Path:
    lines = ["file,mask,prompt"]
    with zipfile.ZipFile(path, "w") as archive:
        for k in range(n):
            image = Image.new("RGB", (96, 72), (230 - k, 235, 240))
            mask = Image.new("L", (96, 72), 0)
            ImageDraw.Draw(image).ellipse([8 + k, 8, 40 + k, 40], fill=(220, 40, 40))
            ImageDraw.Draw(mask).ellipse([8 + k, 8, 40 + k, 40], fill=255)
            for name, im in ((f"set/images/{k:04d}.png", image), (f"set/masks/{k:04d}.png", mask)):
                buffer = io.BytesIO()
                im.save(buffer, "PNG")
                archive.writestr(name, buffer.getvalue())
            lines.append(f"images/{k:04d}.png,masks/{k:04d}.png,a red circle")
        archive.writestr("set/masks.csv", "\n".join(lines) + "\n")
    return path


def _section4(notebook, carried, byod_path: str) -> dict:
    source = _cell(notebook, "USE_BYOD = False")
    source = re.sub(r"^USE_BYOD = .*?(  # @param.*)$", r"USE_BYOD = True\1", source, flags=re.M)
    source = re.sub(r"^BYOD_PATH = .*?(  # @param.*)$", lambda m: f"BYOD_PATH = {byod_path!r}{m.group(1)}", source, flags=re.M)
    namespace = dict(carried)
    exec(compile(source, "<section 4>", "exec"), namespace)
    return namespace


def test_byod_refusal_is_one_message_without_a_traceback(notebook, carried, tmp_path):
    with pytest.raises(ValueError) as caught:
        _section4(notebook, carried, str(_zip(tmp_path / "eleven.zip", 11)))
    exc = caught.value
    assert type(exc).__name__ == "BYODRefusal" and exc.dimer_refusal is True
    rendered = exc._render_traceback_()
    assert len(rendered) == 1 and "Traceback" not in rendered[0]
    assert "holds 11 records (11 distinct images); BYOD needs at least 12 distinct images" in rendered[0]
    assert "Run after" in rendered[0]
    assert exc.__suppress_context__ or exc.__cause__ is None


def test_byod_refusal_wraps_contract_errors_but_not_the_default_path(notebook, carried, tmp_path):
    archive = tmp_path / "bad.zip"
    with zipfile.ZipFile(_zip(tmp_path / "base.zip", 12)) as src, zipfile.ZipFile(archive, "w") as dst:
        for info in src.infolist():
            payload = src.read(info)
            if info.filename == "set/masks.csv":
                payload = payload.replace(b"a red circle", b"x" * 80, 1)  # a phrase over 64 characters
            dst.writestr(info, payload)
    with pytest.raises(ValueError) as caught:
        _section4(notebook, carried, str(archive))
    assert type(caught.value).__name__ == "BYODRefusal"
    assert caught.value._render_traceback_()[0].startswith("BYOD upload refused: ")


def test_template_default_rewrites_copy_matches_the_generator():
    assert tuple(tuple(r) for r in _load("notebook_template").DEFAULT_REWRITES) == tuple(tuple(r) for r in _load("build_notebook").DEFAULT_REWRITES)
