"""Regression tests for the 2026-10-16 review of tutorials/clipseg_segmentation_colab.ipynb (CST-m1..m8, CST-S1).

They read the generated notebook and execute its own Section 4 cell on small synthetic image sets. The target-resize
round trip uses torch (installed in CI, CPU only); every other test needs NumPy and Pillow only.
"""
# ruff: noqa: E501  -- assertion messages and expected texts are kept on one line

from __future__ import annotations

import hashlib
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
    # One implementation: the fix lives in src/, which the tutorial and the workshop both carry (round 2, 2026-10-16).
    src = (ROOT / "src/clipseg_segmentation_pipeline/pipeline.py").read_text(encoding="utf-8")
    assert 'mode="area")[0, 0] for r in batch]' in src and 'mode="nearest"' not in src
    assert "rewrites" not in _load("notebook_template").TEMPLATE


def _fixture(name: str) -> np.ndarray:
    shapes = {
        "off-centre box, 640x480": ((480, 640), (slice(100, 331), slice(57, 419))),
        "odd dimensions, 641x479": ((479, 641), (slice(33, 300), slice(101, 500))),
        "non-integer ratio, 700x500": ((500, 700), (slice(123, 377), slice(211, 589))),
        "thin 3-px column, 640x480": ((480, 640), (slice(None), slice(300, 303))),
        "thin 2-px row, 640x480": ((480, 640), (slice(240, 242), slice(None))),
        "smaller than 352, 150x200": ((200, 150), (slice(40, 130), slice(30, 101))),
        "tiny, 23x17": ((17, 23), (slice(3, 11), slice(5, 16))),
    }
    size, box = shapes[name]
    mask = np.zeros(size, dtype=np.float32)
    mask[box] = 1.0
    return mask


@pytest.mark.parametrize("name", ["off-centre box, 640x480", "odd dimensions, 641x479", "non-integer ratio, 700x500", "thin 3-px column, 640x480", "thin 2-px row, 640x480", "smaller than 352, 150x200", "tiny, 23x17"])
def test_area_targets_on_fixture_geometries(notebook, name):
    """Regression bounds for these fixtures only, not a general guarantee: area mode is adaptive average pooling
    (for inputs below 352 it enlarges), so the targets are soft values in [0, 1] whose mean stays close to the
    mask's area fraction, and the notebook's prediction path (bilinear up, >= 0.5) recovers each fixture."""
    torch = pytest.importorskip("torch")
    mode = _target_mode(notebook)
    mask = _fixture(name)
    height, width = mask.shape
    down = torch.nn.functional.interpolate(torch.from_numpy(mask)[None, None], size=(352, 352), mode=mode)
    assert float(down.min()) >= 0.0 and float(down.max()) <= 1.0
    assert abs(float(down.mean()) - float(mask.mean())) < 0.002
    back = (torch.nn.functional.interpolate(down, size=(height, width), mode="bilinear", align_corners=False)[0, 0] >= 0.5).numpy()
    reference = mask > 0
    assert np.logical_and(back, reference).sum() / np.logical_or(back, reference).sum() > 0.99
    ys, xs = np.nonzero(back)
    y0, x0 = np.nonzero(reference)
    assert abs(xs.mean() - x0.mean()) < 0.1 and abs(ys.mean() - y0.mean()) < 0.1


def test_adapt_refuses_train_validation_overlap():
    pytest.importorskip("PIL")
    from clipseg_segmentation_pipeline import ClipSegSegmentationPipeline

    def record(k: int, rid: str) -> dict:
        image = Image.new("RGB", (64, 48), (200 - 10 * k, 30, 30))
        mask = np.zeros((48, 64), dtype=bool)
        mask[10:30, 10:40] = True
        return {"id": rid, "image": image, "prompt": "a red box", "mask": mask}

    train = [record(k, f"t{k}") for k in range(8)]
    pipe = ClipSegSegmentationPipeline(lambda image, prompts: [], "cpu")
    with pytest.raises(ValueError, match=r"train and validation must not share an image \(exact decoded-pixel match\)"):
        pipe.adapt(train, [record(3, "v-copy-of-t3")])
    with pytest.raises(RuntimeError, match="no loaded model"):  # a disjoint validation split passes the check
        pipe.adapt(train, [record(9, "v9")])


def test_codex_peer_review_wording(notebook):
    text = _markdown(notebook)
    src = (ROOT / "src/clipseg_segmentation_pipeline/pipeline.py").read_text(encoding="utf-8")
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for banned in ("centred like", "moved threshold", "moved cut", "annotation convention", "without leakage", "rather than the task", "Nobody has checked", "upstream training objective", "no dispersion"):
        assert banned not in text and banned not in src and banned not in readme, banned
    assert "soft targets produced by area-mode averaging (adaptive average pooling)" in text
    assert "an observation on a test set that had already been inspected, not independent confirmation" in text
    assert "Both rates improved at 0.5. This does not establish superiority over a validation-tuned frozen model." in text
    assert "following the upstream loss family" in text and "following the upstream loss family" in src
    assert text.count("these results don't identify whether the gain comes from boundaries, region selection, phrase grounding or other learned associations") == 2
    assert "near-duplicate and source-level overlap were not audited" in text and "This tutorial has not audited overlap" in text.replace('"\n        "', "")
    assert "No uncertainty estimate is reported" in text and "no further validation-IoU improvement under this selection rule" in text
    assert "Compare settings on the validation numbers" in text
    assert "positive phrase–image pairs only" in text


def test_repeated_test_scores_are_labelled_exploratory(notebook):
    cell = _cell(notebook, "run_history.append(")
    assert "test_use = 'first test scoring in this session' if not run_history else 'exploratory:" in cell
    assert "{'test_use': test_use," in cell


def test_synthetic_references_are_visible_regions(notebook):
    source = _cell(notebook, "def synthetic_scene(")
    tree_src = source[source.index("def synthetic_scene("): source.index("scene, scene_masks = synthetic_scene()")]
    namespace = {"np": np, "Image": Image, "ImageDraw": ImageDraw}
    exec(tree_src, namespace)
    image, masks = namespace["synthetic_scene"]()
    stacked = np.stack(list(masks.values()))
    assert stacked.sum(axis=0).max() == 1  # no pixel belongs to two references
    assert not (masks["green grass"] & masks["a yellow triangle"]).any() and masks["a yellow triangle"].any()


def test_both_empty_convention_is_documented():
    metrics = (ROOT / "src/clipseg_segmentation_pipeline/metrics.py").read_text(encoding="utf-8")
    assert "both empty" in metrics and "positive phrase-image pairs only" in metrics


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


def test_workshop_carries_the_fixed_src_byte_for_byte():
    workshop = json.loads((ROOT / "tutorials" / "DIMER_MultiModel_Image_Segmentation_Workshop.ipynb").read_text(encoding="utf-8"))
    code = "".join("".join(c["source"]) for c in workshop["cells"] if c["cell_type"] == "code")
    src = (ROOT / "src/clipseg_segmentation_pipeline/pipeline.py").read_text(encoding="utf-8")
    digest = hashlib.sha256(src.encode("utf-8")).hexdigest()
    assert f"'clipseg_reference/pipeline.py': '{digest}'" in code
