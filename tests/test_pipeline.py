import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from clipseg_segmentation_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    LOGIT_SIZE,
    MASK_THRESHOLD,
    MAX_IMAGE_SIDE,
    MAX_PROMPT_CHARS,
    MAX_PROMPTS,
    MAX_TEXT_TOKENS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    ClipSegSegmentationPipeline,
    format_prompts,
    mask_bbox,
    mask_iou,
    stage_missing_files,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "CIDAS/clipseg-rd64-refined"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    assert 0.0 < MASK_THRESHOLD < 1.0 and LOGIT_SIZE == 352
    assert MAX_PROMPTS == 16 and MAX_PROMPT_CHARS == 64 and MAX_TEXT_TOKENS == 77
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION
        paths = [entry["path"] for entry in data["files"]]
        assert "model.safetensors" in paths and "pytorch_model.bin" not in paths


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "clipseg"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "clipseg"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_missing_file_and_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)
    _write_snapshot(tmp_path, b"abc")
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_stage_missing_files_fetches_only_absent_entries_then_verifies(tmp_path):
    """Fresh-clone shape: manifest committed, weight file absent. allow_download fetches exactly that file."""
    payload = b"weights-bytes"
    (tmp_path / "config.json").write_bytes(b"{}")
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {"path": "config.json", "bytes": 2, "sha256": hashlib.sha256(b"{}").hexdigest()},
            {"path": "model.bin", "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()},
        ],
    }
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="allow_download=True"):
        stage_missing_files(tmp_path)
    fetched = []

    def fake_download(relative_path, root):
        fetched.append(relative_path)
        (root / relative_path).write_bytes(payload)

    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == ["model.bin"]
    assert fetched == ["model.bin"]
    listed = verify_snapshot(tmp_path)["files"]
    assert (listed if isinstance(listed, int) else len(listed)) == 2
    assert stage_missing_files(tmp_path, allow_download=True, downloader=fake_download) == []


def test_stage_missing_files_refuses_foreign_manifest(tmp_path):
    manifest = {"modelId": "someone/else", "revision": MODEL_REVISION, "files": []}
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="refusing to stage"):
        stage_missing_files(tmp_path, allow_download=True, downloader=lambda *_: None)


def test_format_prompts_lowercases_and_normalises():
    assert format_prompts(["A photo of a  Cat", "remote control."]) == ["a photo of a cat", "remote control"]
    with pytest.raises(ValueError, match="distinct"):
        format_prompts(["a cat", "A cat."])
    with pytest.raises(TypeError):
        format_prompts("a cat")
    with pytest.raises(TypeError):
        format_prompts(["a cat", 3])
    with pytest.raises(ValueError, match="empty"):
        format_prompts(["a cat", " . "])
    with pytest.raises(ValueError, match="MAX_PROMPTS"):
        format_prompts(["x"] * (MAX_PROMPTS + 1))
    with pytest.raises(ValueError, match="MAX_PROMPTS"):
        format_prompts([])
    with pytest.raises(ValueError, match="MAX_PROMPT_CHARS"):
        format_prompts(["a" * (MAX_PROMPT_CHARS + 1)])


def _probs(queries: list[str], height: int, width: int) -> np.ndarray:
    """One map per query: a bright top-left block for the first query, empty maps for the rest."""
    probs = np.zeros((len(queries), height, width), dtype=np.float32)
    probs[0, : height // 2, : width // 2] = 0.9
    probs[0, height // 2 :, :] = 0.2
    return probs


def _fake_pipeline(calls: list | None = None) -> ClipSegSegmentationPipeline:
    def runner(image: Image.Image, queries: list[str]) -> np.ndarray:
        if calls is not None:
            calls.append((image.mode, list(queries)))
        return _probs(queries, image.height, image.width)

    return ClipSegSegmentationPipeline(runner, "cpu")


def test_segment_output_fields_and_defaults():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.segment(Image.new("L", (40, 20)), ["A red Circle.", "a cat"])
    assert calls == [("RGB", ["a red circle", "a cat"])]
    assert result["queries"] == ["a red circle", "a cat"]
    assert result["threshold"] == MASK_THRESHOLD
    assert (result["width"], result["height"]) == (40, 20)
    assert (result["model_id"], result["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    first, second = result["segments"]
    assert first["prompt"] == "a red circle"
    assert first["mask"].shape == (20, 40) and first["mask"].dtype == bool
    assert first["probability"].shape == (20, 40) and first["probability"].dtype == np.float32
    assert first["area_fraction"] == pytest.approx(0.25)  # the 0.9 block; the 0.2 band is below 0.5
    assert first["max_probability"] == pytest.approx(0.9)
    assert first["bbox"] == [0, 0, 20, 10]
    assert second["area_fraction"] == 0.0 and second["bbox"] is None and second["max_probability"] == 0.0


def test_segment_threshold_is_caller_owned():
    pipe = _fake_pipeline()
    low = pipe.segment(Image.new("RGB", (40, 20)), ["a red circle"], threshold=0.1)
    assert low["segments"][0]["area_fraction"] == pytest.approx(0.75)  # the 0.2 band now counts
    high = pipe.segment(Image.new("RGB", (40, 20)), ["a red circle"], threshold=0.95)
    assert high["segments"][0]["area_fraction"] == 0.0 and high["segments"][0]["bbox"] is None


def test_segment_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.segment(np.zeros((30, 40, 3), dtype=np.uint8), ["a cat"])
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.segment(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)), ["a cat"])
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.segment(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)), ["a cat"])
    with pytest.raises(TypeError, match="not a single string"):
        pipe.segment(Image.new("RGB", (64, 64)), "a cat")
    with pytest.raises(ValueError, match="threshold"):
        pipe.segment(Image.new("RGB", (64, 64)), ["a cat"], threshold=1.5)
    with pytest.raises(ValueError, match="threshold"):
        pipe.segment(Image.new("RGB", (64, 64)), ["a cat"], threshold=True)


def test_segment_rejects_malformed_backend_output():
    bad_shape = ClipSegSegmentationPipeline(lambda image, queries: np.zeros((1, 5, 5), np.float32), "cpu")
    with pytest.raises(RuntimeError, match="shape"):
        bad_shape.segment(Image.new("RGB", (40, 20)), ["a cat"])
    bad_range = ClipSegSegmentationPipeline(
        lambda image, queries: np.full((1, 20, 40), 1.5, np.float32), "cpu"
    )
    with pytest.raises(RuntimeError, match="outside"):
        bad_range.segment(Image.new("RGB", (40, 20)), ["a cat"])


def test_mask_iou_and_mask_bbox():
    a = np.zeros((10, 10), bool)
    a[2:6, 2:6] = True
    b = np.zeros((10, 10), bool)
    b[4:8, 4:8] = True
    assert mask_iou(a, a) == 1.0
    assert mask_iou(a, b) == pytest.approx(4 / 28)
    assert mask_iou(a, np.zeros((10, 10), bool)) == 0.0
    assert mask_iou(np.zeros((10, 10), bool), np.zeros((10, 10), bool)) == 0.0
    with pytest.raises(ValueError, match="shapes differ"):
        mask_iou(a, np.zeros((5, 5), bool))
    assert mask_bbox(a) == [2, 2, 6, 6]
    assert mask_bbox(np.zeros((10, 10), bool)) is None
