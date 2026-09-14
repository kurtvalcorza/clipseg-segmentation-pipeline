"""Role-helper contract: validate_inputs (validation stage) and evaluation_report (evaluation stage)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from clipseg_segmentation_pipeline import (
    INPUT_SCHEMA,
    MASK_THRESHOLD,
    MAX_IMAGE_SIDE,
    MAX_PROMPTS,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    mask_iou,
    validate_inputs,
)

PROMPTS = ["a red circle", "a blue square"]


def _image(width: int = 64, height: int = 48) -> Image.Image:
    return Image.new("RGB", (width, height), "white")


def _mask(height: int, width: int, box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros((height, width), bool)
    x0, y0, x1, y1 = box
    mask[y0:y1, x0:x1] = True
    return mask


def _result(masks: dict[str, np.ndarray], threshold: float = MASK_THRESHOLD) -> dict:
    return {
        "segments": [
            {
                "prompt": prompt,
                "mask": mask,
                "probability": mask.astype(np.float32),
                "area_fraction": float(mask.mean()),
                "max_probability": float(mask.max()),
                "bbox": None,
            }
            for prompt, mask in masks.items()
        ],
        "queries": list(masks),
        "threshold": threshold,
        "width": 64,
        "height": 48,
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs(_image(), ["A red circle", "a blue square."], names=["scene.png"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["image_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["prompts"] == [1, MAX_PROMPTS]
    assert manifest["inputs"] == [{"id": "scene.png", "mode": "RGB", "size": [64, 48], "n_prompts": 2}]
    assert manifest["queries"] == PROMPTS
    assert manifest["threshold"] == MASK_THRESHOLD
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_default_id_and_explicit_threshold() -> None:
    manifest = validate_inputs(_image(), ["a cat"], threshold=0.3)
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["threshold"] == 0.3


def test_validate_inputs_rejects_like_segment() -> None:
    with pytest.raises(TypeError, match="not a single string"):
        validate_inputs(_image(), "a cat")
    with pytest.raises(ValueError, match="MAX_PROMPTS"):
        validate_inputs(_image(), [])
    with pytest.raises(ValueError, match="distinct"):
        validate_inputs(_image(), ["a cat", "A cat"])
    with pytest.raises(ValueError, match="threshold"):
        validate_inputs(_image(), ["a cat"], threshold=-0.1)
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(8, 8), ["a cat"])
    with pytest.raises(ValueError, match="exactly one entry"):
        validate_inputs(_image(), ["a cat"], names=["a", "b"])


def test_evaluation_report_not_measurable_without_reference_masks() -> None:
    result = _result({"a red circle": _mask(48, 64, (0, 0, 32, 24))})
    report = evaluation_report(result, sample_kind="BYOD")
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["n_prompts"] == 1 and report["area_fractions"] == {"a red circle": 0.25}
    assert report["threshold"] == MASK_THRESHOLD
    assert "mean IoU" in report["needs"]
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)
    assert "uncalibrated" in report["decision_rule"]


def test_evaluation_report_sample_sanity_with_reference_masks() -> None:
    predicted = {
        "a red circle": _mask(48, 64, (0, 0, 32, 24)),
        "a blue square": _mask(48, 64, (40, 30, 60, 46)),
    }
    references = {
        "A red circle.": _mask(48, 64, (0, 0, 32, 24)),
        "a blue square": _mask(48, 64, (32, 24, 64, 48)),
    }
    report = evaluation_report(_result(predicted), references)
    assert report["verdict"] == "sample-sanity"
    by_ref = {entry["reference"]: entry for entry in report["metrics"] if entry["id"] == "mask_iou"}
    assert by_ref["a red circle"]["value"] == 1.0
    expected = mask_iou(predicted["a blue square"], references["a blue square"])
    assert by_ref["a blue square"]["value"] == pytest.approx(expected)
    assert by_ref["a blue square"]["reference_area_fraction"] == pytest.approx(0.25)
    miou = [entry for entry in report["metrics"] if entry["id"] == "miou"][0]
    assert miou["value"] == pytest.approx((1.0 + expected) / 2)
    assert "not a segmentation benchmark" in report["reason"]


def test_evaluation_report_rejects_unknown_reference_phrase() -> None:
    result = _result({"a red circle": _mask(48, 64, (0, 0, 32, 24))})
    with pytest.raises(ValueError, match="not among the segmented prompts"):
        evaluation_report(result, {"a cat": _mask(48, 64, (0, 0, 1, 1))})
