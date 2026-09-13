"""Text-prompted (zero-shot) image segmentation with the pinned ``CIDAS/clipseg-rd64-refined`` checkpoint.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the CLIPSeg architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "CIDAS/clipseg-rd64-refined"
MODEL_REVISION = "999e0328d9e10b484360c477313983f9afdd7050"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "clipseg-rd64-refined"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Threshold on the per-pixel sigmoid of the decoder logits. 0.5 is the natural cut of a sigmoid and
# the value the smoke run used; the sigmoid is not calibrated, so the deployment owns tuning it on its
# own labelled masks.
MASK_THRESHOLD = 0.5
# Decoder output resolution: logits are 352x352 for every input (preprocessor_config.json resizes to
# 352x352 without preserving aspect ratio); the pipeline resamples the probability map back to the
# input size bilinearly.
LOGIT_SIZE = 352
# Input ceilings. Image cost is bounded by the fixed resize; the side ceiling only guards memory during
# decoding and resampling. Each prompt is one CLIP text query (77-token context); phrases are short.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_PROMPTS = 16
MAX_PROMPT_CHARS = 64
MAX_TEXT_TOKENS = 77


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks of the same shape; the building block for any mIoU."""
    a_bool, b_bool = np.asarray(a, dtype=bool), np.asarray(b, dtype=bool)
    if a_bool.shape != b_bool.shape:
        raise ValueError(f"mask shapes differ: {a_bool.shape} vs {b_bool.shape}")
    union = np.logical_or(a_bool, b_bool).sum()
    return float(np.logical_and(a_bool, b_bool).sum() / union) if union else 0.0


def mask_bbox(mask: np.ndarray) -> list[int] | None:
    """Tight xyxy pixel box around the true pixels of a mask, or ``None`` for an empty mask."""
    rows = np.flatnonzero(np.asarray(mask, dtype=bool).any(axis=1))
    cols = np.flatnonzero(np.asarray(mask, dtype=bool).any(axis=0))
    if rows.size == 0 or cols.size == 0:
        return None
    return [int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1]


def format_prompts(prompts: Sequence[str]) -> list[str]:
    """Validate a list of phrases and normalise them: stripped, whitespace-collapsed, lower-cased,
    trailing full stop removed, distinct. The pipeline passes the caller's phrases through otherwise
    unchanged (the upstream examples use plain noun phrases such as "a cat")."""
    if isinstance(prompts, str) or not isinstance(prompts, Sequence):
        raise TypeError("prompts must be a list of phrases, not a single string")
    if not 1 <= len(prompts) <= MAX_PROMPTS:
        raise ValueError(f"prompt count {len(prompts)} outside 1..MAX_PROMPTS {MAX_PROMPTS}")
    cleaned: list[str] = []
    for phrase in prompts:
        if not isinstance(phrase, str):
            raise TypeError(f"prompt must be str, got {type(phrase).__name__}")
        text = " ".join(phrase.split()).strip().rstrip(".").strip().lower()
        if not text:
            raise ValueError("prompt phrases must not be empty")
        if len(text) > MAX_PROMPT_CHARS:
            raise ValueError(
                f"prompt {text[:12]!r}... is {len(text)} chars > MAX_PROMPT_CHARS {MAX_PROMPT_CHARS}"
            )
        cleaned.append(text)
    if len(set(cleaned)) != len(cleaned):
        raise ValueError("prompt phrases must be distinct after normalisation")
    return cleaned


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def _check_threshold(name: str, value: Any) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be a number in [0, 1], got {value!r}")
    return float(value)


INPUT_SCHEMA: dict[str, Any] = {
    "input": "one PIL.Image.Image (any mode, converted to RGB) plus 1..MAX_PROMPTS free-text phrases",
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "prompts": [1, MAX_PROMPTS],
    "prompt_chars": [1, MAX_PROMPT_CHARS],
    "prompt_tokens_per_query": [1, MAX_TEXT_TOKENS],
    "threshold": [0.0, 1.0],
    "preprocessing": (
        f"image converted to RGB and resized to {LOGIT_SIZE}x{LOGIT_SIZE} (aspect ratio not preserved, "
        "ImageNet mean/std); phrases normalised into one CLIP text query each (format_prompts); the "
        f"decoder's {LOGIT_SIZE}x{LOGIT_SIZE} logits are passed through a sigmoid and resampled "
        "bilinearly to the input size; the mask is probability >= threshold"
    ),
    "output": (
        "per prompt: a boolean mask and a float32 probability map at input resolution, the mask's area "
        "fraction, tight box and maximum probability; probabilities are uncalibrated sigmoids"
    ),
}


def _check_inputs(image: Any, prompts: Any, threshold: Any) -> tuple[Image.Image, list[str], float]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the checked request.

    ``segment`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    queries = format_prompts(prompts)
    checked = _check_threshold("threshold", threshold)
    return rgb, queries, checked


def validate_inputs(
    image: Image.Image,
    prompts: Sequence[str],
    *,
    threshold: float = MASK_THRESHOLD,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``segment`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, queries, checked = _check_inputs(image, prompts, threshold)
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (segment takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[0] if names else "image-0",
                "mode": image.mode,
                "size": list(image.size),
                "n_prompts": len(prompts),
            }
        ],
        "queries": queries,
        "threshold": checked,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    reference_masks: Mapping[str, np.ndarray] | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``reference_masks`` (phrase -> boolean mask at input resolution) the report carries one
    ``mask_iou`` entry per reference and their mean (``miou``) as sample-sanity evidence; without them
    the verdict is ``not-measurable`` and the report says what labelled data would make the task
    measurable.
    """
    segments = list(result["segments"])
    by_prompt = {segment["prompt"]: segment for segment in segments}
    base = {
        "task": "zero-shot (text-prompted) binary segmentation, one mask per phrase",
        "decision_rule": (
            "each phrase yields a per-pixel sigmoid over the decoder logits; a pixel belongs to the mask "
            "when that sigmoid reaches the threshold; the sigmoid is uncalibrated and masks of different "
            "phrases are independent (they may overlap or leave pixels unassigned)"
        ),
        "threshold": result.get("threshold", MASK_THRESHOLD),
        "sample_kind": sample_kind,
        "n_prompts": len(segments),
        "area_fractions": {segment["prompt"]: segment["area_fraction"] for segment in segments},
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if not reference_masks:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no reference masks were supplied for the evaluated image",
            "needs": (
                "labelled masks on your own images with a phrase vocabulary matching the prompts, scored "
                "per phrase with mask_iou and aggregated into mean IoU at a stated threshold; no such "
                "labelled set ships with this repository"
            ),
        }
    metrics = []
    for phrase, reference in reference_masks.items():
        key = format_prompts([phrase])[0]
        if key not in by_prompt:
            raise ValueError(f"reference phrase {phrase!r} was not among the segmented prompts")
        metrics.append(
            {
                "id": "mask_iou",
                "reference": key,
                "value": mask_iou(by_prompt[key]["mask"], reference),
                "reference_area_fraction": float(np.asarray(reference, dtype=bool).mean()),
                "predicted_area_fraction": by_prompt[key]["area_fraction"],
                "estimation": "one reference mask per phrase on a single scene, no dispersion estimate",
            }
        )
    miou = sum(entry["value"] for entry in metrics) / len(metrics)
    metrics.append(
        {
            "id": "miou",
            "value": miou,
            "estimation": f"mean of {len(metrics)} mask_iou value(s) on one scene, no dispersion estimate",
        }
    )
    return {
        **base,
        "metrics": metrics,
        "verdict": "sample-sanity",
        "reason": (
            f"{len(metrics) - 1} reference mask(s) on one tutorial sample; geometry sanity evidence, "
            "not a segmentation benchmark"
        ),
        "needs": (
            "a labelled mask set from the deployment domain with a matching phrase vocabulary for any "
            "mean-IoU claim"
        ),
    }


@dataclass
class ClipSegSegmentationPipeline:
    """Text-prompted (zero-shot) binary segmentation over the pinned CLIPSeg rd64-refined checkpoint."""

    _runner: Callable[[Image.Image, list[str]], np.ndarray]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ClipSegSegmentationPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import CLIPSegForImageSegmentation, CLIPSegProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = CLIPSegProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = CLIPSegForImageSegmentation.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, dtype=torch.float32, **kwargs
        )
        model = model.to(resolved_device).eval()

        def runner(image: Image.Image, queries: list[str]) -> np.ndarray:
            # One image copy per phrase: CLIPSeg conditions the decoder on each text query separately.
            inputs = processor(
                text=queries, images=[image] * len(queries), padding=True, return_tensors="pt"
            ).to(resolved_device)
            with torch.inference_mode():
                logits = model(**inputs).logits
            if logits.dim() == 2:  # a single prompt may come back squeezed
                logits = logits.unsqueeze(0)
            probs = torch.sigmoid(logits).unsqueeze(1)
            probs = torch.nn.functional.interpolate(
                probs, size=(image.height, image.width), mode="bilinear", align_corners=False
            )
            return probs.squeeze(1).float().cpu().numpy()

        return cls(runner, resolved_device)

    def segment(
        self,
        image: Image.Image,
        prompts: Sequence[str],
        *,
        threshold: float = MASK_THRESHOLD,
    ) -> dict[str, Any]:
        """Segment each phrase in ``prompts``; masks and probability maps are at input resolution."""
        rgb, queries, checked = _check_inputs(image, prompts, threshold)
        probs = np.asarray(self._runner(rgb, queries), dtype=np.float32)
        if probs.shape != (len(queries), rgb.height, rgb.width):
            raise RuntimeError(
                f"backend returned probability maps of shape {probs.shape}, "
                f"expected {(len(queries), rgb.height, rgb.width)}"
            )
        if probs.min() < 0.0 or probs.max() > 1.0:
            raise RuntimeError("backend returned probabilities outside [0, 1]")
        segments = []
        for query, prob in zip(queries, probs, strict=True):
            mask = prob >= checked
            segments.append(
                {
                    "prompt": query,
                    "mask": mask,
                    "probability": prob,
                    "area_fraction": float(mask.mean()),
                    "max_probability": float(prob.max()),
                    "bbox": mask_bbox(mask),
                }
            )
        return {
            "segments": segments,
            "queries": queries,
            "threshold": checked,
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
