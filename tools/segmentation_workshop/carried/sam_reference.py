"""Promptable image segmentation over the pinned ``facebook/sam-vit-base`` (SAM v1, ViT-B) checkpoint.

Weights load only from a digest-verified local snapshot (``weights/<key>/``) or, when explicitly allowed,
from the Hugging Face Hub at the pinned revision. One task method, ``segment``: one object per call from point
clicks and/or one box, returning boolean masks at input resolution plus the model's predicted IoU per mask.
"""

from __future__ import annotations

# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width
import hashlib
import json
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "facebook/sam-vit-base"
MODEL_REVISION = "70c1a07f894ebb5b307fd9eaaee97b9dfc16068f"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "sam-vit-base"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Input ceilings. The processor resizes the longest edge to 1024 and pads to 1024x1024 (see
# preprocessor_config.json), so model cost is fixed; the caller's resolution only sets the size of the
# up-sampled output masks. Prompts are one object per call: up to MAX_PROMPTS point clicks (label 1 =
# foreground, 0 = background) and/or one xyxy box.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_PROMPTS = 16
NUM_MULTIMASK_OUTPUTS = 3  # config.json mask_decoder_config.num_multimask_outputs
MASK_THRESHOLD = 0.0  # logits above this become True in the binarised masks (processor default)
PARAMETER_COUNT = 93_735_472
MASK_DECODER_PARAMETERS = 4_058_340
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = "1.0"
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
WEIGHT_FILE = "model.safetensors"
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
MAX_EVAL_RECORDS = 5_000
PROMPT_KINDS = ("point", "box", "mixed")


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
    """Intersection-over-union of two boolean masks of identical shape; the primitive behind any mIoU."""
    a = np.asarray(a)
    b = np.asarray(b)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if a.dtype != np.bool_ or b.dtype != np.bool_:
        raise TypeError("mask_iou expects boolean arrays")
    union = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / union) if union else 0.0


def validate_image(image: Any) -> Image.Image:
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    if min(width, height) < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {min(width, height)} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if max(width, height) > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {max(width, height)} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    return image.convert("RGB")


def validate_prompts(
    width: int,
    height: int,
    points: Sequence[Sequence[float]] | None,
    point_labels: Sequence[int] | None,
    box: Sequence[float] | None,
) -> tuple[list[list[float]] | None, list[int] | None, list[float] | None]:
    """Check one object's prompts: points inside the image with 0/1 labels, and/or one xyxy box inside it."""
    if points is None and box is None:
        raise ValueError("provide at least one of points or box")
    clean_points = clean_labels = None
    if points is not None:
        if isinstance(points, str) or not isinstance(points, Sequence):
            raise TypeError("points must be a sequence of [x, y] pairs")
        if not 1 <= len(points) <= MAX_PROMPTS:
            raise ValueError(f"point count {len(points)} outside 1..MAX_PROMPTS {MAX_PROMPTS}")
        if point_labels is None or len(point_labels) != len(points):
            raise ValueError("point_labels must be given with one 0/1 entry per point")
        clean_points, clean_labels = [], []
        for (x, y), label in zip(points, point_labels, strict=True):
            if not (0 <= x < width and 0 <= y < height):
                raise ValueError(f"point ({x}, {y}) outside image {width}x{height}")
            if label not in (0, 1) or isinstance(label, bool):
                raise ValueError(f"point label must be 0 or 1, got {label!r}")
            clean_points.append([float(x), float(y)])
            clean_labels.append(int(label))
    elif point_labels is not None:
        raise ValueError("point_labels given without points")
    clean_box = None
    if box is not None:
        if len(box) != 4:
            raise ValueError("box must be [x0, y0, x1, y1]")
        x0, y0, x1, y1 = (float(v) for v in box)
        if not (0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height):
            raise ValueError(f"box {box!r} is not a non-empty xyxy box inside image {width}x{height}")
        clean_box = [x0, y0, x1, y1]
    return clean_points, clean_labels, clean_box


INPUT_SCHEMA: dict[str, Any] = {
    "input": (
        "one PIL.Image.Image (any mode, converted to RGB) plus one object's prompts: point clicks "
        "and/or one xyxy box"
    ),
    "image_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "points": [1, MAX_PROMPTS],
    "point_labels": "one per point, 1 = foreground and 0 = background",
    "box": "at most one [x0, y0, x1, y1] inside the image with x0 < x1 and y0 < y1",
    "objects_per_call": 1,
    "multimask_outputs": NUM_MULTIMASK_OUTPUTS,
    "preprocessing": (
        "image converted to RGB; the processor resizes the longest edge to 1024 and pads to 1024x1024; "
        "returned masks are up-sampled "
        f"to the input resolution and binarised at logit MASK_THRESHOLD={MASK_THRESHOLD}"
    ),
}


def _check_inputs(
    image: Any,
    points: Any,
    point_labels: Any,
    box: Any,
    multimask: Any,
) -> tuple[Image.Image, list[list[float]] | None, list[int] | None, list[float] | None]:
    """Raise TypeError/ValueError naming the first violated ceiling; return the cleaned request.

    ``segment`` and ``validate_inputs`` both route through this function so their acceptance
    criteria cannot diverge.
    """
    rgb = validate_image(image)
    clean_points, clean_labels, clean_box = validate_prompts(
        rgb.width, rgb.height, points, point_labels, box
    )
    if not isinstance(multimask, bool):
        raise TypeError("multimask must be a bool")
    return rgb, clean_points, clean_labels, clean_box


def validate_inputs(
    image: Image.Image,
    *,
    points: Sequence[Sequence[float]] | None = None,
    point_labels: Sequence[int] | None = None,
    box: Sequence[float] | None = None,
    multimask: bool = True,
    names: Sequence[str] | None = None,
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, observations, request, verdict).

    Rejection is reported by raising exactly as ``segment`` would; a caller that wants the finding
    recorded catches the exception and stores ``str(exc)`` under ``findings``.
    """
    _rgb, clean_points, clean_labels, clean_box = _check_inputs(
        image, points, point_labels, box, multimask
    )
    if names is not None and len(names) != 1:
        raise ValueError("names must have exactly one entry (segment takes one image)")
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": [
            {
                "id": names[0] if names else "image-0",
                "mode": image.mode,
                "size": list(image.size),
                "n_points": 0 if clean_points is None else len(clean_points),
                "has_box": clean_box is not None,
            }
        ],
        "points": clean_points,
        "point_labels": clean_labels,
        "box": clean_box,
        "multimask": multimask,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    reference_mask: Any = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With a boolean ``reference_mask`` of the same shape as the returned masks the report carries one
    ``mask_iou`` entry per candidate as sample-sanity geometry evidence; without one the verdict is
    ``not-measurable`` and the report says what labelled data would make the task measurable.
    """
    masks = np.asarray(result["masks"])
    scores = [float(v) for v in result["iou_scores"]]
    best = int(np.argmax(scores)) if scores else None
    base = {
        "task": "promptable single-object image segmentation (point and/or box prompts)",
        "decision_rule": (
            "keep the candidate with the highest model-predicted IoU; the pipeline ships no "
            "acceptance threshold and does not choose for the caller"
        ),
        "score_semantics": (
            "iou_scores are the model's own uncalibrated predicted IoU for each candidate, not a "
            "measured overlap and not a probability; the regression head is unclipped, so a value "
            "may exceed 1.0"
        ),
        "sample_kind": sample_kind,
        "n_masks": int(masks.shape[0]) if masks.ndim == 3 else 0,
        "best_candidate": best,
        "iou_scores_model_predicted": scores,
        "mask_areas_px": [int(mask.sum()) for mask in masks] if masks.ndim == 3 else [],
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if reference_mask is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no ground-truth mask was supplied for the evaluated image",
            "needs": (
                "hand-labelled boolean masks for the prompted objects on your own images, scored with "
                "mask_iou per object and averaged into a mean IoU over a held-out set; no labelled "
                "mask set ships with this repository"
            ),
        }
    reference = np.asarray(reference_mask)
    return {
        **base,
        "metrics": [
            {
                "id": "mask_iou",
                "candidate": index,
                "value": mask_iou(masks[index], reference),
                "selected": index == best,
                "estimation": "one reference mask on a single scene, no dispersion estimate",
            }
            for index in range(masks.shape[0])
        ],
        "reference_area_px": int(reference.sum()),
        "verdict": "sample-sanity",
        "reason": (
            "one reference mask on one tutorial sample; geometry sanity evidence, not a segmentation "
            "benchmark"
        ),
        "needs": (
            "a labelled mask set from the deployment domain for any mean-IoU or boundary-quality claim"
        ),
    }


def _trainable_names(model: Any) -> list[str]:
    """Every parameter of the mask decoder (transformer, upscaling and IoU-prediction heads). The image encoder
    and the prompt encoder stay frozen."""
    return [name for name, _ in model.named_parameters() if name.startswith("mask_decoder.")]


def _check_artifact_manifest(manifest: Mapping[str, Any], artifact_dir: Path, base_sha256: str) -> None:
    """Refuse an adapter that names another base, another format or a file that does not match its digest."""
    if manifest.get("format") != ARTIFACT_FORMAT:
        raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
    base = manifest.get("base", {})
    if base.get("model_id") != MODEL_ID or base.get("revision") != MODEL_REVISION:
        raise ValueError(f"artifact was trained on {base.get('model_id')}@{base.get('revision')}, not {MODEL_ID}@{MODEL_REVISION}")
    if base.get("weight_sha256") != base_sha256:
        raise ValueError("artifact base weight digest does not match the verified snapshot")
    files = manifest.get("files") or []
    if len(files) != 1 or files[0].get("path") != ADAPTER_WEIGHTS:
        raise ValueError(f"artifact manifest must list exactly {ADAPTER_WEIGHTS}")
    weights = artifact_dir / ADAPTER_WEIGHTS
    if not weights.is_file():
        raise FileNotFoundError(f"artifact weights missing: {weights}")
    size = weights.stat().st_size
    if size != files[0].get("bytes"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: size {size} != manifest {files[0].get('bytes')}")
    digest = _sha256(weights)
    if digest != files[0].get("sha256"):
        raise ValueError(f"{ADAPTER_WEIGHTS}: sha256 {digest} != manifest {files[0].get('sha256')}")
    names = manifest.get("tensors") or []
    if not names or any(not str(n).startswith("mask_decoder.") for n in names):
        raise ValueError("artifact tensors must all belong to the mask decoder")
    if (manifest.get("adapter") or {}).get("prompts") not in PROMPT_KINDS:
        raise ValueError(f"artifact adapter.prompts must be one of {PROMPT_KINDS}")


def _low_res_target(mask: np.ndarray) -> Any:
    """A record's boolean mask in the model's low-resolution frame: resized so the longest side is 1024, padded to
    1024x1024 at the bottom/right (as the processor pads the image), then downsampled to the 256x256 decoder grid."""
    import torch

    target = torch.from_numpy(np.asarray(mask, dtype=np.float32))[None, None]
    height, width = target.shape[2], target.shape[3]
    scale = 1024 / max(height, width)
    new_h, new_w = int(round(height * scale)), int(round(width * scale))
    resized = torch.nn.functional.interpolate(target, size=(new_h, new_w), mode="bilinear", align_corners=False)
    canvas = torch.zeros(1, 1, 1024, 1024)
    canvas[:, :, :new_h, :new_w] = resized
    return torch.nn.functional.interpolate(canvas, size=(256, 256), mode="bilinear", align_corners=False)[0]


@dataclass
class SAMViTSegmentationPipeline:
    """Promptable image segmentation (points/box -> masks) over the pinned SAM ViT-B checkpoint.

    Prompted mode only (`SamModel` + `SamProcessor`); automatic grid-prompt mask generation is not exposed."""

    _runner: Callable[..., tuple[np.ndarray, list[float]]]
    device: str
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> SAMViTSegmentationPipeline:
        import torch
        from transformers import SamModel, SamProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        weight_sha256 = None
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            with open(root / MANIFEST_NAME, encoding="utf-8") as handle:
                entries = json.load(handle).get("files", [])
            weight_sha256 = next((e["sha256"] for e in entries if e["path"] == WEIGHT_FILE), None)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        processor = SamProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = SamModel.from_pretrained(
            source, revision=MODEL_REVISION, dtype=torch.float32, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()
        for param in model.parameters():
            param.requires_grad_(False)

        def runner(image, points, labels, box, multimask) -> tuple[np.ndarray, list[float]]:
            prompt_kwargs: dict[str, Any] = {}
            if points is not None:
                prompt_kwargs["input_points"] = [[points]]
                prompt_kwargs["input_labels"] = [[labels]]
            if box is not None:
                prompt_kwargs["input_boxes"] = [[box]]
            inputs = processor(images=image, return_tensors="pt", **prompt_kwargs).to(resolved_device)
            with torch.inference_mode():
                outputs = model(**inputs, multimask_output=multimask)
            # SAM v1 pads the resized image to 1024x1024; reshaped_input_sizes lets post-processing strip it.
            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(),
                inputs["original_sizes"].cpu(),
                inputs["reshaped_input_sizes"].cpu(),
                mask_threshold=MASK_THRESHOLD,
            )[0]
            return masks[0].numpy().astype(np.bool_), [float(v) for v in outputs.iou_scores[0, 0].tolist()]

        return cls(runner, resolved_device, model, processor, weight_sha256)

    def segment(
        self,
        image: Image.Image,
        *,
        points: Sequence[Sequence[float]] | None = None,
        point_labels: Sequence[int] | None = None,
        box: Sequence[float] | None = None,
        multimask: bool = True,
    ) -> dict[str, Any]:
        """Segment one object; returns K boolean masks (K = 3 with multimask, else 1) at input resolution."""
        rgb, clean_points, clean_labels, clean_box = _check_inputs(
            image, points, point_labels, box, multimask
        )
        masks, iou_scores = self._runner(rgb, clean_points, clean_labels, clean_box, multimask)
        masks = np.asarray(masks)
        expected = (NUM_MULTIMASK_OUTPUTS if multimask else 1, rgb.height, rgb.width)
        if masks.dtype != np.bool_ or masks.shape != expected or len(iou_scores) != expected[0]:
            raise RuntimeError(f"backend returned {masks.shape} {masks.dtype}, {len(iou_scores)} scores")
        return {
            "masks": masks,
            "iou_scores": [float(v) for v in iou_scores],
            "multimask": multimask,
            "points": clean_points,
            "point_labels": clean_labels,
            "box": clean_box,
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise RuntimeError("this pipeline has no loaded model (injected runner); use from_pretrained for evaluate/adapt")
        return self._model, self._processor


    def _prompt_kwargs(self, record: Mapping[str, Any], prompt: str) -> dict[str, Any]:
        if prompt == "point":
            return {"points": [record["point"]], "point_labels": [1]}
        if prompt == "box":
            return {"box": record["box"]}
        raise ValueError(f"prompt must be 'point' or 'box', got {prompt!r}")


    def predict_mask(self, record: Mapping[str, Any], *, prompt: str = "point") -> np.ndarray:
        """One boolean mask for a record: `segment` with the record's point (label 1) or box, keeping the multimask
        output the model itself scores highest — the single-mask policy the corpus measures use."""
        result = self.segment(record["image"], multimask=True, **self._prompt_kwargs(record, prompt))
        return result["masks"][int(np.argmax(result["iou_scores"]))]


    def evaluate(self, records: Sequence[Mapping[str, Any]], *, prompt: str = "point", progress: Callable[[int, int], None] | None = None) -> dict[str, Any]:
        """Segment every validated record from one prompt kind and score the masks with `metrics.segmentation_metrics`
        (mean IoU and hit rate, overall and per category)."""
        from .metrics import segmentation_metrics
        from .samples import validate_dataset

        if prompt not in ("point", "box"):
            raise ValueError("prompt must be 'point' or 'box'")
        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        masks = []
        for i, record in enumerate(checked):
            masks.append(self.predict_mask(record, prompt=prompt))
            if progress is not None:
                progress(i + 1, len(checked))
        metrics = segmentation_metrics(masks, checked)
        metrics.update(
            {
                "prompt": prompt,
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics


    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 6,
        lr: float = 5e-5,
        prompts: str = "mixed",
        seed: int = 0,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the mask decoder only: the frozen image encoder embeds every training image once
        (cached), the frozen prompt encoder embeds the record's point or box (`prompts`: 'point', 'box' or 'mixed' —
        a seeded coin per record and epoch), and the decoder's single-mask logits are trained against the target mask
        in the 256x256 low-resolution frame with binary cross-entropy plus a soft Dice term. AdamW (no weight decay),
        gradient clipping at 1.0, one record per step, seeded shuffling, no scheduler. Epoch 0 records the frozen
        model's validation metrics; the epoch with the highest mean of the validation point- and box-prompt IoU is kept
        (the final one
        without a validation split). On any exception the frozen weights are restored."""
        model, processor = self._require_model()  # refuse before importing torch
        import torch

        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 50:
            raise ValueError("epochs must be an int in 1..50")
        if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-2:
            raise ValueError("lr must be in (0, 1e-2]")
        if prompts not in PROMPT_KINDS:
            raise ValueError(f"prompts must be one of {PROMPT_KINDS}")
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        names = _trainable_names(model)
        name_set = set(names)
        device = torch.device(self.device)
        frozen_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
        previous_adapter = self.adapter
        cudnn_flags = (torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark)
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False  # repeatable on one device
        history: list[dict[str, Any]] = []
        started = time.perf_counter()

        def _val() -> dict[str, Any] | None:
            if val_checked is None:
                return None
            point = self.evaluate(val_checked, prompt="point")
            box = self.evaluate(val_checked, prompt="box")
            return {
                "iou": point["iou"],
                "hit_rate": point["hit_rate"],
                "box_iou": box["iou"],
                "box_hit_rate": box["hit_rate"],
                "score": (point["iou"] + box["iou"]) / 2.0,
                "n": point["n"],
            }

        try:
            for param in model.parameters():
                param.requires_grad_(False)
            params = []
            for name, param in model.named_parameters():
                if name in name_set:
                    param.requires_grad_(True)
                    params.append(param)
            n_trainable = sum(p.numel() for p in params)
            embed_started = time.perf_counter()
            embeddings: dict[str, Any] = {}
            with torch.no_grad():  # not inference_mode: the cached embeddings feed a backward pass
                for record in train_checked:
                    pixel_values = processor(images=record["image"], return_tensors="pt")["pixel_values"].to(device)
                    embeddings[record["id"]] = model.get_image_embeddings(pixel_values).clone()
            embed_seconds = round(time.perf_counter() - embed_started, 3)
            targets = {record["id"]: _low_res_target(record["mask"]).to(device) for record in train_checked}
            entry = {"epoch": 0, "train_loss": None, "val": _val(), "note": "frozen model"}
            history.append(entry)
            if progress is not None:
                progress(entry)
            best_epoch, best_score = 0, (history[0]["val"] or {}).get("score", -1.0)
            best_state = frozen_state
            optimizer = torch.optim.AdamW(params, lr=float(lr), weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            for epoch in range(1, epochs + 1):
                model.train()
                order = list(train_checked)
                rng.shuffle(order)
                losses = []
                for record in order:
                    use_box = prompts == "box" or (prompts == "mixed" and rng.random() < 0.5)
                    if use_box:
                        encoded = processor(images=record["image"], input_boxes=[[record["box"]]], return_tensors="pt")
                        output = model(image_embeddings=embeddings[record["id"]], input_boxes=encoded["input_boxes"].to(device), multimask_output=False)
                    else:
                        encoded = processor(images=record["image"], input_points=[[[record["point"]]]], input_labels=[[[1]]], return_tensors="pt")
                        output = model(image_embeddings=embeddings[record["id"]], input_points=encoded["input_points"].to(device), input_labels=encoded["input_labels"].to(device), multimask_output=False)
                    logits = output.pred_masks[0, 0]
                    target = targets[record["id"]]
                    bce = torch.nn.functional.binary_cross_entropy_with_logits(logits, target)
                    prob = torch.sigmoid(logits)
                    dice = 1.0 - (2.0 * (prob * target).sum() + 1.0) / (prob.sum() + target.sum() + 1.0)
                    loss = bce + dice
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(params, 1.0)
                    optimizer.step()
                    losses.append(float(loss.detach()))
                model.eval()
                entry = {"epoch": epoch, "train_loss": sum(losses) / len(losses), "val": _val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                if val_checked is None or entry["val"]["score"] > best_score:
                    best_epoch, best_score = epoch, (entry["val"] or {}).get("score", -1.0)
                    best_state = {k: v.detach().clone() for k, v in model.state_dict().items() if k in name_set}
            model.load_state_dict(best_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
        except BaseException:
            model.load_state_dict(frozen_state, strict=False)
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            self.adapter = previous_adapter
            raise
        finally:
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
        self.adapter = {
            "prompts": prompts,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "epochs": epochs,
            "best_epoch": best_epoch,
            "selection": "highest mean of validation point- and box-prompt IoU" if val_checked is not None else "final epoch (no validation split)",
            "loss": "binary cross-entropy + soft Dice on the 256x256 single-mask logits",
            "lr": float(lr),
            "seed": seed,
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "embedding_seconds": embed_seconds,
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)


    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained mask-decoder tensors as safetensors plus a manifest naming the base, the digests and the
        training configuration. Requires a prior `adapt`."""
        model, _processor = self._require_model()  # refuse before importing torch
        import torch
        from safetensors.torch import save_file

        if self.adapter is None:
            raise RuntimeError("nothing to save: call adapt() first")
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = list(self.adapter["trainable_names"])
        state = model.state_dict()
        tensors = {name: state[name].detach().cpu().contiguous() for name in names}
        weights = out / ADAPTER_WEIGHTS
        save_file(tensors, str(weights), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "version": ARTIFACT_VERSION,
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHT_FILE, "weight_sha256": self.weight_sha256},
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter["history"],
            "tensors": names,
            "files": [{"path": ADAPTER_WEIGHTS, "bytes": weights.stat().st_size, "sha256": _sha256(weights)}],
            "torch": torch.__version__,
            "metadata": dict(metadata or {}),
        }
        with open(out / ADAPTER_MANIFEST, "w", encoding="utf-8") as handle:
            json.dump(manifest, handle, indent=2, ensure_ascii=False)
        return out


    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Overlay a saved adapter onto this (freshly loaded) pipeline after checking its manifest, digest and exact
        tensor set. Refuses tensors outside the mask decoder."""
        model, _processor = self._require_model()  # refuse before importing safetensors
        from safetensors.torch import load_file

        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        _check_artifact_manifest(manifest, artifact, self.weight_sha256 or "")
        expected = _trainable_names(model)
        if sorted(manifest["tensors"]) != sorted(expected):
            raise ValueError("artifact tensor set does not match its recorded configuration")
        tensors = load_file(str(artifact / ADAPTER_WEIGHTS))
        if sorted(tensors) != sorted(expected):
            raise ValueError("artifact tensor names differ from the manifest")
        state = model.state_dict()
        for name, tensor in tensors.items():
            if tuple(tensor.shape) != tuple(state[name].shape):
                raise ValueError(f"artifact tensor {name} has shape {tuple(tensor.shape)}, base has {tuple(state[name].shape)}")
        model.load_state_dict({k: v.to(state[k].device, state[k].dtype) for k, v in tensors.items()}, strict=False)
        model.eval()
        self.adapter = {**manifest["adapter"], "trainable_names": expected, "history": manifest.get("history", [])}
        return dict(self.adapter)


    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> SAMViTSegmentationPipeline:
        """Load the verified base snapshot, then overlay the adapter (verified before deserialising)."""
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
