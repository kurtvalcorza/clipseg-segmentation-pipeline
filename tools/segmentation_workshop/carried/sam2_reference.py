from __future__ import annotations

import copy
import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "facebook/sam2.1-hiera-small"
MODEL_REVISION = "ee5bba1d82bb8749febdf90f45e84b687142ba03"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "sam2.1-hiera-small"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
ARTIFACT_FORMAT = "org.valcorza.sam2.mask-decoder-adapter"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_MANIFEST_NAME = "manifest.json"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
TRAINABLE_PREFIXES = ("mask_decoder.output_hypernetworks_mlps.0.",)
ADAPTATION_METHOD = "frozen-backbone-mask-hypernetwork-gradient-adaptation"
MAX_ADAPTATION_RECORDS = 128
MAX_ADAPTATION_PIXELS = 32 * 1024 * 1024

_ADAPTATION_RUN_FIELDS = (
    "epochs",
    "learning_rate",
    "batch_size",
    "seed",
    "loss",
    "train_manifest",
    "validation_manifest",
    "baseline_validation_mask_iou",
    "history",
    "weight_delta_l2",
)
_DATASET_MANIFEST_FIELDS = {
    "records",
    "unique_ids",
    "positive_pixels",
    "dataset_sha256",
    "verdict",
}
_ADAPTATION_BASE_FIELDS = {
    "method",
    "trainable_prefixes",
    "trainable_parameters",
    "frozen_parameters",
}
_ADAPTATION_METADATA_FIELDS = {
    *_ADAPTATION_BASE_FIELDS,
    *_ADAPTATION_RUN_FIELDS,
    "training_runs",
    "artifact_lineage",
}
_ARTIFACT_LINEAGE_FIELDS = {"manifest_sha256", "weights_sha256", "producer"}

# Input ceilings. The processor resizes every image to 1024x1024 (preprocessor_config.json), so model cost is
# fixed; the caller's resolution only sets the size of the up-sampled output masks. Prompts are one object per
# call: up to MAX_PROMPTS point clicks (label 1 = foreground, 0 = background) and/or one xyxy box.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 16
MAX_PROMPTS = 16
NUM_MULTIMASK_OUTPUTS = 3  # config.json mask_decoder_config.num_multimask_outputs
MASK_THRESHOLD = 0.0  # logits above this become True in the binarised masks (processor default)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_positive_finite_number(value: Any) -> bool:
    return (
        isinstance(value, int | float)
        and not isinstance(value, bool)
        and math.isfinite(float(value))
        and value > 0
    )


def _validate_dataset_manifest(manifest: Any, *, label: str) -> None:
    if not isinstance(manifest, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if set(manifest) != _DATASET_MANIFEST_FIELDS:
        raise ValueError(f"{label} does not match the dataset manifest schema")
    records = manifest["records"]
    unique_ids = manifest["unique_ids"]
    positive_pixels = manifest["positive_pixels"]
    if (
        not isinstance(records, int)
        or isinstance(records, bool)
        or not 2 <= records <= MAX_ADAPTATION_RECORDS
    ):
        raise ValueError(f"{label} has an invalid record count")
    if not isinstance(unique_ids, int) or isinstance(unique_ids, bool) or unique_ids != records:
        raise ValueError(f"{label} has an invalid unique-id count")
    if (
        not isinstance(positive_pixels, int)
        or isinstance(positive_pixels, bool)
        or positive_pixels <= 0
    ):
        raise ValueError(f"{label} has an invalid positive-pixel count")
    digest = manifest["dataset_sha256"]
    if (
        not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{label} has an invalid dataset SHA-256")
    if manifest["verdict"] != "accepted":
        raise ValueError(f"{label} has an invalid verdict")


def _validate_adaptation_run(run: Any, *, label: str) -> None:
    if not isinstance(run, Mapping):
        raise ValueError(f"{label} must be a mapping")
    if set(run) != set(_ADAPTATION_RUN_FIELDS):
        raise ValueError(f"{label} does not match the adaptation run schema")
    missing = [key for key in _ADAPTATION_RUN_FIELDS if key not in run]
    if missing:
        raise ValueError(f"{label} is missing required fields: {missing}")
    epochs = run["epochs"]
    batch_size = run["batch_size"]
    seed = run["seed"]
    if not isinstance(epochs, int) or isinstance(epochs, bool) or epochs <= 0:
        raise ValueError(f"{label} has invalid epochs")
    if not _is_positive_finite_number(run["learning_rate"]):
        raise ValueError(f"{label} has invalid learning rate")
    if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size <= 0:
        raise ValueError(f"{label} has invalid batch size")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"{label} has invalid seed")
    if run["loss"] != "binary-cross-entropy-plus-soft-dice":
        raise ValueError(f"{label} has an unsupported loss")
    _validate_dataset_manifest(run["train_manifest"], label=f"{label} train manifest")
    validation_manifest = run["validation_manifest"]
    if validation_manifest is not None:
        _validate_dataset_manifest(
            validation_manifest, label=f"{label} validation manifest"
        )
    baseline = run["baseline_validation_mask_iou"]
    if validation_manifest is None and baseline is not None:
        raise ValueError(f"{label} has invalid validation baseline")
    if baseline is not None and (
        not isinstance(baseline, int | float)
        or isinstance(baseline, bool)
        or not math.isfinite(float(baseline))
        or not 0 <= float(baseline) <= 1
    ):
        raise ValueError(f"{label} has invalid validation baseline")
    history = run["history"]
    if not isinstance(history, list) or not history:
        raise ValueError(f"{label} requires non-empty training history")
    if len(history) != epochs:
        raise ValueError(f"{label} history length does not match epochs")
    expected_history_fields = {"epoch", "train_loss", "optimizer_steps"}
    if validation_manifest is not None:
        expected_history_fields.add("val_mask_iou")
    expected_steps = math.ceil(run["train_manifest"]["records"] / batch_size)
    for expected_epoch, item in enumerate(history, start=1):
        if not isinstance(item, Mapping) or set(item) != expected_history_fields:
            raise ValueError(f"{label} history entry schema is invalid")
        if item["epoch"] != expected_epoch or isinstance(item["epoch"], bool):
            raise ValueError(f"{label} history has an invalid epoch sequence")
        train_loss = item["train_loss"]
        if (
            not isinstance(train_loss, int | float)
            or isinstance(train_loss, bool)
            or not math.isfinite(float(train_loss))
            or train_loss < 0
        ):
            raise ValueError(f"{label} history has an invalid train loss")
        optimizer_steps = item["optimizer_steps"]
        if (
            not isinstance(optimizer_steps, int)
            or isinstance(optimizer_steps, bool)
            or optimizer_steps != expected_steps
        ):
            raise ValueError(f"{label} history has an invalid optimizer step count")
        if validation_manifest is not None:
            val_mask_iou = item["val_mask_iou"]
            if (
                not isinstance(val_mask_iou, int | float)
                or isinstance(val_mask_iou, bool)
                or not math.isfinite(float(val_mask_iou))
                or not 0 <= float(val_mask_iou) <= 1
            ):
                raise ValueError(f"{label} history has an invalid validation mask IoU")
    if not _is_positive_finite_number(run["weight_delta_l2"]):
        raise ValueError(f"{label} requires a positive finite weight delta")


def _validate_completed_adaptation_metadata(adaptation: Any) -> None:
    """Reject metadata that cannot identify a completed adapter before weights are applied."""
    if not isinstance(adaptation, Mapping):
        raise ValueError("artifact adaptation metadata must be a mapping")
    unknown_fields = set(adaptation) - _ADAPTATION_METADATA_FIELDS
    if unknown_fields:
        raise ValueError(
            f"artifact adaptation metadata contains unknown fields: {sorted(unknown_fields)}"
        )
    if adaptation.get("method") != ADAPTATION_METHOD:
        raise ValueError("artifact adaptation metadata has an unsupported or missing method")
    if adaptation.get("trainable_prefixes") != list(TRAINABLE_PREFIXES):
        raise ValueError("artifact adaptation metadata has incomplete trainable prefixes")
    trainable = adaptation.get("trainable_parameters")
    frozen = adaptation.get("frozen_parameters")
    if not isinstance(trainable, int) or isinstance(trainable, bool) or trainable <= 0:
        raise ValueError("artifact adaptation metadata has invalid trainable parameter count")
    if not isinstance(frozen, int) or isinstance(frozen, bool) or frozen < 0:
        raise ValueError("artifact adaptation metadata has invalid frozen parameter count")
    _validate_adaptation_run(
        _adaptation_run_snapshot(adaptation),
        label="artifact adaptation metadata current run",
    )
    if "training_runs" in adaptation:
        training_runs = adaptation["training_runs"]
        if not isinstance(training_runs, list) or not training_runs:
            raise ValueError("artifact adaptation metadata training_runs must be a non-empty list")
        for index, run in enumerate(training_runs):
            _validate_adaptation_run(run, label=f"artifact adaptation metadata training_runs[{index}]")
        if training_runs[-1] != _adaptation_run_snapshot(adaptation):
            raise ValueError("artifact adaptation metadata latest training run does not match current run")
    if "artifact_lineage" in adaptation:
        artifact_lineage = adaptation["artifact_lineage"]
        if not isinstance(artifact_lineage, list):
            raise ValueError("artifact adaptation metadata artifact_lineage must be a list of mappings")
        for index, item in enumerate(artifact_lineage):
            if not isinstance(item, Mapping) or set(item) != _ARTIFACT_LINEAGE_FIELDS:
                raise ValueError(
                    f"artifact adaptation metadata artifact_lineage[{index}] has invalid fields"
                )
            for digest_field in ("manifest_sha256", "weights_sha256"):
                digest = item[digest_field]
                if (
                    not isinstance(digest, str)
                    or len(digest) != 64
                    or any(character not in "0123456789abcdef" for character in digest)
                ):
                    raise ValueError(
                        f"artifact adaptation metadata artifact_lineage[{index}] has an invalid digest"
                    )
            producer = item["producer"]
            if (
                not isinstance(producer, Mapping)
                or set(producer) != {"pipelineId", "revision"}
                or producer["pipelineId"] != "sam2-segmentation-pipeline"
                or not isinstance(producer["revision"], str)
                or len(producer["revision"]) != 40
                or any(
                    character not in "0123456789abcdef" for character in producer["revision"]
                )
            ):
                raise ValueError(
                    f"artifact adaptation metadata artifact_lineage[{index}] has an invalid producer"
                )


def _adaptation_run_snapshot(adaptation: Mapping[str, Any]) -> dict[str, Any]:
    """Copy the per-run fields retained in cumulative adapter provenance."""
    return {
        key: json.loads(json.dumps(adaptation[key]))
        for key in _ADAPTATION_RUN_FIELDS
        if key in adaptation
    }


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
    "adapted_multimask": (
        "base pipelines default to three candidates; adapted pipelines default to one mask and reject "
        "multimask=True because the adapter trains only the single-mask output hypernetwork"
    ),
    "objects_per_call": 1,
    "multimask_outputs": NUM_MULTIMASK_OUTPUTS,
    "preprocessing": (
        "image converted to RGB; the processor resizes it to 1024x1024; returned masks are up-sampled "
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


def validate_segmentation_dataset(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate paired images, prompts, and boolean masks for bounded adaptation."""
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    if not 2 <= len(records) <= MAX_ADAPTATION_RECORDS:
        raise ValueError(f"record count {len(records)} outside 2..{MAX_ADAPTATION_RECORDS}")

    ids: list[str] = []
    digest = hashlib.sha256()
    positive_pixels = 0
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"record {index} id must be a non-empty string")
        if record_id in ids:
            raise ValueError(f"duplicate record id: {record_id}")
        ids.append(record_id)

        image = validate_image(record.get("image"))
        mask = np.asarray(record.get("mask"))
        if mask.dtype != np.bool_:
            raise TypeError(f"record {record_id} mask must be boolean")
        if mask.shape != (image.height, image.width):
            raise ValueError(
                f"record {record_id} mask shape {mask.shape} != image {(image.height, image.width)}"
            )
        area = int(mask.sum())
        if area == 0 or area == mask.size:
            raise ValueError(f"record {record_id} mask must contain foreground and background")
        clean_points, clean_labels, clean_box = validate_prompts(
            image.width,
            image.height,
            record.get("points"),
            record.get("point_labels"),
            record.get("box"),
        )
        if clean_points is not None and clean_labels is not None:
            for (x, y), label in zip(clean_points, clean_labels, strict=True):
                row = min(int(y), image.height - 1)
                column = min(int(x), image.width - 1)
                pixel_is_foreground = bool(mask[row, column])
                if pixel_is_foreground != bool(label):
                    raise ValueError(
                        f"record {record_id} point ({x}, {y}) label {label} contradicts target mask"
                    )
        if clean_box is not None:
            x0, y0 = (math.floor(value) for value in clean_box[:2])
            x1, y1 = (math.ceil(value) for value in clean_box[2:])
            if not bool(mask[y0:y1, x0:x1].any()):
                raise ValueError(f"record {record_id} box does not overlap target mask foreground")
        positive_pixels += area
        digest.update(record_id.encode("utf-8"))
        digest.update(
            json.dumps(
                {"image_shape": [image.height, image.width, 3], "mask_shape": list(mask.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(np.asarray(image, dtype=np.uint8).tobytes())
        digest.update(mask.tobytes())
        digest.update(
            json.dumps(
                {"points": clean_points, "point_labels": clean_labels, "box": clean_box},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )

    return {
        "records": len(records),
        "unique_ids": len(ids),
        "positive_pixels": positive_pixels,
        "dataset_sha256": digest.hexdigest(),
        "verdict": "accepted",
    }


def _segmentation_record_content_sha256(record: Mapping[str, Any]) -> str:
    """Fingerprint one validated image/mask sample without its ID or mutable prompts."""
    image = validate_image(record["image"])
    mask = np.asarray(record["mask"], dtype=np.bool_)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"image_shape": [image.height, image.width, 3], "mask_shape": list(mask.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(np.asarray(image, dtype=np.uint8).tobytes())
    digest.update(mask.tobytes())
    return digest.hexdigest()


@dataclass
class SAM2SegmentationPipeline:
    """Promptable image segmentation (points/box -> masks) over the pinned SAM 2.1 Hiera-Small checkpoint.

    Image mode only: `Sam2Model` + `Sam2Processor`. Video tracking (`Sam2VideoModel`) is not exposed."""

    _runner: Callable[..., tuple[np.ndarray, list[float]]]
    device: str
    model: Any | None = None
    processor: Any | None = None
    adaptation_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> SAM2SegmentationPipeline:
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
        from transformers import Sam2Model, Sam2Processor
        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = Sam2Processor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        # config.json declares Sam2VideoModel; image-mode Sam2Model loads the same weights (upstream README).
        model = Sam2Model.from_pretrained(source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs)
        model = model.to(resolved_device).eval()

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
            masks = processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), mask_threshold=MASK_THRESHOLD
            )[0]
            return masks[0].numpy().astype(np.bool_), [float(v) for v in outputs.iou_scores[0, 0].tolist()]

        return cls(runner, resolved_device, model=model, processor=processor)

    def freeze_for_adaptation(self) -> dict[str, int]:
        """Freeze the base model and enable the first mask-token hypernetwork only."""
        if self.model is None:
            raise RuntimeError("cannot configure adaptation without an underlying torch model")
        previous_config = dict(self.adaptation_config)
        trainable = frozen = 0
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = name.startswith(TRAINABLE_PREFIXES)
            if parameter.requires_grad:
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
        if trainable == 0:
            raise RuntimeError("SAM2 adaptation selected no trainable parameters")
        base_config = {
            "method": ADAPTATION_METHOD,
            "trainable_prefixes": list(TRAINABLE_PREFIXES),
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
        }
        if any(key in previous_config for key in _ADAPTATION_RUN_FIELDS):
            _validate_completed_adaptation_metadata(previous_config)
            previous_config.update(base_config)
            self.adaptation_config = previous_config
        else:
            self.adaptation_config = base_config
        return {"trainable_parameters": trainable, "frozen_parameters": frozen}

    def finetune(
        self,
        train_records: Sequence[Mapping[str, Any]],
        val_records: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 2,
        learning_rate: float = 2e-5,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Run bounded mask-hypernetwork fine-tuning with BCE plus soft-Dice loss."""
        import random

        import torch
        import torch.nn.functional as F
        from torch.optim import AdamW

        if self.model is None or self.processor is None:
            raise RuntimeError("cannot fine-tune without the underlying model and processor")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not isinstance(learning_rate, int | float) or isinstance(learning_rate, bool):
            raise TypeError("learning_rate must be numeric")
        if not 0 < float(learning_rate) <= 1e-2:
            raise ValueError("learning_rate must be in (0, 1e-2]")
        if not isinstance(seed, int) or isinstance(seed, bool):
            raise ValueError("seed must be an int")
        train_manifest = validate_segmentation_dataset(train_records)
        val_manifest = validate_segmentation_dataset(val_records) if val_records else None
        adaptation_pixels = sum(
            record["image"].width * record["image"].height for record in train_records
        ) + sum(record["image"].width * record["image"].height for record in (val_records or ()))
        if adaptation_pixels > MAX_ADAPTATION_PIXELS:
            raise ValueError(
                f"adaptation aggregate pixel count {adaptation_pixels} exceeds "
                f"MAX_ADAPTATION_PIXELS {MAX_ADAPTATION_PIXELS}"
            )
        if val_records:
            train_ids = {str(record["id"]) for record in train_records}
            val_ids = {str(record["id"]) for record in val_records}
            overlapping_ids = sorted(train_ids & val_ids)
            if overlapping_ids:
                raise ValueError(
                    f"train and validation records overlap by id: {overlapping_ids[:5]}"
                )
            train_content = {
                _segmentation_record_content_sha256(record) for record in train_records
            }
            val_content = {
                _segmentation_record_content_sha256(record) for record in val_records
            }
            if train_content & val_content:
                raise ValueError("train and validation records overlap by image/mask content")

        entry_config = copy.deepcopy(self.adaptation_config)
        entry_requires_grad = {
            name: parameter.requires_grad for name, parameter in self.model.named_parameters()
        }
        entry_parameter_devices = {
            name: parameter.device for name, parameter in self.model.named_parameters()
        }
        entry_parameter_grads = {
            name: None if parameter.grad is None else parameter.grad.detach().clone()
            for name, parameter in self.model.named_parameters()
        }
        entry_buffer_devices = {
            name: buffer.device for name, buffer in self.model.named_buffers()
        }
        entry_training = self.model.training

        def restore_entry_state() -> None:
            self.adaptation_config = copy.deepcopy(entry_config)
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    parameter.data = parameter.data.to(entry_parameter_devices[name])
                    entry_grad = entry_parameter_grads[name]
                    parameter.grad = (
                        None
                        if entry_grad is None
                        else entry_grad.to(entry_parameter_devices[name]).clone()
                    )
                    parameter.requires_grad = entry_requires_grad[name]
                for name, buffer in self.model.named_buffers():
                    buffer.data = buffer.data.to(entry_buffer_devices[name])
            self.model.train(entry_training)

        device = torch.device(self.device)

        def prepare(
            record: Mapping[str, Any],
        ) -> tuple[list[torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
            prompt_kwargs: dict[str, Any] = {}
            if record.get("points") is not None:
                prompt_kwargs["input_points"] = [[record["points"]]]
                prompt_kwargs["input_labels"] = [[record["point_labels"]]]
            if record.get("box") is not None:
                prompt_kwargs["input_boxes"] = [[record["box"]]]
            batch = self.processor(
                images=record["image"], return_tensors="pt", **prompt_kwargs
            ).to(device)
            with torch.inference_mode():
                embeddings = [
                    item.detach().cpu()
                    for item in self.model.get_image_embeddings(batch["pixel_values"])
                ]
            prompts = {
                key: batch[key].detach().cpu()
                for key in ("input_points", "input_labels", "input_boxes")
                if key in batch
            }
            target = torch.from_numpy(np.asarray(record["mask"], dtype=np.float32))
            original_size = torch.tensor([[record["image"].height, record["image"].width]])
            return embeddings, prompts, target, original_size

        def score(
            cached: Sequence[
                tuple[list[torch.Tensor], dict[str, torch.Tensor], torch.Tensor, torch.Tensor]
            ],
        ) -> float:
            values: list[float] = []
            self.model.eval()
            with torch.inference_mode():
                for embeddings, prompts, target, original_size in cached:
                    device_embeddings = [item.to(device) for item in embeddings]
                    device_prompts = {key: value.to(device) for key, value in prompts.items()}
                    device_target = target.to(device)
                    outputs = self.model(
                        image_embeddings=device_embeddings,
                        multimask_output=False,
                        **device_prompts,
                    )
                    public_mask = self.processor.post_process_masks(
                        outputs.pred_masks.cpu(),
                        original_size,
                        mask_threshold=MASK_THRESHOLD,
                    )[0][0, 0]
                    values.append(mask_iou(public_mask.numpy(), device_target.cpu().numpy().astype(bool)))
            return float(np.mean(values)) if values else 0.0

        try:
            torch.manual_seed(seed)
            self.model.to(device).eval()
            cached_train = [prepare(record) for record in train_records]
            cached_val = [prepare(record) for record in val_records] if val_records else []
            baseline_val_iou = score(cached_val) if cached_val else None
        except BaseException:
            restore_entry_state()
            raise

        try:
            if not self.adaptation_config:
                self.freeze_for_adaptation()
            if any(
                parameter.requires_grad and not name.startswith(TRAINABLE_PREFIXES)
                for name, parameter in self.model.named_parameters()
            ):
                raise RuntimeError("parameters outside the declared SAM2 adapter surface are trainable")
            trainable = {
                name: parameter
                for name, parameter in self.model.named_parameters()
                if parameter.requires_grad
            }
            if not trainable:
                raise RuntimeError("model has no trainable parameters")
            before = {
                name: parameter.detach().cpu().clone() for name, parameter in trainable.items()
            }
            optimizer = AdamW(list(trainable.values()), lr=float(learning_rate))
            previous_config = copy.deepcopy(self.adaptation_config)
            prior_runs = previous_config.get("training_runs")
            if prior_runs is None:
                prior_runs = (
                    [_adaptation_run_snapshot(previous_config)]
                    if _is_positive_finite_number(previous_config.get("weight_delta_l2"))
                    and isinstance(previous_config.get("history"), list)
                    and previous_config["history"]
                    else []
                )
            elif not isinstance(prior_runs, list) or not all(
                isinstance(run, Mapping) for run in prior_runs
            ):
                raise RuntimeError("adaptation training_runs metadata is malformed")
            else:
                prior_runs = [json.loads(json.dumps(run)) for run in prior_runs]
            for key in (
                "epochs",
                "learning_rate",
                "batch_size",
                "seed",
                "loss",
                "train_manifest",
                "validation_manifest",
                "baseline_validation_mask_iou",
                "history",
                "weight_delta_l2",
            ):
                self.adaptation_config.pop(key, None)
        except BaseException:
            restore_entry_state()
            raise

        history: list[dict[str, Any]] = []
        try:
            for epoch in range(1, epochs + 1):
                order = list(range(len(cached_train)))
                random.Random(seed + epoch * 17).shuffle(order)
                total_loss = 0.0
                for index in order:
                    embeddings, prompts, target, _original_size = cached_train[index]
                    device_embeddings = [item.to(device) for item in embeddings]
                    device_prompts = {key: value.to(device) for key, value in prompts.items()}
                    device_target = target.to(device)
                    optimizer.zero_grad(set_to_none=True)
                    logits = self.model(
                        image_embeddings=device_embeddings,
                        multimask_output=False,
                        **device_prompts,
                    ).pred_masks[0, 0, 0]
                    target_low = (
                        F.interpolate(
                            device_target[None, None], size=logits.shape, mode="area"
                        )[0, 0]
                        > 0
                    ).to(logits.dtype)
                    probabilities = logits.sigmoid()
                    bce = F.binary_cross_entropy_with_logits(logits, target_low)
                    dice = 1.0 - (2.0 * (probabilities * target_low).sum() + 1.0) / (
                        probabilities.sum() + target_low.sum() + 1.0
                    )
                    loss = bce + dice
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("fine-tuning produced a non-finite loss")
                    loss.backward()
                    optimizer.step()
                    if any(
                        not bool(torch.isfinite(parameter).all())
                        for parameter in trainable.values()
                    ):
                        raise RuntimeError("fine-tuning produced non-finite adapter weights")
                    total_loss += float(loss.item())
                epoch_data: dict[str, Any] = {
                    "epoch": epoch,
                    "train_loss": round(total_loss / len(cached_train), 6),
                    "optimizer_steps": len(cached_train),
                }
                if cached_val:
                    epoch_data["val_mask_iou"] = round(score(cached_val), 6)
                history.append(epoch_data)

            delta_sq = 0.0
            for name, parameter in trainable.items():
                delta_sq += float(
                    torch.sum((parameter.detach().cpu() - before[name]) ** 2).item()
                )
            weight_delta_l2 = delta_sq**0.5
            if not math.isfinite(weight_delta_l2):
                raise RuntimeError("fine-tuning produced a non-finite weight delta")
            if weight_delta_l2 == 0.0:
                raise RuntimeError("fine-tuning completed without changing adapter weights")
            self.model.eval()
            completed_run = {
                "epochs": epochs,
                "learning_rate": float(learning_rate),
                "batch_size": 1,
                "seed": seed,
                "loss": "binary-cross-entropy-plus-soft-dice",
                "train_manifest": train_manifest,
                "validation_manifest": val_manifest,
                "baseline_validation_mask_iou": baseline_val_iou,
                "history": history,
                "weight_delta_l2": weight_delta_l2,
            }
            stored_run = json.loads(json.dumps(completed_run))
            self.adaptation_config.update(stored_run)
            self.adaptation_config["training_runs"] = [
                *prior_runs,
                json.loads(json.dumps(stored_run)),
            ]
        except BaseException:
            with torch.no_grad():
                for name, parameter in trainable.items():
                    parameter.copy_(before[name].to(device=parameter.device, dtype=parameter.dtype))
            restore_entry_state()
            raise
        return json.loads(json.dumps(history))

    def evaluate_adaptation(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Evaluate adapted masks against ground truth and a prompt-box baseline."""
        validate_segmentation_dataset(records)
        model_ious: list[float] = []
        box_ious: list[float] = []
        box_model_ious: list[float] = []
        for record in records:
            result = self.segment(
                record["image"],
                points=record.get("points"),
                point_labels=record.get("point_labels"),
                box=record.get("box"),
                multimask=False,
            )
            target = np.asarray(record["mask"], dtype=np.bool_)
            model_iou = mask_iou(result["masks"][0], target)
            model_ious.append(model_iou)
            box = record.get("box")
            if box is not None:
                baseline = np.zeros_like(target)
                x0, y0 = (math.floor(float(value)) for value in box[:2])
                x1, y1 = (math.ceil(float(value)) for value in box[2:])
                baseline[y0:y1, x0:x1] = True
                box_ious.append(mask_iou(baseline, target))
                box_model_ious.append(model_iou)
        box_baseline_mean_iou = float(np.mean(box_ious)) if box_ious else None
        box_prompt_model_mean_iou = float(np.mean(box_model_ious)) if box_model_ious else None
        return {
            "records": len(records),
            "mean_mask_iou": float(np.mean(model_ious)),
            "per_record_mask_iou": model_ious,
            "box_baseline_records": len(box_ious),
            "box_prompt_model_mean_iou": box_prompt_model_mean_iou,
            "box_baseline_mean_iou": box_baseline_mean_iou,
            "delta_over_box_baseline": (
                None
                if box_baseline_mean_iou is None
                else float(box_prompt_model_mean_iou - box_baseline_mean_iou)
            ),
        }

    def save_artifact(self, output_dir: str | Path, *, producer_revision: str) -> Path:
        """Write a safe mask-hypernetwork adapter plus a closed integrity manifest."""
        from safetensors.torch import save_file

        if self.model is None:
            raise RuntimeError("artifact export requires an underlying model")
        try:
            _validate_completed_adaptation_metadata(self.adaptation_config)
        except ValueError as exc:
            raise RuntimeError("artifact export requires completed fine-tuning metadata") from exc
        if len(producer_revision) != 40 or any(ch not in "0123456789abcdef" for ch in producer_revision):
            raise ValueError("producer_revision must be a lowercase 40-hex Git commit")
        state = {}
        for name, tensor in self.model.state_dict().items():
            if not name.startswith(TRAINABLE_PREFIXES):
                continue
            if not bool(tensor.isfinite().all()):
                raise RuntimeError(f"cannot export non-finite adapter tensor: {name}")
            state[name] = tensor.detach().cpu().contiguous()
        if not state:
            raise RuntimeError("no adapter tensors selected for export")
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise FileExistsError(f"artifact directory is not empty: {root}")
        weights_path = root / ARTIFACT_WEIGHTS_NAME
        save_file(state, str(weights_path))
        manifest = {
            "artifactSpec": "1.0",
            "format": ARTIFACT_FORMAT,
            "formatVersion": ARTIFACT_FORMAT_VERSION,
            "artifactClass": "ADAPTER",
            "artifactKind": "sam2-mask-hypernetwork-adapter",
            "producer": {"pipelineId": "sam2-segmentation-pipeline", "revision": producer_revision},
            "createdAtUtc": datetime.now(UTC).isoformat(),
            "baseModel": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "adaptation": copy.deepcopy(self.adaptation_config),
            "trainablePrefixes": list(TRAINABLE_PREFIXES),
            "retainedData": {"containsTrainingRecords": False, "containsSupportRecords": False},
            "serialization": "safetensors",
        }
        (root / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return root

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify and load a SAM2 adapter without code-capable deserialization."""
        from safetensors.torch import load_file

        if self.model is None:
            raise RuntimeError("cannot load an artifact without an underlying torch model")
        root = Path(artifact_dir)
        manifest_path = root / ARTIFACT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
        expected_files = {ARTIFACT_MANIFEST_NAME, ARTIFACT_WEIGHTS_NAME}
        actual_entries = {path.name for path in root.iterdir()}
        if actual_entries != expected_files:
            raise ValueError(
                f"artifact directory must contain exactly {sorted(expected_files)}, "
                f"found {sorted(actual_entries)}"
            )
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"unrecognized artifact format: {manifest.get('format')}")
        if manifest.get("formatVersion") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(f"unsupported artifact formatVersion: {manifest.get('formatVersion')}")
        if manifest.get("baseModel") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
            raise ValueError("artifact base model identity is incompatible")
        if manifest.get("trainablePrefixes") != list(TRAINABLE_PREFIXES):
            raise ValueError("artifact trainable prefixes do not match this pipeline")
        producer = manifest.get("producer")
        if (
            not isinstance(producer, Mapping)
            or set(producer) != {"pipelineId", "revision"}
            or producer["pipelineId"] != "sam2-segmentation-pipeline"
            or not isinstance(producer["revision"], str)
            or len(producer["revision"]) != 40
            or any(character not in "0123456789abcdef" for character in producer["revision"])
        ):
            raise ValueError("artifact producer identity is invalid")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must inventory exactly one weights file")
        entry = files[0]
        if entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError("artifact manifest names an unexpected weights path")
        weights_path = root / ARTIFACT_WEIGHTS_NAME
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights not found: {weights_path}")
        if weights_path.stat().st_size != entry.get("bytes") or _sha256(weights_path) != entry.get("sha256"):
            raise ValueError("artifact weights failed size or SHA-256 verification")
        adaptation = manifest.get("adaptation")
        _validate_completed_adaptation_metadata(adaptation)
        state = load_file(str(weights_path), device=self.device)
        expected = {
            name for name in self.model.state_dict() if name.startswith(TRAINABLE_PREFIXES)
        }
        if set(state) != expected:
            raise ValueError("artifact tensor inventory does not match the declared adapter surface")
        destination = self.model.state_dict()
        for name, tensor in state.items():
            expected_tensor = destination[name]
            if tensor.shape != expected_tensor.shape or tensor.dtype != expected_tensor.dtype:
                raise ValueError(
                    f"artifact tensor {name} has shape/dtype {tuple(tensor.shape)}/{tensor.dtype}; "
                    f"expected {tuple(expected_tensor.shape)}/{expected_tensor.dtype}"
                )
            if not bool(tensor.isfinite().all()):
                raise ValueError(f"artifact tensor {name} contains non-finite values")
        self.model.load_state_dict(state, strict=False)
        self.model.to(self.device).eval()
        # A fresh base model starts fully trainable. Reapply the declared adapter surface so a
        # verified artifact can be continued with ``finetune`` without exposing base parameters.
        self.freeze_for_adaptation()
        loaded_adaptation = copy.deepcopy(adaptation)
        prior_artifacts = loaded_adaptation.get("artifact_lineage", [])
        loaded_adaptation["artifact_lineage"] = [
            *json.loads(json.dumps(prior_artifacts)),
            {
                "manifest_sha256": _sha256(manifest_path),
                "weights_sha256": entry["sha256"],
                "producer": copy.deepcopy(producer),
            },
        ]
        self.adaptation_config = loaded_adaptation
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> SAM2SegmentationPipeline:
        """Construct a fresh pinned base model and attach a verified adapter."""
        pipeline = cls.from_pretrained(
            device=device, weights_dir=weights_dir, allow_download=allow_download
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline

    def segment(
        self,
        image: Image.Image,
        *,
        points: Sequence[Sequence[float]] | None = None,
        point_labels: Sequence[int] | None = None,
        box: Sequence[float] | None = None,
        multimask: bool | None = None,
    ) -> dict[str, Any]:
        """Segment one object, defaulting adapted pipelines to their trained single-mask head."""
        resolved_multimask = not bool(self.adaptation_config) if multimask is None else multimask
        rgb, clean_points, clean_labels, clean_box = _check_inputs(
            image, points, point_labels, box, resolved_multimask
        )
        if self.adaptation_config and resolved_multimask:
            raise ValueError(
                "adapted pipelines require multimask=False because the adapter trains only "
                "the single-mask output hypernetwork"
            )
        masks, iou_scores = self._runner(
            rgb, clean_points, clean_labels, clean_box, resolved_multimask
        )
        masks = np.asarray(masks)
        expected = (NUM_MULTIMASK_OUTPUTS if resolved_multimask else 1, rgb.height, rgb.width)
        if masks.dtype != np.bool_ or masks.shape != expected or len(iou_scores) != expected[0]:
            raise RuntimeError(f"backend returned {masks.shape} {masks.dtype}, {len(iou_scores)} scores")
        return {
            "masks": masks,
            "iou_scores": [float(v) for v in iou_scores],
            "multimask": resolved_multimask,
            "points": clean_points,
            "point_labels": clean_labels,
            "box": clean_box,
            "width": rgb.width,
            "height": rgb.height,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
