"""Text-prompted (zero-shot) image segmentation with the pinned ``CIDAS/clipseg-rd64-refined`` checkpoint, plus
the adaptation contract for labelled (image, phrase, mask) records: corpus evaluation at a stated threshold,
bounded fine-tuning of the CLIPSeg decoder on cached CLIP activations, and a verified adapter artifact.

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the CLIPSeg architecture comes from the pinned ``transformers`` release,
the weights are SafeTensors, and no model-repository code is executed.
"""
# ruff: noqa: E501  -- adaptation-contract lines are kept at the fleet width

from __future__ import annotations

import hashlib
import json
import random
import time
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
WEIGHTS_FILE = "model.safetensors"

# Adaptation contract: the CLIPSeg decoder (three transformer layers over the CLIP activations extracted at
# `EXTRACT_LAYERS`, the FiLM conditioning on the phrase embedding, the reduce projections and the transposed
# convolution) is the adapter — the part the upstream authors trained; the CLIP vision and text towers stay
# frozen, so their outputs are computed once per record and cached (fp16 on the host).
PARAMETER_COUNT = 150_747_746
DECODER_PARAMETERS = 1_127_009
EXTRACT_LAYERS = (3, 6, 9)
_TRAINABLE_PREFIXES = ("decoder.",)
ARTIFACT_FORMAT = f"org.valcorza.{MODEL_KEY}.adapter.v1"
ARTIFACT_VERSION = 1
ADAPTER_WEIGHTS = "adapter.safetensors"
ADAPTER_MANIFEST = "manifest.json"
MIN_SCORED_RECORDS = 50  # below this a scored set is labelled a small sample
MAX_EVAL_RECORDS = 5_000
EVAL_BATCH_SIZE = 8
GRAD_CLIP = 1.0


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


def _weight_digest(root: Path) -> str | None:
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        return None
    with open(manifest_path, encoding="utf-8") as handle:
        entries = json.load(handle).get("files", [])
    return next((e["sha256"] for e in entries if e["path"] == WEIGHTS_FILE), None)


def _trainable_names(model: Any) -> list[str]:
    """The decoder's tensors; the CLIP vision and text towers and their projections stay frozen."""
    return [name for name, _ in model.named_parameters() if name.startswith(_TRAINABLE_PREFIXES)]


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
    if not names or any(not str(n).startswith(_TRAINABLE_PREFIXES) for n in names):
        raise ValueError("artifact tensors must all belong to the CLIPSeg decoder")
    adapter = manifest.get("adapter") or {}
    threshold = adapter.get("threshold")
    if isinstance(threshold, bool) or not isinstance(threshold, int | float) or not 0.0 <= float(threshold) <= 1.0:
        raise ValueError("artifact manifest must record adapter.threshold, the mask threshold the epoch was selected at")


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
    _batch_runner: Callable[[list[Image.Image], list[str]], list[np.ndarray]] | None = None
    _model: Any = None
    _processor: Any = None
    weight_sha256: str | None = None
    adapter: dict[str, Any] | None = None

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
        for param in model.parameters():
            param.requires_grad_(False)
        weight_sha256 = _weight_digest(root) if (root / MANIFEST_NAME).is_file() else None

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

        def batch_runner(images: list[Image.Image], queries: list[str]) -> list[np.ndarray]:
            # One phrase per image: each (image, phrase) pair is one decoder query.
            inputs = processor(text=queries, images=images, padding=True, return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                logits = model(**inputs).logits
            if logits.dim() == 2:
                logits = logits.unsqueeze(0)
            probs = torch.sigmoid(logits).unsqueeze(1)
            out = []
            for k, image in enumerate(images):
                resampled = torch.nn.functional.interpolate(probs[k : k + 1], size=(image.height, image.width), mode="bilinear", align_corners=False)
                out.append(resampled[0, 0].float().cpu().numpy())
            return out

        return cls(runner, resolved_device, batch_runner, model, processor, weight_sha256, None)

    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise RuntimeError("this pipeline has no loaded model (injected runner); use from_pretrained")
        return self._model, self._processor

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

    # ------------------------------------------------------------------------------------------------------
    # Adaptation contract: batched (image, phrase) segmentation, corpus evaluation, bounded fine-tuning, artifacts
    # ------------------------------------------------------------------------------------------------------

    def segment_batch(
        self,
        pairs: Sequence[tuple[Image.Image, str]],
        *,
        threshold: float = MASK_THRESHOLD,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> list[dict[str, Any]]:
        """Segment one phrase per image for many (image, phrase) pairs, `batch_size` pairs per forward; one
        ``{prompt, mask, probability, area_fraction, max_probability, bbox}`` per pair, in order. With an injected
        runner and no batch runner the pairs are segmented one by one through the runner."""
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 64:
            raise ValueError("batch_size must be an int in 1..64")
        checked = [_check_inputs(image, [prompt], threshold) for image, prompt in pairs]
        cut = _check_threshold("threshold", threshold)
        out: list[dict[str, Any]] = []
        for start in range(0, len(checked), batch_size):
            batch = checked[start : start + batch_size]
            if self._batch_runner is not None:
                probs = self._batch_runner([rgb for rgb, _, _ in batch], [q[0] for _, q, _ in batch])
            else:
                probs = [np.asarray(self._runner(rgb, q), dtype=np.float32)[0] for rgb, q, _ in batch]
            if len(probs) != len(batch):
                raise RuntimeError(f"backend returned {len(probs)} probability maps for {len(batch)} pairs")
            for (rgb, queries, _), prob in zip(batch, probs, strict=True):
                prob = np.asarray(prob, dtype=np.float32)
                if prob.shape != (rgb.height, rgb.width):
                    raise RuntimeError(f"backend returned a probability map of shape {prob.shape}, expected {(rgb.height, rgb.width)}")
                if prob.min() < 0.0 or prob.max() > 1.0:
                    raise RuntimeError("backend returned probabilities outside [0, 1]")
                mask = prob >= cut
                out.append({"prompt": queries[0], "mask": mask, "probability": prob, "area_fraction": float(mask.mean()), "max_probability": float(prob.max()), "bbox": mask_bbox(mask)})
            if progress is not None:
                progress(len(out), len(checked))
        return out

    def evaluate(
        self,
        records: Sequence[Mapping[str, Any]],
        *,
        threshold: float = MASK_THRESHOLD,
        batch_size: int = EVAL_BATCH_SIZE,
        progress: Callable[[int, int], None] | None = None,
    ) -> dict[str, Any]:
        """Segment every validated record's phrase on its image and score the thresholded masks against the record
        masks with ``metrics.segmentation_metrics`` (mean IoU, micro IoU, Dice, pixel precision and recall). Works
        with an injected runner too."""
        from .metrics import segmentation_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        items = self.segment_batch([(r["image"], r["prompt"]) for r in checked], threshold=threshold, batch_size=batch_size, progress=progress)
        metrics = segmentation_metrics([item["mask"] for item in items], checked)
        metrics.update(
            {
                "threshold": float(threshold),
                "predicted_area_fraction": sum(item["area_fraction"] for item in items) / len(items),
                "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
                "adapted": self.adapter is not None,
                "seconds": round(time.perf_counter() - started, 3),
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
            }
        )
        return metrics

    def _encode(self, records: Sequence[Mapping[str, Any]], batch_size: int, progress: Callable[[int, int], None] | None = None) -> tuple[list[Any], Any, Any]:
        """Run the frozen towers once per record: the CLIP vision activations at `EXTRACT_LAYERS` (stored fp16 on
        the host), the phrase embeddings, and the reference masks resampled to the decoder's logit grid."""
        model, processor = self._require_model()
        import torch

        activations: list[list[Any]] = [[] for _ in EXTRACT_LAYERS]
        conditionals, targets = [], []
        with torch.no_grad():
            for start in range(0, len(records), batch_size):
                batch = records[start : start + batch_size]
                inputs = processor(text=[r["prompt"] for r in batch], images=[r["image"] for r in batch], padding=True, return_tensors="pt").to(self.device)
                vision = model.clip.vision_model(pixel_values=inputs["pixel_values"], output_hidden_states=True)
                for slot, layer in enumerate(EXTRACT_LAYERS):
                    activations[slot].append(vision.hidden_states[layer + 1].to("cpu", torch.float16))
                conditionals.append(model.clip.get_text_features(inputs["input_ids"], attention_mask=inputs["attention_mask"]).to("cpu"))
                resized = [torch.nn.functional.interpolate(torch.from_numpy(np.asarray(r["mask"], dtype=np.float32))[None, None], size=(LOGIT_SIZE, LOGIT_SIZE), mode="nearest")[0, 0] for r in batch]
                targets.append(torch.stack(resized))
                if progress is not None:
                    progress(min(start + batch_size, len(records)), len(records))
        return [torch.cat(slot) for slot in activations], torch.cat(conditionals), torch.cat(targets)

    def _decoder_logits(self, activations: Sequence[Any], conditionals: Any) -> Any:
        """The decoder on cached tower outputs: exactly what `CLIPSegForImageSegmentation.forward` computes after
        its frozen CLIP steps (parity with the full forward is asserted by the model-backed tests)."""
        model, _ = self._require_model()
        import torch

        return model.decoder([a.to(self.device, torch.float32) for a in activations], conditionals.to(self.device, torch.float32)).logits

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None,
        *,
        epochs: int = 8,
        lr: float = 3e-4,
        batch_size: int = 8,
        seed: int = 0,
        threshold: float = MASK_THRESHOLD,
        progress: Callable[[Mapping[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded fine-tuning of the CLIPSeg decoder on labelled (image, phrase, mask) records with the per-pixel
        binary cross-entropy over the decoder's logit grid — the loss the upstream authors trained the decoder
        with. The frozen CLIP towers are run once per record under no gradient and their outputs cached (the
        vision activations at `EXTRACT_LAYERS` and the phrase embedding), so each step runs only the decoder; the
        logits equal the full model's exactly. AdamW (no weight decay), gradient clipping at `GRAD_CLIP`, seeded
        shuffling, no scheduler, no augmentation. Epoch 0 records the frozen model's validation rates at
        `threshold`; the epoch with the highest validation mean IoU (the earliest on ties) is kept. On any
        exception the frozen decoder is restored."""
        from .metrics import segmentation_metrics
        from .samples import validate_dataset

        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 100:
            raise ValueError("epochs must be an int in 1..100")
        if isinstance(batch_size, bool) or not isinstance(batch_size, int) or not 1 <= batch_size <= 128:
            raise ValueError("batch_size must be an int in 1..128")
        if not isinstance(lr, int | float) or isinstance(lr, bool) or not 0 < lr <= 1e-2:
            raise ValueError("lr must be a number in (0, 1e-2]")
        cut = _check_threshold("threshold", threshold)
        train_checked = validate_dataset(train)["records"]
        val_checked = validate_dataset(val, min_records=1)["records"] if val is not None else None
        model, _processor = self._require_model()
        import torch

        started = time.perf_counter()
        names = _trainable_names(model)
        params = {name: param for name, param in model.named_parameters() if name in set(names)}
        n_trainable = sum(p.numel() for p in params.values())
        backup = {name: param.detach().clone() for name, param in params.items()}
        previous_adapter = self.adapter
        cudnn_flags = torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark
        torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
        try:
            model.eval()
            activations, conditionals, targets = self._encode(train_checked, EVAL_BATCH_SIZE)
            cached_val = self._encode(val_checked, EVAL_BATCH_SIZE) if val_checked is not None else None
            cache_seconds = round(time.perf_counter() - started, 3)

            def score_val() -> dict[str, Any] | None:
                if val_checked is None or cached_val is None:
                    return None
                with torch.no_grad():
                    masks = []
                    for start in range(0, len(val_checked), EVAL_BATCH_SIZE):
                        logits = self._decoder_logits([a[start : start + EVAL_BATCH_SIZE].float() for a in cached_val[0]], cached_val[1][start : start + EVAL_BATCH_SIZE])
                        probs = torch.sigmoid(logits).unsqueeze(1)
                        for k, record in enumerate(val_checked[start : start + EVAL_BATCH_SIZE]):
                            prob = torch.nn.functional.interpolate(probs[k : k + 1], size=(record["image"].height, record["image"].width), mode="bilinear", align_corners=False)[0, 0]
                            masks.append((prob >= cut).cpu().numpy())
                m = segmentation_metrics(masks, val_checked)
                return {k: m[k] for k in ("miou", "iou_micro", "dice", "pixel_precision", "pixel_recall", "n")}

            for name, param in model.named_parameters():
                param.requires_grad_(name in params)
            history: list[dict[str, Any]] = [{"epoch": 0, "train_loss": None, "val": score_val(), "note": "frozen model"}]
            if progress is not None:
                progress(history[-1])
            best_epoch, best_score = 0, (history[0]["val"] or {}).get("miou", -1.0)
            best_state = {name: param.detach().clone() for name, param in params.items()}
            optimizer = torch.optim.AdamW(list(params.values()), lr=lr, weight_decay=0.0)
            rng = random.Random(seed)
            torch.manual_seed(seed)
            order = list(range(len(train_checked)))
            for epoch in range(1, epochs + 1):
                rng.shuffle(order)
                model.decoder.train()
                total, steps = 0.0, 0
                for start in range(0, len(order), batch_size):
                    idx = torch.tensor(order[start : start + batch_size])
                    optimizer.zero_grad(set_to_none=True)
                    logits = self._decoder_logits([a[idx].float() for a in activations], conditionals[idx])
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, targets[idx].to(self.device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(list(params.values()), GRAD_CLIP)
                    optimizer.step()
                    total += float(loss.detach())
                    steps += 1
                model.eval()
                entry = {"epoch": epoch, "train_loss": round(total / max(steps, 1), 5), "val": score_val()}
                history.append(entry)
                if progress is not None:
                    progress(entry)
                score = (entry["val"] or {}).get("miou")
                if val_checked is None or (score is not None and score > best_score):
                    best_epoch, best_score = epoch, score if score is not None else best_score
                    best_state = {name: param.detach().clone() for name, param in params.items()}
            with torch.no_grad():
                for name, param in params.items():
                    param.copy_(best_state[name])
        except BaseException:
            with torch.no_grad():
                for name, param in params.items():
                    param.copy_(backup[name])
            model.eval()
            self.adapter = previous_adapter
            raise
        finally:
            for param in model.parameters():
                param.requires_grad_(False)
            model.eval()
            torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = cudnn_flags
        self.adapter = {
            "threshold": cut,
            "trainable_names": names,
            "n_trainable": n_trainable,
            "n_total": sum(p.numel() for p in model.parameters()),
            "extract_layers": list(EXTRACT_LAYERS),
            "epochs": epochs,
            "batch_size": batch_size,
            "best_epoch": best_epoch,
            "selection": "highest validation mean IoU" if val_checked is not None else "final epoch (no validation split)",
            "loss": f"per-pixel binary cross-entropy over the {LOGIT_SIZE}x{LOGIT_SIZE} decoder logits against the reference mask resampled to that grid; computed on cached CLIP activations",
            "lr": float(lr),
            "seed": seed,
            "n_train": len(train_checked),
            "n_val": len(val_checked) if val_checked is not None else 0,
            "cache_seconds": cache_seconds,
            "history": history,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the trained tensors as safetensors plus a manifest naming the base, the digests, the threshold and
        the training configuration. Requires a prior `adapt`."""
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
            "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_file": WEIGHTS_FILE, "weight_sha256": self.weight_sha256},
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
        tensor set. Refuses tensors outside the decoder."""
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
    ) -> ClipSegSegmentationPipeline:
        """Check the adapter manifest against the base snapshot's recorded weight digest, load the verified base, then
        overlay the adapter (checked again, and the tensor set, before deserialising). A refused manifest never loads
        a model."""
        artifact = Path(artifact_dir)
        manifest_path = artifact / ADAPTER_MANIFEST
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest missing: {manifest_path}")
        with open(manifest_path, encoding="utf-8") as handle:
            manifest = json.load(handle)
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        _check_artifact_manifest(manifest, artifact, _weight_digest(root) or "")
        pipe = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipe.load_artifact(artifact_dir)
        return pipe
