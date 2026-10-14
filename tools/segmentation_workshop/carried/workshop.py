"""Standalone staged workshop worker; heavy dependencies live in an isolated process.

Reference modules and manifests are embedded by the notebook generator. No repository
or executable source is downloaded. All metric grids retain the decoded input size.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import urllib.error
import uuid
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROLES = ("train", "validation", "test", "activity")
MINIMUM = dict(zip(ROLES, (8, 2, 2, 1), strict=True))
DEFAULT_COUNTS = dict(zip(ROLES, (120, 40, 40, 12), strict=True))
PARITY_ATOL = 1e-5
# Exit code for a refused BYOD input. The worker prints the reason once and the notebook cell stops without
# raising; any other failure keeps its traceback and still raises.
REFUSED_EXIT = 65


class InputRefused(ValueError):
    """A BYOD manifest, image or mask breaks a documented input rule (not a defect of this worker)."""


def refuse(error: Exception) -> None:
    print(
        f"Refused: {error}. Fix that file or manifest entry in BYOD_PATH and run this cell again.",
        flush=True,
    )
    raise SystemExit(REFUSED_EXIT)


def read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def peak_host_rss() -> int | None:
    try:
        import resource
    except ImportError:
        return None  # CPU validation on Windows; canonical Colab records Linux KiB.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss


def overlap(prediction: np.ndarray, reference: np.ndarray) -> dict:
    p, r = np.asarray(prediction), np.asarray(reference)
    if p.dtype != bool or r.dtype != bool or p.ndim != 2 or p.shape != r.shape:
        raise ValueError("Scoring needs aligned two-dimensional boolean masks")
    tp, fp, fn = (int(x) for x in ((p & r).sum(), (p & ~r).sum(), (~p & r).sum()))
    union = tp + fp + fn
    return {
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "pixels": int(p.size),
        "iou": tp / union if union else None,
        "dice": 2 * tp / (2 * tp + fp + fn) if union else None,
        "precision": tp / (tp + fp) if tp + fp else None,
        "recall": tp / (tp + fn) if tp + fn else None,
        "predicted_area_fraction": float(p.mean()),
        "precision_note": None if tp + fp else "Undefined: no predicted foreground pixels",
        "overlap_note": None if union else "Undefined: reference and prediction both empty",
    }


def aggregate(rows: list[dict]) -> dict:
    def mean(key: str) -> float | None:
        values = [r[key] for r in rows if r[key] is not None]
        return float(np.mean(values)) if values else None

    tp, fp, fn = (sum(r[k] for r in rows) for k in ("tp", "fp", "fn"))
    return {
        "n": len(rows),
        "mean_target_iou": mean("iou"),
        "mean_dice": mean("dice"),
        "pooled_iou": tp / (tp + fp + fn) if tp + fp + fn else None,
        "pixel_precision": tp / (tp + fp) if tp + fp else None,
        "pixel_recall": tp / (tp + fn) if tp + fn else None,
        "overlap_defined_n": sum(r["iou"] is not None for r in rows),
        "empty_prediction_n": sum(r["tp"] + r["fp"] == 0 for r in rows),
        "empty_reference_n": sum(r["tp"] + r["fn"] == 0 for r in rows),
        "empty_reference_false_positive_n": sum(r["tp"] + r["fn"] == 0 and r["fp"] > 0 for r in rows),
        "failed_calls": 0,
        "policy": "A failed call aborts the stage; never silently excluded",
    }


def verify_parity(before: np.ndarray, after: np.ndarray) -> dict:
    a, b = np.asarray(before), np.asarray(after)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Parity arrays must be aligned and finite")
    difference = float(np.max(np.abs(a - b)))
    equal = bool(np.array_equal(a >= 0.5, b >= 0.5))
    if difference > PARITY_ATOL or not equal:
        raise ValueError("Fresh reload probability or binary mask parity failed")
    return {"max_abs_probability_difference": difference, "identical_masks": equal}


def prompts(mask: np.ndarray) -> dict:
    """Four-connected largest component; ties and distance maxima use row-major order."""
    from scipy.ndimage import distance_transform_edt, label

    a = np.asarray(mask, dtype=bool)
    if a.ndim != 2 or not a.any():
        raise ValueError("A geometric prompt requires a positive two-dimensional mask")
    components, n = label(a)
    sizes = np.bincount(components.ravel())[1:]
    component = components == (int(np.argmax(sizes)) + 1)
    # Padding makes the image edge an actual boundary, including full-image masks.
    distances = distance_transform_edt(np.pad(component, 1))[1:-1, 1:-1]
    y, x = np.unravel_index(int(np.argmax(distances)), a.shape)
    ys, xs = np.nonzero(a)
    box = [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]
    return {
        "point": [int(x), int(y)],
        "box": box,
        "components": int(n),
        "coordinate_contract": "xyxy pixel edges, upper bounds exclusive; wrapper receives unchanged",
        "assistance": "Reference-derived simulated informed click / oracle localization",
    }


def expand_box(box: list, size: tuple[int, int], fraction: float) -> list[float]:
    x0, y0, x1, y1 = box
    w, h = size
    dx, dy = fraction * (x1 - x0), fraction * (y1 - y0)
    return [max(0.0, x0 - dx), max(0.0, y0 - dy), min(float(w), x1 + dx), min(float(h), y1 + dy)]


SIDES = ("left", "top", "right", "bottom")


def box_change(base: list, actual: list, size: tuple[int, int], fraction: float) -> dict:
    """Requested versus realised expansion of one box; a side is clipped when the image edge stopped it."""
    x0, y0, x1, y1 = base
    w, h = size
    dx, dy = fraction * (x1 - x0), fraction * (y1 - y0)
    requested = [dx, dy, dx, dy]
    realised = [x0 - actual[0], y0 - actual[1], actual[2] - x1, actual[3] - y1]
    return {
        "base_box": base,
        "requested_fraction": fraction,
        "requested_expansion_pixels": requested,
        # actual_expansion_pixels is the field name used before revision 0.2.0; both hold the same values.
        "actual_expansion_pixels": realised,
        "realised_expansion_pixels": realised,
        "clipped_sides": [s for s, want, got in zip(SIDES, requested, realised, strict=True) if got < want - 1e-9],
        "whole_image_box": [float(v) for v in actual] == [0.0, 0.0, float(w), float(h)],
        "box_area_fraction": (actual[2] - actual[0]) * (actual[3] - actual[1]) / (w * h),
    }


def change_counts(paired: list[dict]) -> dict:
    return {
        "n": len(paired),
        "improved": sum(r["delta_iou"] > 0 for r in paired),
        "unchanged": sum(r["delta_iou"] == 0 for r in paired),
        "worsened": sum(r["delta_iou"] < 0 for r in paired),
    }


def validate_roles(records: list[dict], *, default: bool = False) -> dict:
    from clipseg_reference import samples

    ids, identities, groups = set(), set(), {}
    splits = {role: [] for role in ROLES}
    for index, entry in enumerate(records):
        if not isinstance(entry, dict):
            raise InputRefused(f"records[{index}] must be an object with id, image, mask, prompt and role")
        where = f"records[{index}] (id {entry.get('id')!r})"
        role = entry.get("role")
        if not isinstance(role, str) or role not in splits:
            raise InputRefused(f"{where}: unknown role {role!r}; expected one of {ROLES}")
        try:
            checked = samples.validate_dataset([entry], min_records=1)["records"][0]
        except ValueError as exc:
            # The record is validated on its own, so the library numbers it records[0]; restore its position.
            raise InputRefused(str(exc).replace("records[0]", where, 1)) from None
        except OSError as exc:
            raise InputRefused(f"{where}: the image or mask could not be decoded ({exc})") from None
        if checked["image"].getexif().get(274, 1) != 1:
            raise InputRefused(f"{where}: normalize EXIF orientation together with the mask before using this image")
        identity = samples.image_digest(checked["image"])
        if checked["id"] in ids:
            raise InputRefused(f"{where}: duplicate id; every record needs a unique id")
        if identity in identities:
            raise InputRefused(f"{where}: the decoded image duplicates an earlier record; roles must contain distinct images")
        ids.add(checked["id"])
        identities.add(identity)
        group = entry.get("group_id", entry.get("group", entry.get("source_id", checked["id"])))
        if isinstance(group, bool) or not isinstance(group, (str, int)):
            raise InputRefused(f"{where}: group_id must be a string or an integer, not {type(group).__name__}")
        if group in groups and groups[group] != role:
            raise InputRefused(f"{where}: related group {group!r} is in both {groups[group]!r} and {role!r}")
        groups[group] = role
        checked.update(role=role, group=group)
        for key, actual in (
            ("image_sha256", identity),
            ("mask_sha256", samples.mask_digest(checked["mask"])),
        ):
            if key in entry and entry[key] != actual:
                raise InputRefused(f"{checked['id']}: {key} integrity mismatch")
            checked[key] = actual
        splits[role].append(checked)
    for role, values in splits.items():
        expected = DEFAULT_COUNTS[role] if default else MINIMUM[role]
        if len(values) < expected or (default and len(values) != expected):
            raise InputRefused(f"{role}: expected {'exactly' if default else 'at least'} {expected} records")
    try:
        samples.check_split_disjoint(splits)
    except ValueError as exc:
        raise InputRefused(str(exc)) from None
    return splits


def load_records(root: Path) -> dict:
    entries = read_json(root / "data/records.json")["records"]
    for item in entries:
        item["image"] = root / item["image"]
        item["mask"] = root / item["mask"]
    return validate_roles(entries)


def persist_records(root: Path, records: list[dict], *, default: bool) -> None:
    splits = validate_roles(records, default=default)
    exported, prompt_manifest = [], []
    for role in ROLES:
        for record in splits[role]:
            rid = record["id"]
            image_path, mask_path = f"data/images/{rid}.png", f"data/masks/{rid}.png"
            (root / image_path).parent.mkdir(parents=True, exist_ok=True)
            (root / mask_path).parent.mkdir(parents=True, exist_ok=True)
            record["image"].save(root / image_path)
            Image.fromarray(record["mask"].astype(np.uint8) * 255).save(root / mask_path)
            exported.append(
                {
                    **{k: v for k, v in record.items() if k not in ("image", "mask")},
                    "image": image_path,
                    "mask": mask_path,
                }
            )
            prompt_manifest.append(
                {"id": rid, "role": role, "text": record["prompt"], **prompts(record["mask"])}
            )
    write_json(root / "data/records.json", {"records": exported, "private_byod": not default})
    write_json(root / "outputs/sample_manifest.json", {"records": exported, "private_byod": not default})
    write_json(root / "outputs/prompt_manifest.json", {"records": prompt_manifest})
    baseline_rows = []
    for role in ("validation", "test", "activity"):
        for record in splits[role]:
            box = prompts(record["mask"])["box"]
            filled = np.zeros_like(record["mask"])
            x0, y0, x1, y1 = box
            filled[y0:y1, x0:x1] = True
            for name, mask in (("empty", np.zeros_like(filled)), ("oracle_filled_box", filled)):
                baseline_rows.append(
                    {
                        "id": record["id"],
                        "role": role,
                        "model": name,
                        "protocol": "unassisted" if name == "empty" else "oracle_box",
                        **overlap(mask, record["mask"]),
                    }
                )
    write_json(root / "outputs/baselines.json", baseline_rows)
    write_json(
        root / "outputs/prepare.json",
        {
            "counts": {r: len(v) for r, v in splits.items()},
            "data_manifest_sha256": digest(root / "data/records.json"),
            "invalid_records": 0,
            "duplicates": 0,
            "policy": "Invalid or duplicate input aborts; no silent exclusions",
        },
    )
    preview(root, splits["activity"][0], "reference_first")
    disconnected = next((r for r in splits["activity"] if prompts(r["mask"])["components"] > 1), None)
    if disconnected is not None:
        preview(root, disconnected, "reference_disconnected")
    write_json(
        root / "outputs/frozen_experiment.json",
        {
            "schema": "segmentation-workshop-v1",
            "data_sha256": digest(root / "data/records.json"),
            "prompts_sha256": digest(root / "outputs/prompt_manifest.json"),
            "seed": 42,
            "epochs": 4,
            "batch_size": 4,
            "learning_rate": 3e-4,
            "clipseg_threshold": 0.5,
            "selection": "highest validation mean target IoU, earliest tie including epoch zero",
            "parity_atol": PARITY_ATOL,
            "parity_rtol": 0,
            "identical_masks_required": True,
            "sam_candidate_selection": "argmax predicted quality; earliest tie",
            "box_expansions": [0, 0.1, 0.25],
        },
    )


def preview(root: Path, record: dict, name: str, predictions: list[str] | None = None) -> None:
    panels = [("Original", None), ("Reference: " + record["prompt"], record["mask"])]
    for model in predictions or []:
        path = root / "outputs/predictions" / model / f"{record['id']}.png"
        panels.append((model, np.asarray(Image.open(path)) > 0))
    canvas = Image.new("RGB", (280 * len(panels), 310), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (label, mask) in enumerate(panels):
        rgb = np.asarray(record["image"]).copy()
        if mask is not None:
            rgb[mask] = (rgb[mask] * 0.5 + np.array([0, 220, 120]) * 0.5).astype(np.uint8)
        image = Image.fromarray(rgb)
        image.thumbnail((280, 280))
        canvas.paste(image, (280 * index, 30))
        draw.text((280 * index + 3, 4), label, fill="black")
    canvas.save(root / "outputs" / f"{name}.png")


def prepare(root: Path, byod: Path | None = None) -> None:
    from clipseg_reference import samples

    if byod is not None:
        manifest_path = byod / "manifest.json"
        if not manifest_path.is_file():
            raise InputRefused(f"BYOD_PATH must be a directory containing manifest.json; not found: {manifest_path}")
        if manifest_path.stat().st_size > 1024**2:
            raise InputRefused("BYOD manifest exceeds 1 MiB")
        try:
            data = read_json(manifest_path)
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise InputRefused(f"manifest.json is not valid UTF-8 JSON ({exc})") from None
        records = data.get("records") if isinstance(data, dict) else None
        if not isinstance(records, list) or not 13 <= len(records) <= 256:
            raise InputRefused('manifest.json must be {"records": [...]} with 13–256 records across the four roles')
        total_pixels = 0
        for index, r in enumerate(records):
            if not isinstance(r, dict):
                raise InputRefused(f"records[{index}] must be an object with id, image, mask, prompt and role")
            where = f"records[{index}] (id {r.get('id')!r})"
            missing = [k for k in ("id", "image", "mask", "prompt", "role") if k not in r]
            if missing:
                raise InputRefused(f"{where}: missing required field(s) {missing}")
            for key in ("image", "mask"):
                text = r[key]
                if not isinstance(text, str):
                    raise InputRefused(f"{where}: {key} must be a relative path string")
                relative = Path(text)
                if relative.is_absolute() or ".." in relative.parts or "\\" in text or ":" in text:
                    raise InputRefused(f"{where}: {key} {text!r} must be a simple relative path")
                candidate = (byod / text).resolve()
                if not candidate.is_relative_to(byod.resolve()):
                    raise InputRefused(f"{where}: {key} {text!r} must remain inside the selected data directory")
                if not candidate.is_file():
                    raise InputRefused(f"{where}: {key} file not found: {text}")
                r[key] = candidate
                if candidate.stat().st_size > 20 * 1024**2:
                    raise InputRefused(f"{where}: {key} {text} exceeds 20 MiB")
                try:
                    header = Image.open(candidate)
                except OSError as exc:
                    raise InputRefused(f"{where}: {key} {text} is not a readable image") from exc
                with header:
                    if not all(16 <= side <= 4096 for side in header.size):
                        raise InputRefused(f"{where}: {key} {text} is {header.size[0]}x{header.size[1]}; sides must be 16–4096 pixels")
                    try:
                        header.load()
                    except (OSError, SyntaxError, ValueError) as exc:
                        raise InputRefused(f"{where}: {key} {text} could not be decoded ({exc})") from None
                    if header.getexif().get(274, 1) != 1:
                        raise InputRefused(f"{where}: normalize the EXIF orientation of {key} {text} before use")
                    if key == "image":
                        total_pixels += header.width * header.height
                        if total_pixels > 64 * 1024**2:
                            raise InputRefused(f"{where}: BYOD exceeds 64 million decoded image pixels in total")
            with Image.open(r["mask"]) as mask:
                array = np.asarray(mask)
                if array.ndim != 2 or not set(np.unique(array)).issubset({0, 1, 255}):
                    raise InputRefused(f"{where}: mask must be a single-channel binary 0/1 or 0/255 image")
        persist_records(root, records, default=False)
        return
    manifest = read_json(root / "sample.json")
    for attempt in range(3):
        print(f"Verifying and loading pinned FoodSeg103 row groups (attempt {attempt + 1}/3)", flush=True)
        try:
            corpus = samples.read_corpus(samples.fetch_corpus(cache_dir=root / "cache"))
            break
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if attempt == 2:
                raise
            time.sleep(attempt + 1)
    by_id = {r["id"]: r for r in corpus}
    chosen = []
    for pin in manifest["records"]:
        original = by_id[pin["id"]]
        for key in ("prompt", "class_id", "source_id", "source_row_group"):
            if original[key] != pin[key]:
                raise ValueError(f"Selection identity mismatch: {pin['id']} {key}")
        chosen.append({**original, **pin})
    persist_records(root, chosen, default=True)


def device() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Select a Colab GPU runtime, then start the notebook again. CPU qualification is unsupported."
        )
    torch.cuda.reset_peak_memory_stats()
    return "cuda"


def timed(call):
    import torch

    torch.cuda.synchronize()
    start = time.perf_counter()
    value = call()
    torch.cuda.synchronize()
    return value, time.perf_counter() - start


def save_prediction(root: Path, name: str, record: dict, mask: np.ndarray, probability=None) -> None:
    target = root / "outputs/predictions" / name
    target.mkdir(parents=True, exist_ok=True)
    Image.fromarray(mask.astype(np.uint8) * 255).save(target / f"{record['id']}.png")
    if probability is not None:
        if not np.isfinite(probability).all():
            raise ValueError("Non-finite CLIPSeg probabilities")
        np.save(target / f"{record['id']}.npy", probability, allow_pickle=False)


def clipseg_rows(root: Path, pipe, records: list[dict], name: str) -> list[dict]:
    rows = []
    for r in records:
        result, elapsed = timed(lambda r=r: pipe.segment(r["image"], [r["prompt"]], threshold=0.5))
        segment = result["segments"][0]
        save_prediction(root, name, r, segment["mask"], segment["probability"])
        rows.append(
            {
                "id": r["id"],
                "role": r["role"],
                "model": name,
                "protocol": "text",
                "seconds": elapsed,
                **overlap(segment["mask"], r["mask"]),
            }
        )
    return rows


def clipseg(root: Path) -> None:
    import torch
    from clipseg_reference.pipeline import ClipSegSegmentationPipeline

    splits = load_records(root)
    pipe = ClipSegSegmentationPipeline.from_pretrained(
        device=device(), weights_dir=root / "weights/clipseg", allow_download=True
    )
    pipe.segment(splits["validation"][0]["image"], [splits["validation"][0]["prompt"]])  # untimed warm-up
    evaluate = splits["validation"] + splits["test"] + splits["activity"]
    original = {
        n: p.detach().cpu().clone() for n, p in pipe._model.named_parameters() if n.startswith("decoder.")
    }
    updates = []

    def progress(entry):
        maximum = max(
            float((p.detach().cpu() - original[n]).abs().max())
            for n, p in pipe._model.named_parameters()
            if n in original
        )
        updates.append({"epoch": entry["epoch"], "max_abs_decoder_change": maximum})
        print(json.dumps(entry), flush=True)

    step_count = 0
    original_optimizer = torch.optim.AdamW

    def observe_step(*_args, **_kwargs):
        nonlocal step_count
        step_count += 1

    def observed_optimizer(*args, **kwargs):
        optimizer = original_optimizer(*args, **kwargs)
        permitted = {id(p) for n, p in pipe._model.named_parameters() if n in original}
        if any(id(p) not in permitted for group in optimizer.param_groups for p in group["params"]):
            raise RuntimeError("Adaptation optimizer includes a frozen tower parameter")
        optimizer.register_step_post_hook(observe_step)
        return optimizer

    try:
        torch.optim.AdamW = observed_optimizer
        adaptation = pipe.adapt(
            splits["train"],
            splits["validation"],
            epochs=4,
            batch_size=4,
            lr=3e-4,
            seed=42,
            threshold=0.5,
            progress=progress,
        )
    finally:
        torch.optim.AdamW = original_optimizer
    if len(updates) != 5 or not any(x["max_abs_decoder_change"] > 0 for x in updates[1:]):
        raise RuntimeError("Four adaptation epochs and actual decoder updates were not observed")
    adaptation["observed_epoch_updates"] = updates
    adaptation["optimizer_steps"] = step_count
    if step_count != 4 * math.ceil(len(splits["train"]) / 4):
        raise RuntimeError("Observed optimizer step count differs from the bounded recipe")
    adaptation["selected_frozen_epoch_zero"] = adaptation["best_epoch"] == 0
    pipe.save_artifact(
        root / "outputs/adapter",
        metadata={"data_manifest_sha256": digest(root / "data/records.json"), "observed_updates": updates},
    )
    # Selection is now persisted. Evaluate the same final test images with the exact
    # frozen decoder, then restore the selected decoder; no second resident model.
    write_json(
        root / "outputs/selection.json",
        {
            "best_epoch": adaptation["best_epoch"],
            "threshold": 0.5,
            "adapter_manifest_sha256": digest(root / "outputs/adapter/manifest.json"),
            "test_used_for_selection": False,
        },
    )
    selected_state = {n: p.detach().cpu().clone() for n, p in pipe._model.named_parameters() if n in original}
    with torch.no_grad():
        for n, p in pipe._model.named_parameters():
            if n in original:
                p.copy_(original[n].to(p.device))
    try:
        baseline = clipseg_rows(root, pipe, evaluate, "clipseg_frozen")
    finally:
        with torch.no_grad():
            for n, p in pipe._model.named_parameters():
                if n in selected_state:
                    p.copy_(selected_state[n].to(p.device))
    write_json(root / "outputs/clipseg_frozen.json", baseline)
    selected = clipseg_rows(root, pipe, evaluate, "clipseg_selected_live")
    write_json(root / "outputs/clipseg_selected_live.json", selected)
    write_json(root / "outputs/training_history.json", adaptation)
    preview(root, splits["test"][0], "clipseg_first", ["clipseg_frozen", "clipseg_selected_live"])
    write_json(
        root / "outputs/clipseg.json",
        {
            "pid": os.getpid(),
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "precision": "float32 inference and decoder; fp16 cached encoder activations",
            "warmup": "One validation image, excluded from timed calls",
            "selected_epoch": adaptation["best_epoch"],
        },
    )


def reload_stage(root: Path) -> None:
    from clipseg_reference.pipeline import ClipSegSegmentationPipeline

    if read_json(root / "outputs/clipseg.json")["pid"] == os.getpid():
        raise RuntimeError("Reload qualification must execute in a fresh process")
    splits = load_records(root)
    pipe = ClipSegSegmentationPipeline.from_artifact(
        root / "outputs/adapter", device=device(), weights_dir=root / "weights/clipseg"
    )
    pipe.segment(splits["validation"][0]["image"], [splits["validation"][0]["prompt"]])
    records = splits["validation"] + splits["test"] + splits["activity"]
    rows = clipseg_rows(root, pipe, records, "clipseg_reloaded")
    parity = []
    for r in records:
        before = root / "outputs/predictions/clipseg_selected_live" / r["id"]
        after = root / "outputs/predictions/clipseg_reloaded" / r["id"]
        a, b = (
            np.load(Path(str(before) + ".npy"), allow_pickle=False),
            np.load(Path(str(after) + ".npy"), allow_pickle=False),
        )
        try:
            checked = verify_parity(a, b)
        except ValueError as exc:
            write_json(
                root / "outputs/reload_parity.json", {"passed": False, "atol": PARITY_ATOL, "records": parity}
            )
            raise RuntimeError(
                f"Fresh reload parity failed for {r['id']}; diagnose rather than widening tolerance"
            ) from exc
        parity.append({"id": r["id"], **checked})
    live = read_json(root / "outputs/clipseg_selected_live.json")
    for a, b in zip(live, rows, strict=True):
        if (a["id"], a["iou"], a["dice"]) != (b["id"], b["iou"], b["dice"]):
            raise RuntimeError("Fresh reload metric parity failed")
    write_json(root / "outputs/clipseg_reloaded.json", rows)
    write_json(
        root / "outputs/reload_parity.json",
        {
            "passed": True,
            "atol": PARITY_ATOL,
            "rtol": 0,
            "pid": os.getpid(),
            "metric_parity": True,
            "records": parity,
        },
    )


def sam_stage(root: Path, name: str) -> None:
    import torch

    if name == "sam":
        from sam_reference import SAMViTSegmentationPipeline as Pipeline
    else:
        from sam2_reference import SAM2SegmentationPipeline as Pipeline
    splits = load_records(root)
    prompt_by_id = {r["id"]: r for r in read_json(root / "outputs/prompt_manifest.json")["records"]}
    pipe = Pipeline.from_pretrained(device=device(), weights_dir=root / "weights" / name, allow_download=True)
    first = splits["validation"][0]
    pipe.segment(first["image"], box=prompt_by_id[first["id"]]["box"], multimask=True)
    rows, experiments = [], []
    for r in splits["validation"] + splits["test"] + splits["activity"]:
        prompt = prompt_by_id[r["id"]]
        variants = [
            ("point", {"points": [prompt["point"]], "point_labels": [1]}),
            ("box", {"box": prompt["box"]}),
        ]
        if name == "sam2" and r["role"] == "activity":
            variants += [
                (f"box_expand_{fraction}", {"box": expand_box(prompt["box"], r["image"].size, fraction)})
                for fraction in (0.0, 0.1, 0.25)
            ]
        for protocol, kwargs in variants:
            result, elapsed = timed(
                lambda r=r, kwargs=kwargs: pipe.segment(r["image"], **kwargs, multimask=True)
            )
            qualities = np.asarray(result["iou_scores"])
            if not np.isfinite(qualities).all():
                raise ValueError("Non-finite predicted mask quality")
            selected = int(np.argmax(qualities))
            mask = result["masks"][selected]
            save_prediction(root, f"{name}_{protocol}", r, mask)
            row = {
                "id": r["id"],
                "role": r["role"],
                "model": name,
                "protocol": protocol,
                "candidate_index": selected,
                "predicted_quality": float(qualities[selected]),
                "seconds": elapsed,
                "prompt": kwargs,
                **overlap(mask, r["mask"]),
            }
            rows.append(row)
            if protocol.startswith("box_expand"):
                fraction = float(protocol.rsplit("_", 1)[1])
                experiments.append(
                    {
                        **row,
                        "phrase": r["prompt"],
                        **box_change(prompt["box"], kwargs["box"], r["image"].size, fraction),
                    }
                )
    write_json(root / f"outputs/{name}_metrics.json", rows)
    write_json(
        root / f"outputs/{name}.json",
        {
            "peak_gpu_bytes": torch.cuda.max_memory_allocated(),
            "gpu": torch.cuda.get_device_name(),
            "warmup": "One validation box, excluded from timing",
            "precision": "float32",
            "prompt_manifest_sha256": digest(root / "outputs/prompt_manifest.json"),
        },
    )
    if name == "sam2":
        write_json(root / "outputs/controlled_experiment.json", experiments)
        deltas = []
        baseline = {r["id"]: r for r in experiments if r["protocol"] == "box_expand_0.0"}
        for fraction in (0.1, 0.25):
            paired = [
                {
                    "id": r["id"],
                    "phrase": r["phrase"],
                    "iou_tight_box": baseline[r["id"]]["iou"],
                    "iou_expanded_box": r["iou"],
                    "delta_iou": r["iou"] - baseline[r["id"]]["iou"],
                    "delta_dice": r["dice"] - baseline[r["id"]]["dice"],
                    "clipped_sides": r["clipped_sides"],
                    "whole_image_box": r["whole_image_box"],
                    "box_area_fraction": r["box_area_fraction"],
                    "realised_expansion_pixels": r["realised_expansion_pixels"],
                }
                for r in experiments
                if r["protocol"] == f"box_expand_{fraction}"
            ]
            counts = change_counts(paired)
            deltas.append(
                {
                    "expansion": fraction,
                    "records": paired,
                    "improved": counts["improved"],
                    "unchanged": counts["unchanged"],
                    "worsened": counts["worsened"],
                    # A box stopped by an image edge grew on fewer sides than requested; keep it visible.
                    "clipped_records": sum(bool(r["clipped_sides"]) for r in paired),
                    "whole_image_records": sum(r["whole_image_box"] for r in paired),
                    "by_clipping": {
                        "clipped": change_counts([r for r in paired if r["clipped_sides"]]),
                        "not_clipped": change_counts([r for r in paired if not r["clipped_sides"]]),
                    },
                }
            )
        write_json(root / "outputs/box_experiment.json", deltas)
        preview(
            root,
            splits["activity"][0],
            "activity_expansion",
            ["sam2_box_expand_0.0", "sam2_box_expand_0.1", "sam2_box_expand_0.25"],
        )
    preview(root, splits["test"][0], f"{name}_first", [f"{name}_point", f"{name}_box"])


def report(root: Path) -> None:
    rows = []
    for filename in (
        "baselines.json",
        "clipseg_frozen.json",
        "clipseg_reloaded.json",
        "sam_metrics.json",
        "sam2_metrics.json",
    ):
        rows.extend(read_json(root / "outputs" / filename))
    groups = sorted({(r["model"], r["protocol"], r["role"]) for r in rows})
    summaries = [
        {
            "model": model,
            "protocol": protocol,
            "role": role,
            **aggregate(
                [r for r in rows if (r["model"], r["protocol"], r["role"]) == (model, protocol, role)]
            ),
        }
        for model, protocol, role in groups
    ]
    write_json(root / "outputs/metrics.json", {"rows": rows, "summary": summaries})
    frozen = {r["id"]: r for r in rows if r["model"] == "clipseg_frozen" and r["role"] == "test"}
    changes = [
        {"id": r["id"], "iou_change": r["iou"] - frozen[r["id"]]["iou"]}
        for r in rows
        if r["model"] == "clipseg_reloaded" and r["role"] == "test"
    ]
    paired_counts = {
        "improved": sum(r["iou_change"] > 0 for r in changes),
        "unchanged": sum(r["iou_change"] == 0 for r in changes),
        "worsened": sum(r["iou_change"] < 0 for r in changes),
    }
    write_json(root / "outputs/paired_changes.json", {"records": changes, **paired_counts})
    history = read_json(root / "outputs/training_history.json")
    reload = read_json(root / "outputs/reload_parity.json")
    markdown = [
        "# Segmentation results",
        "",
        "Small-sample evidence; compare within equal prompt protocols.",
        "",
    ]
    for title, models in (
        ("CLIPSeg: identical text prompts", {"clipseg_frozen", "clipseg_reloaded", "empty"}),
        ("SAM models: reference-derived geometric prompts", {"sam", "sam2", "oracle_filled_box"}),
    ):
        markdown += [
            "## " + title,
            "",
            "| Model | Protocol | n | Mean target IoU | Dice |",
            "|---|---|---:|---:|---:|",
        ]
        for row in summaries:
            if row["role"] == "test" and row["model"] in models:
                markdown.append(
                    f"| {row['model']} | {row['protocol']} | {row['n']} | "
                    f"{row['mean_target_iou']:.4f} | {row['mean_dice']:.4f} |"
                )
        markdown += [""]
    worst = max((r["max_abs_probability_difference"] for r in reload["records"]), default=None)
    markdown += [
        "Epoch selected on validation: "
        + str(history["best_epoch"])
        + (" (the frozen decoder: no adapted epoch beat it)" if history["best_epoch"] == 0 else ""),
        "",
        f"Fresh reload: {'passed' if reload['passed'] else 'FAILED'} on {len(reload['records'])} records in a new "
        f"process (maximum probability difference {worst}, tolerance {reload['atol']}; identical masks: "
        f"{all(r['identical_masks'] for r in reload['records'])}; metric parity: {reload.get('metric_parity')}).",
        "",
        f"CLIPSeg test images, frozen to adapted (reloaded) IoU: {paired_counts['improved']} improved, "
        f"{paired_counts['unchanged']} unchanged, {paired_counts['worsened']} worsened (of {len(changes)}).",
        "",
    ]
    box_path = root / "outputs/box_experiment.json"
    if box_path.is_file():
        markdown += [
            "## SAM 2 box-precision activity (activity images)",
            "",
            "| Expansion per side | Images | Improved | Unchanged | Worsened | Clipped by an image edge | Whole-image box |",
            "|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for entry in read_json(box_path):
            markdown.append(
                f"| {entry['expansion']:.0%} | {len(entry['records'])} | {entry['improved']} | {entry['unchanged']} | "
                f"{entry['worsened']} | {entry.get('clipped_records', 'n/a')} | {entry.get('whole_image_records', 'n/a')} |"
            )
        markdown += [""]
    markdown += [
        "A reference-derived box is localization assistance, not autonomous recognition.",
        "Personal learning notes are optional; no submission is required.",
    ]
    (root / "outputs/summary.md").write_text("\n".join(markdown) + "\n", encoding="utf-8")
    keys = sorted(set().union(*(r.keys() for r in rows)))
    with (root / "outputs/per_image_metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(
            {k: json.dumps(v) if isinstance(v, (dict, list)) else v for k, v in r.items()} for r in rows
        )
    with (root / "outputs/per_image_metrics.csv").open(encoding="utf-8", newline="") as f:
        restored = list(csv.DictReader(f))
    if len(restored) != len(rows) or any(
        saved["id"] != row["id"]
        or saved["model"] != row["model"]
        or saved["role"] != row["role"]
        or float(saved["iou"]) != row["iou"]
        for saved, row in zip(restored, rows, strict=True)
    ):
        raise RuntimeError("Exported per-image metric table failed reload verification")
    data = load_records(root)
    private = read_json(root / "data/records.json")["private_byod"]
    # Fixed first case plus deterministic best/worst changes and a disconnected activity case.
    chosen = {
        data["test"][0]["id"],
        min(changes, key=lambda r: (r["iou_change"], r["id"]))["id"],
        max(changes, key=lambda r: (r["iou_change"], r["id"]))["id"],
    }
    disconnected = next(
        (r for r in data["activity"] if prompts(r["mask"])["components"] > 1), data["activity"][0]
    )
    chosen.add(disconnected["id"])
    figure_files = []
    for record in data["test"] + data["activity"]:
        if record["id"] not in chosen:
            continue
        panels = [("Reference", record["mask"])]
        for name in ("clipseg_frozen", "clipseg_reloaded", "sam_box", "sam2_box"):
            panels.append(
                (
                    name,
                    np.asarray(Image.open(root / "outputs/predictions" / name / f"{record['id']}.png")) > 0,
                )
            )
        canvas = Image.new("RGB", (240 * len(panels), 270), "white")
        draw = ImageDraw.Draw(canvas)
        for index, (name, mask) in enumerate(panels):
            pixels = np.asarray(record["image"]).copy()
            pixels[mask] = (pixels[mask] * 0.5 + np.array([0, 220, 120]) * 0.5).astype(np.uint8)
            panel = Image.fromarray(pixels)
            panel.thumbnail((240, 240))
            canvas.paste(panel, (index * 240, 30))
            draw.text((index * 240 + 3, 5), name, fill="black")
        relative = f"figures/{record['id']}.png"
        (root / "outputs/figures").mkdir(exist_ok=True)
        canvas.save(root / "outputs" / relative)
        figure_files.append(relative)
    summary = {
        "status": "completed_current_run",
        "release_qualification": "Requires maintainer review of exact executed notebook and outputs",
        "counts": {r: len(v) for r, v in data.items()},
        "adaptation": read_json(root / "outputs/training_history.json"),
        "reload": read_json(root / "outputs/reload_parity.json"),
        "metric_summary": summaries,
        "source_manifest_sha256": digest(root / "data/records.json"),
        "figures": figure_files,
        "comparisons": "CLIPSeg frozen/adapted text; SAM/SAM2 matched oracle prompts; no overall winner.",
        "private_images_exported": False,
        "private_byod": private,
    }
    write_json(root / "outputs/run_summary.json", summary)
    # Explicit generated-file allowlist: no arbitrary directory walk, tokens, raw BYOD data or npy files.
    filenames = [
        "run_summary.json",
        "sample_manifest.json",
        "prompt_manifest.json",
        "per_image_metrics.csv",
        "metrics.json",
        "paired_changes.json",
        "training_history.json",
        "reload_parity.json",
        "controlled_experiment.json",
        "prepare.json",
        "clipseg.json",
        "sam.json",
        "sam2.json",
        "frozen_experiment.json",
        "selection.json",
        "box_experiment.json",
        "summary.md",
    ]
    import importlib.metadata
    import platform

    provenance = {
        "python": platform.python_version(),
        "source": read_json(root / "source.json"),
        "dependency_lock_sha256": digest(root / "requirements.txt"),
        "packages": {d.metadata["Name"]: d.version for d in importlib.metadata.distributions()},
        "report_process_peak_host_rss_kib": peak_host_rss(),
        "stage_receipt_scope": "Snapshot before report completed; prior stages must be complete",
        "stages": read_json(root / "outputs/stage_receipts.json"),
        "models": {
            name: read_json(root / "weights" / name / "dimer-base-manifest.json")
            for name in ("clipseg", "sam", "sam2")
        },
    }
    write_json(root / "outputs/provenance.json", provenance)
    filenames.append("provenance.json")
    from clipseg_reference.pipeline import ADAPTER_MANIFEST, ADAPTER_WEIGHTS

    filenames += [f"adapter/{ADAPTER_MANIFEST}", f"adapter/{ADAPTER_WEIGHTS}"]
    if not private:
        filenames += figure_files
    write_json(root / "outputs/checksums.json", {p: digest(root / "outputs" / p) for p in filenames})
    filenames.append("checksums.json")
    with zipfile.ZipFile(root / "outputs/segmentation_results.zip", "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in filenames:
            bundle.write(root / "outputs" / name, name)
    with zipfile.ZipFile(root / "outputs/segmentation_results.zip") as bundle:
        for name, expected in read_json(root / "outputs/checksums.json").items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != expected:
                raise RuntimeError("Report ZIP reload integrity failed")
    write_json(
        root / "outputs/report_identity.json",
        {"sha256": digest(root / "outputs/segmentation_results.zip"), "reload_verified": True},
    )
    print(
        json.dumps({"summary": summaries, "bundle": str(root / "outputs/segmentation_results.zip")}),
        flush=True,
    )


def begin_stage(root: Path, stage: str) -> dict:
    path = root / "outputs/stage_receipts.json"
    if stage == "prepare":
        receipts = {"run_id": uuid.uuid4().hex, "stages": {}}
    else:
        receipts = read_json(path)
        needed = {
            "clipseg": ["prepare"],
            "reload": ["prepare", "clipseg"],
            "sam": ["prepare"],
            "sam2": ["prepare"],
            "report": ["prepare", "clipseg", "reload", "sam", "sam2"],
        }[stage]
        for prior in needed:
            if receipts["stages"].get(prior, {}).get("status") != "complete":
                raise RuntimeError(f"Current run needs a completed {prior} stage before {stage}")
        for name, key in (
            ("data/records.json", "data_sha256"),
            ("outputs/prompt_manifest.json", "prompts_sha256"),
        ):
            if digest(root / name) != receipts[key]:
                raise RuntimeError("Data or prompt manifest changed: rerun prepare and all downstream stages")
    # Invalidate descendants when repeating a stage so old qualification cannot survive.
    descendants = {
        "clipseg": ("reload", "report"),
        "reload": ("report",),
        "sam": ("report",),
        "sam2": ("report",),
    }
    for child in descendants.get(stage, ()):
        receipts["stages"].pop(child, None)
    receipts["stages"][stage] = {"status": "running", "pid": os.getpid()}
    write_json(path, receipts)
    return receipts


def finish_stage(root: Path, stage: str, receipts: dict, elapsed: float) -> None:
    if stage == "prepare":
        receipts["data_sha256"] = digest(root / "data/records.json")
        receipts["prompts_sha256"] = digest(root / "outputs/prompt_manifest.json")
    receipts["stages"][stage] = {"status": "complete", "pid": os.getpid(), "elapsed_seconds": elapsed}
    receipts["stages"][stage]["peak_host_rss_kib"] = peak_host_rss()
    write_json(root / "outputs/stage_receipts.json", receipts)


def infer_unlabelled(root: Path, image_path: Path, phrase: str, point: list | None, box: list | None) -> None:
    import gc

    import sam2_reference
    import sam_reference
    from clipseg_reference import pipeline as clip

    if not image_path.is_file() or image_path.stat().st_size > 20 * 1024**2:
        raise ValueError("Choose an image file up to 20 MiB")
    with Image.open(image_path) as raw:
        if not all(16 <= side <= 4096 for side in raw.size) or raw.getexif().get(274, 1) != 1:
            raise ValueError("Expected an oriented image with sides 16–4096 pixels")
        image = raw.convert("RGB")
    kwargs = {"points": [point], "point_labels": [1]} if point is not None else {"box": box}
    clip.validate_inputs(image, [phrase])
    sam_reference.validate_inputs(image, **kwargs)
    sam2_reference.validate_inputs(image, **kwargs)
    selected_device = device()
    output = root / "unlabelled" / uuid.uuid4().hex[:12]
    output.mkdir(parents=True)
    results = []
    for name, cls in (
        ("clipseg", clip.ClipSegSegmentationPipeline),
        ("sam", sam_reference.SAMViTSegmentationPipeline),
        ("sam2", sam2_reference.SAM2SegmentationPipeline),
    ):
        pipe = (
            cls.from_artifact(
                root / "outputs/adapter", device=selected_device, weights_dir=root / "weights/clipseg"
            )
            if name == "clipseg"
            else cls.from_pretrained(
                device=selected_device, weights_dir=root / "weights" / name, allow_download=True
            )
        )
        result = (
            pipe.segment(image, [phrase], threshold=0.5)
            if name == "clipseg"
            else pipe.segment(image, **kwargs, multimask=True)
        )
        mask = (
            result["segments"][0]["mask"]
            if name == "clipseg"
            else result["masks"][int(np.argmax(result["iou_scores"]))]
        )
        Image.fromarray(mask.astype(np.uint8) * 255).save(output / f"{name}.png")
        results.append(
            {
                "model": name,
                "area_fraction": float(mask.mean()),
                "iou": None,
                "dice": None,
                "reason": "not-measurable: no reference mask",
                "prompt": phrase if name == "clipseg" else kwargs,
            }
        )
        del pipe
        gc.collect()
        import torch

        torch.cuda.empty_cache()
    write_json(output / "results.json", {"input_sha256": digest(image_path), "results": results})
    write_json(root / "unlabelled_output.json", {"path": str(output)})
    print(json.dumps({"path": str(output), "results": results}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--stage",
        choices=("prepare", "clipseg", "reload", "sam", "sam2", "report", "byod", "infer"),
        required=True,
    )
    parser.add_argument("--byod", type=Path)
    parser.add_argument("--image", type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--point", type=json.loads)
    parser.add_argument("--box", type=json.loads)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.stage == "infer":
        if args.image is None or (args.point is None) == (args.box is None):
            raise ValueError(
                "Unlabelled inference requires --image, --prompt and exactly one --point or --box"
            )
        infer_unlabelled(root, args.image, args.prompt, args.point, args.box)
        return
    receipts = begin_stage(root, args.stage) if args.stage != "byod" else None
    started = time.perf_counter()
    if args.stage == "prepare":
        try:
            prepare(root, args.byod)
        except ValueError as error:
            # Only a documented BYOD input rule (InputRefused) is a refusal; anything else is a genuine failure.
            if args.byod is None or not isinstance(error, InputRefused):
                raise
            receipts["stages"][args.stage]["status"] = "refused"
            write_json(root / "outputs/stage_receipts.json", receipts)
            refuse(error)
    elif args.stage == "clipseg":
        clipseg(root)
    elif args.stage == "reload":
        reload_stage(root)
    elif args.stage in ("sam", "sam2"):
        sam_stage(root, args.stage)
    elif args.stage == "report":
        report(root)
    else:
        if args.byod is None:
            raise ValueError("--byod must name a directory containing manifest.json")
        child = root / "byod_runs" / uuid.uuid4().hex[:12]
        child.mkdir(parents=True)
        (child / "weights").symlink_to(root / "weights", target_is_directory=True)
        for name in ("source.json", "requirements.txt"):
            shutil.copyfile(root / name, child / name)
        write_json(root / "byod_output.json", {"path": str(child)})
        for stage in ("prepare", "clipseg", "reload", "sam", "sam2", "report"):
            command = [sys.executable, __file__, "--root", str(child), "--stage", stage]
            if stage == "prepare":
                command += ["--byod", str(args.byod.resolve())]
            completed = subprocess.run(command)
            if completed.returncode == REFUSED_EXIT:
                raise SystemExit(REFUSED_EXIT)  # the child printed the reason once
            if completed.returncode:
                raise subprocess.CalledProcessError(completed.returncode, command)
    if receipts is not None:
        finish_stage(root, args.stage, receipts, time.perf_counter() - started)


if __name__ == "__main__":
    main()
