"""Offline checks of the adaptation contract: the (image, phrase, mask) record contract and its refusals, the
pinned-corpus refusals and the draw, splitting, the BYOD loader, the metrics and baselines, `segment_batch` /
`evaluate` with an injected runner, the artifact-manifest checks, and the model-free refusals of `adapt` / artifacts."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from PIL import Image

from clipseg_segmentation_pipeline import (
    ARTIFACT_FORMAT,
    FOODSEG103_CLASSES,
    MODEL_ID,
    MODEL_REVISION,
    SAMPLE_SPLIT,
    ClipSegSegmentationPipeline,
    build_sample_dataset,
    check_split_disjoint,
    coerce_mask,
    dataset_digest,
    empty_baseline,
    fetch_corpus,
    full_baseline,
    image_digest,
    largest_class,
    load_byod_dataset,
    mask_digest,
    mask_metrics,
    read_corpus,
    segmentation_metrics,
    split_dataset,
    validate_dataset,
    write_dataset_csv,
)
from clipseg_segmentation_pipeline import pipeline as pl
from clipseg_segmentation_pipeline import samples as sm

PROMPTS = ["bread", "steak", "rice", "broccoli"]


def _image(i, size=(64, 48)):
    image = Image.new("RGB", size, ((i * 37) % 256, 120, 90))
    image.putpixel((i % size[0], 0), (255, 0, 0))
    return image


def _mask(i, size=(64, 48)):
    mask = np.zeros((size[1], size[0]), dtype=bool)
    x = (i * 7) % 40
    mask[8:32, x : x + 16] = True
    return mask


def _record(i, prompt=None, **extra):
    return {"id": f"r{i:03d}", "image": _image(i), "prompt": PROMPTS[i % len(PROMPTS)] if prompt is None else prompt, "mask": _mask(i), **extra}


def _records(n=12):
    return [_record(i) for i in range(n)]


def _echo_runner(image, queries):
    """An injected runner that returns the record's own mask as a probability map (a side table by pixel digest)."""
    mask = _echo_runner.table.get(image_digest(image))
    prob = np.full((image.height, image.width), 0.1, dtype=np.float32)
    if mask is not None:
        prob[mask] = 0.9
    return np.stack([prob] * len(queries))


_echo_runner.table = {}


# --- record contract -------------------------------------------------------------------------------------------


def test_validate_dataset_accepts_records_and_reports_counts_and_digest():
    info = validate_dataset(_records())
    assert info["n_records"] == 12 and info["n_prompts"] == 4 and info["prompts"] == sorted(PROMPTS)
    assert info["image_width"] == {"min": 64, "max": 64} and 0.0 < info["mask_area_fraction"]["mean"] < 1.0
    assert info["digest"] == dataset_digest(_records()) and info["model_id"] == MODEL_ID
    # prompts are normalised like queries; masks may be images (non-zero = true)
    picture = Image.fromarray((_mask(0) * 255).astype(np.uint8))
    item = validate_dataset([_record(0, prompt="  Bread. ", mask=picture)] + _records()[1:])["records"][0]
    assert item["prompt"] == "bread" and np.array_equal(item["mask"], _mask(0))


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda r: r.pop("mask"), "missing 'mask'"),
        (lambda r: r.update(id="bad id!"), "id must match"),
        (lambda r: r.update(image=Image.new("RGB", (8, 8))), "MIN_IMAGE_SIDE"),
        (lambda r: r.update(prompt="   "), "must not be empty"),
        (lambda r: r.update(prompt="x" * 65), "MAX_PROMPT_CHARS"),
        (lambda r: r.update(mask=np.zeros((10, 10), dtype=bool)), "does not match"),
        (lambda r: r.update(mask=np.zeros((48, 64), dtype=bool)), "no true pixel"),
        (lambda r: r.update(mask="not a mask"), "image file not found"),
    ],
)
def test_validate_dataset_refuses_malformed_records(mutate, message):
    records = _records()
    mutate(records[3])
    with pytest.raises(ValueError, match=message):
        validate_dataset(records)


def test_validate_dataset_enforces_bounds_and_unique_ids():
    with pytest.raises(ValueError, match="8..5000 are required"):
        validate_dataset(_records(4))
    with pytest.raises(ValueError, match="records must be a list"):
        validate_dataset({"id": "x"})
    dup = _records()
    dup[1]["id"] = dup[0]["id"]
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset(dup)
    assert coerce_mask(np.ones((48, 64), dtype=np.uint8), (64, 48)).all()
    with pytest.raises(ValueError, match="boolean or numeric"):
        coerce_mask(np.array([["a"] * 64] * 48), (64, 48))


def test_validate_dataset_refuses_before_importing_model_libraries(forbid_model_imports):
    with pytest.raises(ValueError, match="no true pixel"):
        validate_dataset([_record(0, mask=np.zeros((48, 64), dtype=bool))] + _records()[1:])


def test_digests_and_split_disjointness():
    a, b = _record(0), _record(0)
    assert image_digest(a["image"]) == image_digest(b["image"]) and mask_digest(a["mask"]) == mask_digest(b["mask"])
    assert image_digest(_record(1)["image"]) != image_digest(a["image"])
    assert dataset_digest([a, _record(1)]) == dataset_digest([_record(1), a])
    assert dataset_digest([a]) != dataset_digest([dict(a, prompt="rice")])
    splits = {"train": _records()[:9], "test": _records()[9:]}
    assert check_split_disjoint(splits) == {"train": 9, "test": 3}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint({"train": _records()[:9], "test": [_record(0)]})


def test_split_dataset_is_seeded_and_deduplicates():
    records = _records(20)
    splits = split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=1)
    assert check_split_disjoint(splits) == {"test": 4, "validation": 4, "train": 12}
    assert splits == split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=1)
    assert splits != split_dataset(records, val_fraction=0.2, test_fraction=0.2, seed=2)
    dup = records + [dict(records[0], id="dup")]
    assert sum(len(part) for part in split_dataset(dup).values()) == 20
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)


# --- pinned corpus ---------------------------------------------------------------------------------------------


def test_pins_classes_and_draw_sizes():
    assert sm.CORPUS_REVISION == "176acc3edd2432ee126bda6fb01469eadeb018df" and sm.CORPUS_ROW_GROUPS == 8
    assert sorted(sm.ROW_GROUP_PINS) == list(range(8)) and all(len(d) == 64 and n > 4_000_000 for d, n in sm.ROW_GROUP_PINS.values())
    assert len(FOODSEG103_CLASSES) == 104 and FOODSEG103_CLASSES[0] == "background" and FOODSEG103_CLASSES[58] == "bread" and FOODSEG103_CLASSES[103] == "other ingredients"
    assert SAMPLE_SPLIT == {"train": 600, "validation": 60, "test": 140} and len(sm.SAMPLE_DIGEST) == 64
    assert largest_class(np.array([[0, 0, 58], [58, 46, 103]])) == 58
    assert largest_class(np.array([[0, 103], [103, 0]])) is None


def _png(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def _fake_rows(n=6):
    rows = []
    for i in range(n):
        image = np.full((32, 32, 3), (i * 30) % 256, dtype=np.uint8)
        label = np.zeros((32, 32), dtype=np.uint8)
        label[4:20, 4:20] = 58 if i % 2 == 0 else 46
        label[24:30, 24:30] = 103
        rows.append({"image": {"bytes": _png(image), "path": f"{i}.jpg"}, "label": {"bytes": _png(label), "path": f"{i}.png"}, "id": i})
    return rows


def test_fetch_corpus_refuses_a_row_group_that_does_not_match_its_pin(tmp_path, monkeypatch):
    rows = _fake_rows()
    table = pa.Table.from_pylist(rows)
    shard = tmp_path / "shard.parquet"
    pq.write_table(table, shard)
    monkeypatch.setattr(sm, "CORPUS_ROWS", len(rows))
    with pytest.raises(ValueError, match="sha256"):
        fetch_corpus(cache_dir=tmp_path / "cache", groups=[0], opener=lambda url: str(shard))
    assert not (tmp_path / "cache" / "validation-rg0.parquet").exists()
    with pytest.raises(ValueError, match="no pin"):
        fetch_corpus(cache_dir=tmp_path / "cache", groups=[30], opener=lambda url: str(shard))
    pq.write_table(table, tmp_path / "cache" / "validation-rg0.parquet")  # a stale cache file is refetched and refused the same way
    with pytest.raises(ValueError, match="sha256"):
        fetch_corpus(cache_dir=tmp_path / "cache", groups=[0], opener=lambda url: str(shard))


def test_read_corpus_picks_the_largest_ingredient_and_skips_unusable_images():
    rows = [{"image": r["image"]["bytes"], "label": r["label"]["bytes"], "id": r["id"]} for r in _fake_rows(4)]
    only_other = np.zeros((32, 32), dtype=np.uint8)
    only_other[:16] = 103
    rows.append({"image": rows[0]["image"], "label": _png(only_other), "id": 99})
    records = read_corpus({0: rows})
    assert len(records) == 4 and records[0]["prompt"] == "bread" and records[1]["prompt"] == "steak"
    assert records[0]["id"] == "foodseg103-val-0" and records[0]["class_id"] == 58 and records[0]["source_id"] == 0
    assert records[0]["mask"].shape == (32, 32) and int(records[0]["mask"].sum()) == 256


def test_build_sample_dataset_draws_seeded_sizes():
    records = _records(20)
    splits = build_sample_dataset(records, seed=3, sizes={"train": 12, "validation": 3, "test": 5})
    assert {k: len(v) for k, v in splits.items()} == {"train": 12, "validation": 3, "test": 5}
    assert splits == build_sample_dataset(records, seed=3, sizes={"train": 12, "validation": 3, "test": 5})
    with pytest.raises(ValueError, match="need"):
        build_sample_dataset(records[:5], sizes={"train": 12, "validation": 3, "test": 5})


@pytest.mark.skipif(not all((sm.DEFAULT_CACHE_DIR / f"validation-rg{g}.parquet").is_file() for g in range(8)), reason="pinned row groups not cached")
def test_default_draw_matches_the_pinned_digest_when_the_row_groups_are_cached():
    splits = build_sample_dataset(read_corpus(fetch_corpus()))
    assert check_split_disjoint(splits) == SAMPLE_SPLIT
    assert dataset_digest(splits["train"] + splits["validation"] + splits["test"]) == sm.SAMPLE_DIGEST


# --- BYOD --------------------------------------------------------------------------------------------------------


def test_load_byod_dataset_reads_images_masks_and_prompts_from_a_zip_or_directory(tmp_path):
    root = tmp_path / "masks"
    root.mkdir()
    for i in range(3):
        _image(i).save(root / f"img{i}.png")
        Image.fromarray((_mask(i) * 255).astype(np.uint8)).save(root / f"img{i}_mask.png")
    (root / "masks.csv").write_text("file,mask,prompt\nimg0.png,img0_mask.png,bread\nimg1.png,img1_mask.png,Steak.\nimg2.png,img2_mask.png,rice\n", encoding="utf-8")
    records = load_byod_dataset(root)
    assert [r["id"] for r in records] == ["img0", "img1", "img2"]
    checked = validate_dataset(records, min_records=1)["records"]
    assert checked[1]["prompt"] == "steak" and np.array_equal(checked[0]["mask"], _mask(0))
    archive = tmp_path / "masks.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for file in root.iterdir():
            z.write(file, f"nested/{file.name}")
    assert [r["id"] for r in load_byod_dataset(archive)] == ["img0", "img1", "img2"]
    _image(9).save(root / "extra.png")
    with pytest.raises(ValueError, match="no masks.csv row"):
        load_byod_dataset(root)
    (root / "extra.png").unlink()
    (root / "masks.csv").write_text("file,mask,prompt\nmissing.png,img0_mask.png,bread\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing file"):
        load_byod_dataset(root)
    (root / "masks.csv").write_text("file,text\nimg0.png,x\n", encoding="utf-8")
    with pytest.raises(ValueError, match="columns file, mask and prompt"):
        load_byod_dataset(root)
    with pytest.raises(ValueError, match="neither a directory nor a zip"):
        load_byod_dataset(tmp_path / "nope.txt")
    out = write_dataset_csv(checked, tmp_path / "out" / "train.csv")
    lines = out.read_text(encoding="utf-8").splitlines()
    assert lines[0] == "id,file,mask,prompt,width,height,mask_area_fraction,class_id,source_id,source_row_group" and len(lines) == 4


# --- metrics and baselines --------------------------------------------------------------------------------------


def test_mask_metrics_segmentation_metrics_and_baselines():
    records = _records()
    perfect = segmentation_metrics([r["mask"] for r in records], records)
    assert perfect["miou"] == 1.0 and perfect["iou_micro"] == 1.0 and perfect["dice"] == 1.0 and perfect["pixel_recall"] == 1.0
    half = np.zeros((48, 64), dtype=bool)
    half[8:20, 0:64] = True  # rows 8..20 of every column: overlaps the 24-row-tall box for 12 of its rows
    m = mask_metrics(half, _mask(0))
    assert m["intersection"] == 12 * 16 and m["reference"] == 24 * 16 and m["predicted"] == 12 * 64
    assert m["iou"] == pytest.approx(192 / (384 + 768 - 192)) and m["recall"] == 0.5 and m["precision"] == 0.25
    with pytest.raises(ValueError, match="shape"):
        mask_metrics(np.zeros((10, 10), dtype=bool), _mask(0))
    with pytest.raises(ValueError, match="predictions for"):
        segmentation_metrics([half], records)
    empty = empty_baseline(records)
    assert empty["miou"] == 0.0 and empty["pixel_recall"] == 0.0 and empty["baseline"].startswith("empty")
    full = full_baseline(records)
    assert full["pixel_recall"] == 1.0 and full["miou"] == pytest.approx(float(_mask(0).mean())) and full["baseline"].startswith("full")


# --- segment_batch / evaluate with an injected runner ----------------------------------------------------------


def test_segment_batch_and_evaluate_score_the_runner_and_flag_small_samples():
    records = _records()
    _echo_runner.table = {image_digest(r["image"]): r["mask"] for r in records[:6]}
    pipe = ClipSegSegmentationPipeline(_echo_runner, "cpu")
    items = pipe.segment_batch([(r["image"], r["prompt"]) for r in records], batch_size=5)
    assert len(items) == 12 and items[0]["prompt"] == "bread" and np.array_equal(items[0]["mask"], records[0]["mask"]) and items[7]["area_fraction"] == 0.0
    report = pipe.evaluate(records)
    assert report["n"] == 12 and report["miou"] == 0.5 and report["verdict"] == "measured-small-sample" and report["adapted"] is False and report["threshold"] == 0.5
    assert pipe.evaluate(records, threshold=0.05)["pixel_recall"] == 1.0  # everything above 0.05 -> full masks
    with pytest.raises(ValueError, match="batch_size"):
        pipe.segment_batch([(records[0]["image"], "bread")], batch_size=0)
    with pytest.raises(ValueError, match="threshold"):
        pipe.evaluate(records, threshold=1.5)
    batched = ClipSegSegmentationPipeline(_echo_runner, "cpu", lambda images, queries: [_echo_runner(im, [q])[0] for im, q in zip(images, queries, strict=True)])
    assert [i["area_fraction"] for i in batched.segment_batch([(r["image"], r["prompt"]) for r in records])] == [i["area_fraction"] for i in items]
    wrong = ClipSegSegmentationPipeline(_echo_runner, "cpu", lambda images, queries: [np.zeros((3, 3), dtype=np.float32)] * len(images))
    with pytest.raises(RuntimeError, match="probability map of shape"):
        wrong.segment_batch([(records[0]["image"], "bread")])


def test_adapt_and_artifacts_require_a_loaded_model(forbid_model_imports):
    pipe = ClipSegSegmentationPipeline(_echo_runner, "cpu")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.adapt(_records(), None)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), None, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), None, lr=0.5)
    with pytest.raises(ValueError, match="threshold"):
        pipe.adapt(_records(), None, threshold=2)
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.save_artifact("x")
    with pytest.raises(RuntimeError, match="no loaded model"):
        pipe.load_artifact("x")


# --- artifact manifest checks --------------------------------------------------------------------------------------


def _manifest(tmp_path, **overrides):
    weights = tmp_path / "adapter.safetensors"
    weights.write_bytes(b"tensor-bytes")
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base": {"model_id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": "base-digest"},
        "tensors": ["decoder.film_mul.weight", "decoder.layers.0.self_attn.q_proj.weight", "decoder.transposed_convolution.0.weight"],
        "adapter": {"best_epoch": 1, "threshold": 0.5},
        "files": [{"path": "adapter.safetensors", "bytes": weights.stat().st_size, "sha256": hashlib.sha256(b"tensor-bytes").hexdigest()}],
    }
    manifest.update(overrides)
    return manifest


def test_check_artifact_manifest_accepts_a_consistent_manifest_and_refuses_each_deviation(tmp_path):
    pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="format"):
        pl._check_artifact_manifest(_manifest(tmp_path, format="other"), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="trained on"):
        pl._check_artifact_manifest(_manifest(tmp_path, base={"model_id": "x", "revision": MODEL_REVISION, "weight_sha256": "base-digest"}), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="base weight digest"):
        pl._check_artifact_manifest(_manifest(tmp_path), tmp_path, "another-digest")
    bad = _manifest(tmp_path)
    bad["files"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="sha256"):
        pl._check_artifact_manifest(bad, tmp_path, "base-digest")
    with pytest.raises(ValueError, match="exactly adapter.safetensors"):
        pl._check_artifact_manifest(_manifest(tmp_path, files=[]), tmp_path, "base-digest")
    for name in ("clip.vision_model.encoder.layers.0.x", "clip.text_model.encoder.layers.0.x", "clip.visual_projection.weight", "clip.text_projection.weight"):
        with pytest.raises(ValueError, match="CLIPSeg decoder"):
            pl._check_artifact_manifest(_manifest(tmp_path, tensors=[name]), tmp_path, "base-digest")
    with pytest.raises(ValueError, match="adapter.threshold"):
        pl._check_artifact_manifest(_manifest(tmp_path, adapter={"best_epoch": 1}), tmp_path, "base-digest")


def test_trainable_names_selects_the_decoder():
    class _Param:
        def numel(self):
            return 1

    class _Model:
        def named_parameters(self):
            names = ["clip.vision_model.encoder.layers.0.x", "clip.visual_projection.weight", "decoder.film_mul.weight", "decoder.reduces.0.weight", "decoder.layers.2.mlp.fc1.weight", "decoder.transposed_convolution.0.weight", "clip.text_model.encoder.layers.0.x", "clip.text_projection.weight"]
            return [(n, _Param()) for n in names]

    assert pl._trainable_names(_Model()) == ["decoder.film_mul.weight", "decoder.reduces.0.weight", "decoder.layers.2.mlp.fc1.weight", "decoder.transposed_convolution.0.weight"]
    assert pl._TRAINABLE_PREFIXES == ("decoder.",) and pl.DECODER_PARAMETERS == 1_127_009 and pl.PARAMETER_COUNT == 150_747_746 and pl.EXTRACT_LAYERS == (3, 6, 9)


def test_manifest_json_round_trip(tmp_path):
    payload = {"epoch": 1, "train_loss": 0.3, "val": {"miou": 0.5, "iou_micro": 0.6, "dice": 0.6, "pixel_precision": 0.7, "pixel_recall": 0.6, "n": 60}}
    (tmp_path / "h.json").write_text(json.dumps([payload]), encoding="utf-8")
    assert json.loads((tmp_path / "h.json").read_text(encoding="utf-8"))[0]["val"]["miou"] == 0.5
