"""Model-backed checks that run only where the pinned snapshot is staged: the measured model facts, batched
segmentation equal to single-image segmentation, the decoder on cached CLIP activations equal to the full forward,
corpus evaluation on drawn shapes, a short adaptation of the decoder, the artifact round trip with reload parity,
the loader's scope check, the transactional guarantee and — where CUDA is visible — the same path on the
accelerator. Skipped when the weights are absent."""
# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest
from PIL import Image, ImageDraw

from clipseg_segmentation_pipeline import (
    DECODER_PARAMETERS,
    DEFAULT_WEIGHTS_DIR,
    PARAMETER_COUNT,
    WEIGHTS_FILE,
    ClipSegSegmentationPipeline,
)
from clipseg_segmentation_pipeline.pipeline import _TRAINABLE_PREFIXES

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / WEIGHTS_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

SHAPES = (("a red square", (220, 30, 30)), ("a blue circle", (30, 60, 220)), ("a green triangle", (30, 160, 60)))


def _record(i, size=(256, 192)):
    """A white scene with one coloured shape; the phrase names it and the mask is its pixels."""
    rng = np.random.default_rng(i)
    kind = i % len(SHAPES)
    x, y = int(rng.integers(16, 120)), int(rng.integers(16, 80))
    image = Image.new("RGB", size, (255, 255, 255))
    draw = ImageDraw.Draw(image)
    if kind == 0:
        draw.rectangle((x, y, x + 80, y + 80), fill=SHAPES[0][1])
    elif kind == 1:
        draw.ellipse((x, y, x + 80, y + 80), fill=SHAPES[1][1])
    else:
        draw.polygon([(x, y + 80), (x + 40, y), (x + 80, y + 80)], fill=SHAPES[2][1])
    image.putpixel((i % size[0], 0), (i % 256, 0, 0))
    mask = np.asarray(image.convert("RGB")).sum(axis=2) < 700  # coloured pixels
    return {"id": f"shape{i:02d}", "image": image, "prompt": SHAPES[kind][0], "mask": mask}


@pytest.fixture(autouse=True)
def _release_memory():
    yield
    import gc

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def records():
    return [_record(i) for i in range(24)]


@pytest.fixture(scope="module")
def pipe():
    return ClipSegSegmentationPipeline.from_pretrained(weights_dir=DEFAULT_WEIGHTS_DIR)


def test_model_facts_batched_segmentation_decoder_parity_and_frozen_evaluation(pipe, records):
    assert sum(p.numel() for p in pipe._model.parameters()) == PARAMETER_COUNT
    assert sum(p.numel() for n, p in pipe._model.named_parameters() if n.startswith(_TRAINABLE_PREFIXES)) == DECODER_PARAMETERS
    assert pipe.weight_sha256 is not None and len(pipe.weight_sha256) == 64
    single = [pipe.segment(r["image"], [r["prompt"]])["segments"][0] for r in records[:4]]
    batched = pipe.segment_batch([(r["image"], r["prompt"]) for r in records[:4]], batch_size=4)
    for s, b in zip(single, batched, strict=True):
        # batch-size-dependent CPU matmul kernels move the sigmoid by O(1e-4); a pairing bug would move it by O(0.5)
        assert np.abs(s["probability"] - b["probability"]).max() < 1e-3 and s["prompt"] == b["prompt"]
    # the decoder on cached activations equals the full forward
    activations, conditionals, targets = pipe._encode(records[:3], 8)
    inputs = pipe._processor(text=[r["prompt"] for r in records[:3]], images=[r["image"] for r in records[:3]], padding=True, return_tensors="pt").to(pipe.device)
    with torch.no_grad():
        head = pipe._decoder_logits([a.float() for a in activations], conditionals).float().cpu().numpy()
        full = pipe._model(**inputs).logits.float().cpu().numpy()
    assert np.abs(head - full).max() < 5e-2 and tuple(targets.shape) == (3, 352, 352)  # fp16 cache
    metrics = pipe.evaluate(records[:12])
    assert metrics["n"] == 12 and metrics["adapted"] is False and metrics["verdict"] == "measured-small-sample" and metrics["threshold"] == 0.5
    assert 0.0 <= metrics["miou"] <= 1.0 and len(metrics["rows"]) == 12 and metrics["rows"][0]["prompt"] == "a red square"


def test_short_adaptation_and_artifact_round_trip(pipe, records, tmp_path):
    result = pipe.adapt(records[:18], records[18:], epochs=2, lr=1e-4, batch_size=6)
    assert result["n_trainable"] == DECODER_PARAMETERS and result["n_total"] == PARAMETER_COUNT and result["threshold"] == 0.5
    assert result["history"][0]["note"] == "frozen model" and result["history"][1]["train_loss"] > 0.0
    assert set(result["history"][1]["val"]) == {"miou", "iou_micro", "dice", "pixel_precision", "pixel_recall", "n"} and result["best_epoch"] in (0, 1, 2)
    assert all(n.startswith("decoder.") for n in result["trainable_names"])
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["tensors"]) == len(result["trainable_names"]) and manifest["base"]["weight_sha256"] == pipe.weight_sha256
    assert manifest["metadata"] == {"note": "test"} and manifest["adapter"]["selection"] == "highest validation mean IoU"
    reloaded = ClipSegSegmentationPipeline.from_artifact(artifact, weights_dir=DEFAULT_WEIGHTS_DIR)
    a = pipe.segment_batch([(r["image"], r["prompt"]) for r in records[:4]])
    b = reloaded.segment_batch([(r["image"], r["prompt"]) for r in records[:4]])
    assert all(np.abs(x["probability"] - y["probability"]).max() < 1e-5 for x, y in zip(a, b, strict=True))
    assert reloaded.adapter["best_epoch"] == result["best_epoch"] and reloaded.evaluate(records[:12])["adapted"] is True
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_no_validation_keeps_the_final_epoch_and_reloads_it(pipe, records, tmp_path):
    result = pipe.adapt(records[:18], None, epochs=2, batch_size=6)
    assert result["best_epoch"] == 2 == result["epochs"] and result["selection"].startswith("final epoch")
    assert all(entry["val"] is None for entry in result["history"]) and len(result["history"]) == 3
    artifact = pipe.save_artifact(tmp_path / "final")
    reloaded = ClipSegSegmentationPipeline.from_artifact(artifact, weights_dir=DEFAULT_WEIGHTS_DIR)
    state, other = pipe._model.state_dict(), reloaded._model.state_dict()
    assert all(torch.equal(state[name], other[name]) for name in result["trainable_names"])


def test_adapt_refuses_bad_hyperparameters_and_datasets(pipe, records):
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(records[:18], None, epochs=0)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(records[:18], None, epochs=1, lr=0.5)
    with pytest.raises(ValueError, match="batch_size"):
        pipe.adapt(records[:18], None, epochs=1, batch_size=0)
    with pytest.raises(ValueError, match="8..5000"):
        pipe.adapt(records[:4], None, epochs=1)
    with pytest.raises(ValueError, match="no true pixel"):
        pipe.adapt([*records[:17], {**records[17], "mask": np.zeros_like(records[17]["mask"])}], None, epochs=1)
    assert not any(p.requires_grad for p in pipe._model.parameters())


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_configuration(pipe, records, tmp_path):
    from safetensors.torch import load_file, save_file

    pipe.adapt(records[:18], None, epochs=1, batch_size=6)
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    fewer = tmp_path / "fewer"
    shutil.copytree(artifact, fewer)
    (fewer / "manifest.json").write_text(json.dumps({**manifest, "tensors": manifest["tensors"][:-1]}))
    with pytest.raises(ValueError, match="does not match its recorded configuration"):
        ClipSegSegmentationPipeline.from_artifact(fewer, weights_dir=DEFAULT_WEIGHTS_DIR)
    extra = tmp_path / "extra"
    shutil.copytree(artifact, extra)
    tensors = load_file(str(extra / "adapter.safetensors"))
    tensors["decoder.zz_extra"] = torch.zeros(1)
    save_file(tensors, str(extra / "adapter.safetensors"), metadata={"format": "pt"})
    digest = hashlib.sha256((extra / "adapter.safetensors").read_bytes()).hexdigest()
    files = [{**manifest["files"][0], "bytes": (extra / "adapter.safetensors").stat().st_size, "sha256": digest}]
    (extra / "manifest.json").write_text(json.dumps({**manifest, "files": files}))
    with pytest.raises(ValueError, match="tensor names differ"):
        ClipSegSegmentationPipeline.from_artifact(extra, weights_dir=DEFAULT_WEIGHTS_DIR)
    tower = tmp_path / "tower"
    shutil.copytree(artifact, tower)
    (tower / "manifest.json").write_text(json.dumps({**manifest, "tensors": [*manifest["tensors"], "clip.vision_model.encoder.layers.0.x"]}))
    with pytest.raises(ValueError, match="CLIPSeg decoder"):
        ClipSegSegmentationPipeline.from_artifact(tower, weights_dir=DEFAULT_WEIGHTS_DIR)


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe, records):
    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}
    adapter_before = pipe.adapter

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(records[:18], None, epochs=2, batch_size=6, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before)
    assert pipe.adapter is adapter_before
    assert not any(p.requires_grad for p in pipe._model.parameters())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not visible")
def test_the_default_device_is_cuda_when_visible(pipe):
    assert pipe.device == "cuda:0" and next(pipe._model.parameters()).device.type == "cuda"
