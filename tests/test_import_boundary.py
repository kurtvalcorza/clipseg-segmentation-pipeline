"""Import-boundary contract (fleet RTM-001).

Rejected requests never import model libraries; valid snapshots still reach them.
"""

import hashlib
import json

import pytest

from clipseg_segmentation_pipeline.pipeline import (
    MANIFEST_NAME,
    MODEL_ID,
    MODEL_REVISION,
    ClipSegSegmentationPipeline,
)

_CONFIG = json.dumps({"model_type": "clipseg", "projection_dim": 512}).encode()


def _snapshot(root, tamper=False):
    (root / "config.json").write_bytes(_CONFIG)
    digest = "0" * 64 if tamper else hashlib.sha256(_CONFIG).hexdigest()
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [{"path": "config.json", "bytes": len(_CONFIG), "sha256": digest}],
    }
    (root / MANIFEST_NAME).write_text(json.dumps(manifest), encoding="utf-8")


def test_from_pretrained_refuses_without_snapshot_before_model_imports(tmp_path, forbid_model_imports):
    with pytest.raises(FileNotFoundError, match="allow_download=False"):
        ClipSegSegmentationPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_refuses_tampered_snapshot_before_model_imports(tmp_path, forbid_model_imports):
    _snapshot(tmp_path, tamper=True)
    with pytest.raises(ValueError, match="sha256"):
        ClipSegSegmentationPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)


def test_from_pretrained_valid_snapshot_reaches_model_import(tmp_path, forbid_model_imports):
    _snapshot(tmp_path)
    with pytest.raises(AssertionError, match="model dependency imported before rejection"):
        ClipSegSegmentationPipeline.from_pretrained(device="cpu", weights_dir=tmp_path, allow_download=False)
