# CLIPSeg rd64-refined text-prompted segmentation pipeline

DIMER pipeline for **CLIPSeg** (`CIDAS/clipseg-rd64-refined`), Lüddecke and Ecker's ~151M-parameter zero-shot segmenter — a frozen CLIP ViT-B/16 with a small transformer decoder that turns a free-text phrase into a per-pixel probability map — pinned to an immutable Hugging Face revision and loaded only from a digest-verified local snapshot. The pipeline accepts one image and 1–16 phrases, returns one binary mask and one probability map per phrase at input resolution under a caller-owned threshold, and ships `mask_iou` for callers who bring reference masks; it returns no calibrated score and no instance separation. On top of inference it carries the **adaptation contract for labelled (image, phrase, mask) records**: mean and micro IoU, Dice, pixel precision and recall at a stated threshold, two non-adapted baselines, a bounded fine-tuning of the CLIPSeg decoder on cached CLIP activations, and a verified safetensors adapter that reloads against the pinned base.

## Upstream alignment

- Model: `CIDAS/clipseg-rd64-refined`
- Revision: `999e0328d9e10b484360c477313983f9afdd7050`
- Upstream weight license: Apache-2.0 (the Hub tag; the original code repository states its MIT licence does not cover the weights without naming another — recorded in `docs/WEIGHTS.md`)
- Upstream task: zero-shot / referring image segmentation (PhraseCut)
- Repository adaptation: **bounded supervised fine-tuning of the decoder** — the three transformer layers over the reduced CLIP activations, the FiLM conditioning, the reduce projections and the transposed convolution (1,127,009 of 150,747,746 parameters; the part the upstream authors trained) on `{id, image, prompt, mask}` records with the per-pixel binary cross-entropy over the 352×352 logit grid; the CLIP image and text towers stay frozen and their outputs are cached. Trained tensors are exported as a safetensors adapter with a manifest and overlaid on a freshly loaded, re-verified base. Every other phrase is inference-only; the tuned decoder serves them too, which the tutorial shows on one drawn scene and the card records.

## Quick start

```python
from PIL import Image
from clipseg_segmentation_pipeline import ClipSegSegmentationPipeline, mask_iou

pipe = ClipSegSegmentationPipeline.from_pretrained()        # stages + verifies weights/clipseg-rd64-refined first
result = pipe.segment(Image.open("scene.jpg"), ["a red circle", "a cat"], threshold=0.5)
for segment in result["segments"]:
    print(segment["prompt"], segment["area_fraction"], segment["max_probability"], segment["bbox"])
    segment["mask"]          # bool array (H, W); segment["probability"] is the float32 sigmoid map

# score against a reference mask you drew or labelled yourself (IoU; the building block for mIoU)
print(mask_iou(result["segments"][0]["mask"], reference_mask))

from clipseg_segmentation_pipeline import fetch_sample_dataset
splits = fetch_sample_dataset()                    # 800 digest-pinned FoodSeg103 records (largest ingredient + mask), 600 / 60 / 140
print(pipe.evaluate(splits["test"])["miou"])       # frozen mean IoU at MASK_THRESHOLD
pipe.adapt(splits["train"], splits["validation"])  # the decoder only, highest-validation-mIoU epoch kept
print(pipe.evaluate(splits["test"])["miou"])
pipe.save_artifact("outputs/adapter")
again = ClipSegSegmentationPipeline.from_artifact("outputs/adapter")   # re-verifies the base, checks the manifest, overlays
```

Install into a Python 3.12 environment that already holds the pinned dependencies with `pip install -e . --no-deps`; run `pytest -q -o addopts= tests` for the offline test suite (no weights needed; `tests/test_model_backed.py` runs only where the snapshot is staged). On a fresh clone the manifest is committed but the weights are not: `ClipSegSegmentationPipeline.from_pretrained(allow_download=True)` fetches exactly the missing manifest-listed files at the pinned revision, then verifies them.

## Weights layout

```
weights/clipseg-rd64-refined/
  dimer-base-manifest.json   # modelId, revision, per-file bytes + SHA-256 (8 files)
  config.json                # CLIPSegForImageSegmentation: CLIP ViT-B/16 + reduce_dim 64 decoder, complex transposed conv
  preprocessor_config.json   # ViTImageProcessor: resize 352x352, ImageNet mean/std
  merges.txt  vocab.json  special_tokens_map.json  tokenizer_config.json
  model.safetensors          # git-ignored, 603,049,496 bytes
  README.md
```

`pytorch_model.bin` exists upstream and is deliberately not listed (pickle; DIMER does not accept `.bin`).

## Input ceilings and request parameters

`MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`; `MAX_PROMPTS = 16`, `MAX_PROMPT_CHARS = 64` (distinct after normalisation; `MAX_TEXT_TOKENS = 77`, the CLIP context); `MASK_THRESHOLD = 0.5` (caller-owned, in `[0, 1]`); `LOGIT_SIZE = 352` (the decoder's fixed output resolution, documentation only). See `MODEL_CARD.md` for who owns the threshold, what the probabilities are and are not, and the measured CPU timings.

## Adaptation contract

- **Records:** `{id, image, prompt, mask}` — a PIL image (sides within the ceilings), one phrase (normalised like a query) and a boolean height × width mask (or a mask image whose non-zero pixels are the mask) with at least one true pixel; `validate_dataset` checks the structure, `split_dataset` de-duplicates by decoded pixels and `check_split_disjoint` asserts no image is shared. The default sample (`samples.py`) is the first eight parquet row groups of the FoodSeg103 validation shard (`EduardoPacheco/FoodSeg103`, Apache-2.0; food photographs with pixel-wise ingredient masks) read over HTTPS range requests at an immutable Hub revision, each row group refused on any SHA-256 or byte-total mismatch, each image's largest ingredient class becoming the phrase and its pixels the mask (`largest_class`; background and *other ingredients* excluded); `load_byod_dataset` reads a zip or directory of images and mask images plus `masks.csv`.
- **Measures (`metrics.py`):** `segmentation_metrics` — mean IoU (macro), micro IoU (pixel-pooled), Dice, pixel precision and recall at the threshold, with the per-record rows; `empty_baseline` (0 by construction) and `full_baseline` (IoU = the reference's area fraction).
- **Fine-tuning:** `adapt(train, val, *, epochs=8, lr=3e-4, batch_size=8, seed=0, threshold=0.5)` runs the frozen CLIP towers once per record (the image activations at layers 3, 6 and 9 kept in half precision on the host, the phrase embedding) and caches them, then trains the decoder on those activations with the per-pixel binary cross-entropy against the mask resampled to the 352×352 grid, AdamW (no weight decay), gradient clipping at 1.0 and seeded shuffling; the logits equal the full model's exactly. Epoch 0 records the frozen validation rates; the epoch with the highest validation mean IoU (the earliest on ties) is kept; on any exception the frozen decoder is restored.
- **Artifact:** `save_artifact` writes `adapter.safetensors` (about 4.5 MB) + `manifest.json` (`org.valcorza.clipseg-rd64-refined.adapter.v1`: base identity and weight digest, tensor names, file size and SHA-256, the threshold, configuration, history); `from_artifact` re-verifies the base and checks the manifest, digest and exact tensor set before deserialising.
- **Build record (Tesla T4, seed 42 split):** frozen mean IoU @P:FROZEN_MIOU@ / Dice @P:FROZEN_DICE@ on the 140 held-out records (@P:FROZEN_READ@; the full-mask baseline scores @P:FULL_MIOU@), adapted **@P:ADAPTED_MIOU@** / **@P:ADAPTED_DICE@** (epoch @P:BEST_EPOCH@ of 8), reload parity 8/8; the drawn scene after adaptation: @P:DRAWING_AFTER@. One seeded split of one 800-record sample at one threshold; no dispersion estimate.

## Tutorials

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/tutorials/clipseg_segmentation_colab.ipynb)

`tutorials/clipseg_segmentation_colab.ipynb` is declared `E2E` / `GUIDED` under DIMER Notebook Specification 2.0 and is **standalone** (§4): generated by `tools/build_notebook.py`, it carries the three pipeline modules, the model identity, the manifest digests and the runtime pins, so the exported notebook runs without this repository (parity enforced by `tests/test_notebook_parity.py`). Its default `Run all` path stages and verifies the pinned snapshot, fetches the eight pinned FoodSeg103 row groups and turns the 800 images into records split 600 / 60 / 140, segments a drawn scene through the inference contract, measures the frozen model's mean IoU, Dice, pixel precision and recall on the held-out records beside the empty-mask and full-mask baselines, runs `adapt` with validation-mIoU epoch selection, scores the held-out records again, writes six mask panels and re-segments the scene with the adapted model, and exports the adapter and reloads it with verified mask parity. BYOD (your own images, masks and `masks.csv`) is optional and gated off by default. See `tutorials/README.md` for the registry and `docs/release-verification.md` for the release gate.

## Release status

**Candidate.** Static/unit checks — including the standalone generator parity checks (`tools/build_notebook.py --check`, `tests/test_notebook_parity.py`) — do not constitute clean-runtime notebook evidence. The earlier `TASK-INFERENCE` notebook's Kaggle CPU run (2026-09-14) is retained as history and is not evidence for the `E2E` blob; the supported-runtime run of the exact release revision is recorded in `docs/release-verification.md` when it exists.

## Documentation

- `MODEL_CARD.md` — MODEL_CARD_SPEC 1.1 card, provenance digests, input/output contract, measured runtime.
- `docs/WEIGHTS.md` — weight provenance, the licence discrepancy note, the adapter artifacts, the sample corpus and hosting notes.
- `STATUS.md` — release status.

## Licensing

This repository's code is Apache-2.0 (see `LICENSE`). The upstream weights carry the Hub's Apache-2.0 tag; see `docs/WEIGHTS.md` for the recorded discrepancy with the original code repository's statement, and `MODEL_CARD.md`. The FoodSeg103 sample is Apache-2.0 (LARC-CMU-SMU) and is not redistributed.

## AI Assistance Disclosure

This repository’s code and accompanying documentation were developed with generative AI assistance for code development and technical writing under maintainer direction. The maintainer remains responsible for reviewing the implementation, validating results, and making release decisions. AI assistance does not constitute independent verification, provider endorsement, or release approval.
