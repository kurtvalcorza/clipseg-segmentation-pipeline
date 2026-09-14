---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: image-segmentation
base_model: CIDAS/clipseg-rd64-refined
date_published: "2022-11-01"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt` 2022-11-01T14:25:57Z, https://huggingface.co/api/models/CIDAS/clipseg-rd64-refined — the Transformers-format release); the CLIPSeg paper is arXiv:2112.10003 (2021-12) and the pinned revision is the Hub's `main` as of 2026-09-14"
---

# CLIPSeg rd64-refined — Text-Prompted Image Segmentation (Inference)

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CIDAS%2Fclipseg--rd64--refined-ffcc4d?style=flat)](https://huggingface.co/CIDAS/clipseg-rd64-refined)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-timojl%2Fclipseg-181717?style=flat&logo=github&logoColor=white)](https://github.com/timojl/clipseg)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2112.10003-b31b1b.svg)](https://arxiv.org/abs/2112.10003)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://www.apache.org/licenses/LICENSE-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — stage and verify the pinned upstream revision in a fresh runtime, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/tutorials/clipseg_segmentation_colab.ipynb) [`clipseg_segmentation_colab.ipynb`](https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/tutorials/clipseg_segmentation_colab.ipynb)  
  *Six phrases over a scene of coloured shapes drawn in code with the pinned `CIDAS/clipseg-rd64-refined` weights: one mask per phrase under a caller-owned threshold, and `mask_iou` / `miou` against the drawn masks as sanity evidence only — no PhraseCut benchmark.*

---

#### Description

`CIDAS/clipseg-rd64-refined` is the Transformers-format release of the CLIPSeg model with 64-dimensional reduced decoder activations and the "refined" (complex transposed convolution) head, from "Image Segmentation Using Text and Image Prompts" (Lüddecke and Ecker, arXiv:2112.10003), released by the authors' group (CIDAS, University of Göttingen) and pinned here to revision `999e0328d9e10b484360c477313983f9afdd7050` (the Hub's `main` on 2026-09-14). The snapshot `config.json` declares `CLIPSegForImageSegmentation`: a CLIP ViT-B/16 image encoder (12 layers, hidden size 768, 224-pixel positional grid interpolated to the 352×352 input) and CLIP text encoder (12 layers, hidden size 512, 77-token context, vocabulary 49,408), both frozen upstream, plus a small transformer decoder (`reduce_dim` 64, 4 heads, feed-forward width 2048) that reads the image encoder's activations at layers 3, 6 and 9, is conditioned on the text embedding via FiLM at its first layer, and upsamples to a 352×352 logit map with a complex transposed convolution — about 151M parameters in the 603 MB float32 `model.safetensors`. The decoder was trained on PhraseCut (phrase–mask pairs from Visual Genome) with prompt augmentation; the CLIP towers are the public CLIP weights. At inference the processor (`ViTImageProcessor`, `preprocessor_config.json`: 352×352 resize without preserving aspect ratio, ImageNet mean/std) encodes the image once per phrase, the decoder produces one logit map per phrase, and this package applies a sigmoid, resamples the map bilinearly to the input size and thresholds it. Nothing is trained or adapted here. What this repository adds is packaging: `verify_snapshot` and `stage_missing_files` (manifest digest checking and fresh-clone staging), `ClipSegSegmentationPipeline.from_pretrained` (verified local loading with `trust_remote_code=False`), `format_prompts` (phrase normalisation and ceilings), `segment` (input validation, one mask and probability map per phrase at input resolution with area fraction, tight box and maximum probability, backend output checks), `mask_iou`, `mask_bbox`, and the `validate_inputs` and `evaluation_report` stage helpers.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is zero-shot referring segmentation: input one image (`PIL.Image.Image`, any mode, converted to RGB), one to sixteen free-text phrases and a threshold; output, per phrase, a boolean mask and a float32 probability map at input resolution plus the mask's area fraction, tight box and maximum probability. Envisioned applications are interactive masking and cut-outs in image-editing tooling ("select the red mug"), coarse region proposals for annotation tools where a person refines the mask, quick coverage or presence estimates for a named thing across an image collection, and prototyping of language-guided vision systems — with the mask checked by a person before it drives anything. Within DIMER the pipeline is an inference component and a zero-configuration baseline for text-prompted segmentation, not a certified segmenter for any domain.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision developers, annotation-tool builders and researchers integrating text-prompted segmentation into research prototypes, internal tooling, or the DIMER workbench. A user is expected to understand that the per-pixel probability is an *uncalibrated sigmoid* — a pixel at 0.9 is not 90 % likely to belong to the phrase — that the threshold is theirs to choose and trades mask area for precision, that each phrase's mask is independent of the others (masks can overlap and pixels can belong to none; there is no class competition and no instance separation), that the decoder works at 352×352 so boundaries are soft and thin structures may vanish, that the phrase wording matters (CLIP's text tower responds to synonyms, colours and attributes differently), that the model was trained on photographs with PhraseCut-style phrases so drawings, documents, medical and aerial imagery and non-English phrases are distribution shifts, and that segmentation quality can only be measured on reference masks they supply. Users who need instance masks, exhaustive semantic labels, sharp boundaries, or the image-prompt (one-shot) mode are expected to know none of that is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** no instance separation (one mask per phrase however many matching objects), no exhaustive labelling (masks are independent and non-exclusive), no calibrated confidence, no abstention (an absent phrase yields an empty or spurious mask with no signal), no image-prompt conditioning, no boundary refinement beyond bilinear resampling of a 352×352 map, and no batching across images.
2. **Input boundary:** `segment` rejects non-PIL images (`TypeError`), sides below `MIN_IMAGE_SIDE = 16` px or above `MAX_IMAGE_SIDE = 4096` px, a single string instead of a list, empty, non-string, over-long (`MAX_PROMPT_CHARS = 64`) or duplicate-after-normalisation phrases, more than `MAX_PROMPTS = 16` phrases, and thresholds outside `[0, 1]` (`ValueError`/`TypeError`). Every image is resized to 352×352 without preserving aspect ratio, so thin panoramas are squashed and objects smaller than a few of the 22×22 patch cells are unlikely to be segmented; phrases longer than the 77-token CLIP context are truncated by the tokenizer.
3. **Input boundary:** the decoder's training data is PhraseCut — Visual Genome photographs with phrases naming objects, attributes and relations — over frozen CLIP towers trained on web image–text pairs. Flat drawings (the tutorial's shapes), documents, charts, medical, microscopy and satellite imagery, and non-English phrases fall outside what the upstream authors evaluated and what this repository measured; results on them are undefined, not merely degraded. An image with nothing matching the phrase still produces a probability map (see §Risks and harms).
4. **Decision boundary:** not for autonomous decisions that act on masks — automated retouching or redaction shipped without review, measurement of areas or counts for safety, clinical, agricultural or industrial decisions, content moderation, surveillance masking — without a human checking the mask, and a locally measured mean IoU on the deployment's own labelled masks at the chosen threshold.

#### Factors

###### Groups

This pipeline is human-centric whenever a phrase names a person or a personal attribute: `a person`, `a woman`, `a child`, `a man in a turban`, `a wheelchair user`, or any phrase naming skin, hair, dress, age or disability makes the model mask people by that description, and the CLIP text tower is documented in the CLIP literature to encode social biases in how such phrases match images. Neither the upstream authors nor this repository audited mask quality or false positives by depicted group, and the tutorial contains no people at all. Non-human groups whose accuracy is unknown, not known to be equal: non-photographic images (the tutorial's drawn shapes, where the model nonetheless scored IoU 0.91–0.96), non-Western objects, food, clothing and scenes, low-light and low-resolution captures, thin or small objects, and phrases in languages other than English. An operator whose images contain people is responsible for a fairness audit on their own image set, stratified by depicted group and phrase type, before relying on the output — and for deciding which phrases about people are permitted at all (see §Use cases).

###### Instrumentation

The upstream "instruments" are web photography as collected for CLIP's pretraining and the Visual Genome photographs behind PhraseCut, annotated by crowd workers with polygons. Inference images arrive from whatever produced them — a phone camera, a scanner, a render, a drawing library — and resolution, exposure, colour balance, compression and aspect ratio all change the visual evidence; the fixed 352×352 resize discards aspect ratio and detail regardless of the source, so a 4096×4096 input carries no more information than a 352×352 one, and the resampled mask inherits the 352-pixel grid's softness. The pipeline validates type, size and phrase shape only; it cannot detect an unusual capture, a non-photographic image, or a phrase that names nothing present. The synthetic tutorial scene (flat Pillow shapes, no texture or lighting) is itself a rendering instrument unlike any PhraseCut photograph, which is why its IoUs are sanity evidence rather than a measurement.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, float32 on CPU; CUDA is used automatically when visible (float32) but was not exercised for this card. Measured on the reference machine with the GPU hidden (`CUDA_VISIBLE_DEVICES=-1`) and the Hub offline (`HF_HUB_OFFLINE=1`): `verify_snapshot` on the 8-file, 603 MB snapshot 0.32 s; load 4.95 s; six phrases on one 640×480 drawn scene 0.82 s, one phrase 0.15 s; one phrase on a 4096×4096 blank image 0.42 s — cost is one image encoding plus one decoder pass per phrase, roughly independent of the input resolution after the resize. Data environment: the model assumes a photograph and a phrase naming something visible in it; the synthetic tutorial scene violates the first assumption on purpose (flat shapes) and the model still found every shape, an observation about simple high-contrast regions, not a property. Cluttered scenes, thin structures, non-English phrases and absent phrases are where this repository did not measure, and the pipeline reports no signal when they occur.

#### Metrics

###### Performance Measures

The pipeline reports no accuracy measure. The per-pixel probability is an uncalibrated sigmoid and the mask is its cut at the caller's threshold; `area_fraction`, `max_probability` and `bbox` describe the mask, not its quality. The repository ships the field's own building block because it is what a caller would use to evaluate: `mask_iou(a, b)` — intersection-over-union of two boolean masks — and the public `evaluation_report(result, reference_masks=None)` stage returns, when reference masks are supplied, one `mask_iou` entry per phrase (with the reference and predicted area fractions) and their mean `miou` with the verdict `sample-sanity`, or the verdict `not-measurable` naming the labelled set that would be required when they are not. Mean IoU over a labelled set at a stated threshold is the conventional measure (PhraseCut reports mIoU and IoU@0.5 on its test split); it needs masks and a matching phrase vocabulary from the deployment domain that the caller must supply, and neither PhraseCut nor Pascal-5i is bundled. The upstream paper's PhraseCut mIoU for this variant (upstream-reported; not restated here because the hosted card gives no number) is not reproduced or claimed by this pipeline.

###### Decision thresholds

The model has no decision of its own: it emits a logit per pixel per phrase. The decision parameter is the **threshold** `threshold` on the sigmoid, default `MASK_THRESHOLD = 0.5` (the natural cut of a sigmoid, chosen by this repository; not a calibration) in `[0, 1]`, exposed on `segment` and `validate_inputs` and recorded in every result and report. On the tutorial scene the smoke run recorded IoU 0.81–0.96 at 0.3, 0.91–0.96 at 0.5 and 0.83–0.90 at 0.7: a lower threshold grows every mask, a higher one shrinks it, and the right value depends on whether the deployment prefers over- or under-segmentation — which it must decide on its own labelled masks. `format_prompts` applies one fixed normalisation (whitespace collapsed, lower-cased, trailing full stop removed) so the same phrase always produces the same query. A deployment owns the threshold and the phrase vocabulary, and decides how a mask is verified before it is used.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings and the masks of six phrases on one drawn scene, not accuracy. Run-to-run variability comes only from floating-point kernel selection across CPU builds and accelerators; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable, but CPU and CUDA probability maps can differ in the low decimals and a pixel near the threshold can flip; the drawn scene uses no text rendering, so its bytes do not depend on the Pillow build. On the drawn scene the model scored `mask_iou` 0.964 (red circle), 0.960 (blue square), 0.908 (yellow triangle) and 0.958 (green grass) at 0.5 — `miou` 0.947 — and left `a cat` and `the sky` empty (maximum probability 0.00 and 0.01); on a blank white image and on uniform noise `a red circle` and `a cat` were empty too (maximum ≤ 0.02): four observations on flat shapes plus four empty-mask observations, not an estimate. A caller who needs an accuracy estimate must supply labelled masks and compute mean IoU over many images or bootstrap resamples themselves at their chosen threshold; a caller who needs a confidence per mask has none from this model.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The upstream paper describes training the decoder on PhraseCut (about 340k phrase–region pairs on Visual Genome photographs, which are Flickr images that include real, identifiable people) with the CLIP towers frozen at OpenAI's public weights, themselves trained on 400M web image–text pairs whose collection was not published and which the CLIP authors describe as skewed towards people and societies most connected to the internet; personal data in both corpora is therefore present by construction. Neither was audited here. This repository distributes code, tests, and documentation; it does not distribute the 603,049,496-byte `model.safetensors`, which is staged locally under `weights/clipseg-rd64-refined/` and git-ignored, and it ships no photographs — the tutorial scene is drawn in code. **Licence note:** the Hub checkpoint carries an Apache-2.0 tag, while the original code repository's MIT `LICENSE` and README state that the MIT licence does not apply to the weights without naming another; this repository follows the Hub tag for the artifact it redistributes and records the discrepancy (see `docs/WEIGHTS.md`). The operator must audit the images they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check and will mask `a face` as readily as `a red circle`.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — masking people by appearance for surveillance or redaction, measuring lesions, crops, damage or defects from a mask, driving robots or vehicles from a segmented region, automated moderation or blurring — would be admissible only with human review of every mask (the model yields a map for any phrase on any image and gives no signal), a locally measured mean IoU on the deployment's own labelled masks at the chosen threshold, stratified by depicted group where people are involved, a documented threshold and phrase policy, an explicit list of phrases about people that are refused before they reach the model, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `stage_missing_files` refuses a manifest whose `modelId`/`revision` differ from the package constants and fetches only manifest-listed files at that revision when `allow_download=True`; `verify_snapshot` then checks all 8 listed files' byte sizes and SHA-256 before any load; `from_pretrained` loads only from the verified directory with `local_files_only=True`, always passes `trust_remote_code=False`, and the smoke run loaded and segmented with `HF_HUB_OFFLINE=1`. The upstream `pytorch_model.bin` (pickle) is neither listed nor loaded. A test flips one hex digit of a manifest digest and asserts the loader refuses; another asserts a foreign manifest is refused; the import-boundary tests assert that a missing or tampered snapshot is refused before `torch` or `transformers` is imported.
- **Input integrity:** the public `validate_inputs(image, prompts, *, threshold)` stage applies exactly the checks `segment` applies (both route through one shared private checker) and returns an input manifest recording the schema, the ceilings, the observed input, the normalised phrases, the threshold and the verdict; `validate_image` rejects non-PIL inputs and sides outside 16–4096 px; `format_prompts` rejects non-list, empty, non-string, over-long, too many or duplicate phrases; thresholds outside `[0, 1]` are rejected; `segment` raises when the backend returns maps of the wrong shape or values outside `[0, 1]`; `evaluation_report` rejects a reference phrase that was not segmented.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; deterministic thresholding; fixed phrase normalisation; every result carries `model_id`, `model_revision`, the normalised queries, the threshold and the image size.
- **Refusals:** no batching across images, no download without the explicit flag, no Hub access at inference time, no pickle deserialisation, no image-prompt mode, no attempt to guess whether a phrase names anything present, and no filtering of phrase content — that policy is the operator's to implement around the pipeline.
- No statistical mitigation (class balancing, subsampling) applies: no training happens in this repository.

###### Risks and harms

- **Spurious masks:** the model has no abstention — every phrase yields a probability map, and while absent phrases stayed empty in the smoke run, a phrase that partially matches something (a colour, a shape, a texture) can produce a confident-looking region with no signal; downstream consumers that trust a mask (editors, redaction, measurement) inherit the error silently.
- **Threshold dependence:** the same phrase yields larger or smaller masks as the threshold moves (IoU 0.81–0.96 across 0.3–0.7 on the tutorial scene); a deployment that never tunes it on labelled masks ships an arbitrary trade-off.
- **Masking people by description:** phrases naming appearance, dress, ethnicity or disability mask people by those descriptions with CLIP's documented biases; such use can cause direct harm and is prohibited below.
- **Soft, coarse boundaries:** the 352×352 decoder blurs edges and drops thin or small structures; a mask used for measurement or cut-outs carries that error.
- **Automation bias:** a clean-looking mask over a drawn shape invites trust that an uncalibrated sigmoid has not earned.
- **Bias amplification:** any image population or phrase vocabulary PhraseCut and CLIP under-represent (non-Western scenes, non-English phrases, specialised domains) is reproduced as uneven mask quality, undetected because no per-group evaluation exists.
- **Resource use:** a 603 MB model and ~0.15 s per phrase on the reference CPU; sixteen phrases on one image scale linearly, and the CUDA path was not measured.

###### Use cases

Prohibited even where the model would work: segmenting images of people by phrases that name or infer protected characteristics (race, ethnicity, religion, health, disability, sexual orientation), or to identify, track, profile or surveil individuals; processing images the operator has no right to process, including intimate imagery and licence-restricted material; presenting masks as verified measurements, evidence or ground truth without human review; deceptive editing that presents automatically masked or removed content as authentic; and any use that violates the upstream weight licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the images processed. Autonomous high-consequence actions triggered by unreviewed masks are prohibited by the intended-use contract above.

## Immutable provenance

- Model: `CIDAS/clipseg-rd64-refined`
- Revision: `999e0328d9e10b484360c477313983f9afdd7050`
- Snapshot manifest: `weights/clipseg-rd64-refined/dimer-base-manifest.json`, 8 files, `totalBytes` 604641231
- `model.safetensors` SHA-256: `d00ca85d6b859f9d07b7cfb8ef26fe9771cb275b34c9368f2ecf603139307f55` (603,049,496 bytes, float32)
- `config.json` SHA-256: `c023375966d31b3b1392764f7bd91df47098ce19f62f11b0263d8eedcf708bcd` (4,732 bytes; `CLIPSegForImageSegmentation`, `reduce_dim` 64, `extract_layers` [3, 6, 9], `use_complex_transposed_convolution` true)
- `preprocessor_config.json` SHA-256: `4fb09ebcfd7651205ca8299b993c30088e5535ef350a72d46c5c4580eeac0440` (380 bytes; `ViTImageProcessor`, 352×352, ImageNet mean/std)
- Weight format: SafeTensors; loader `CLIPSegForImageSegmentation.from_pretrained(<dir>, local_files_only=True, trust_remote_code=False, dtype=float32)` with `CLIPSegProcessor` from the same directory (`text=<phrases>, images=[<image>] * n, padding=True`). The upstream `pytorch_model.bin` is not part of the manifest and is never loaded.

## Input/output contract

- `ClipSegSegmentationPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — stages missing manifest files (only with `allow_download=True`), verifies digests, loads; `device` defaults to `cuda:0` when visible, else `cpu`; float32 on both.
- `segment(image, prompts, *, threshold=0.5) -> dict` with keys `segments` (one per phrase: `prompt`, `mask` bool `(H, W)`, `probability` float32 `(H, W)`, `area_fraction`, `max_probability`, `bbox` xyxy or `None`), `queries` (normalised), `threshold`, `width`, `height`, `model_id`, `model_revision`.
- `format_prompts(prompts) -> list[str]`; `mask_iou(a, b) -> float`; `mask_bbox(mask) -> list[int] | None`.
- Ceilings and constants: `MIN_IMAGE_SIDE = 16`, `MAX_IMAGE_SIDE = 4096`, `MAX_PROMPTS = 16`, `MAX_PROMPT_CHARS = 64`, `MAX_TEXT_TOKENS = 77`, `MASK_THRESHOLD = 0.5`, `LOGIT_SIZE = 352`, `INPUT_SCHEMA`.
- `validate_inputs(image, prompts, *, threshold, names) -> dict`; `evaluation_report(result, reference_masks=None, *, sample_kind) -> dict` where `reference_masks` maps phrases to boolean masks at input resolution; `verify_snapshot(path=None) -> dict`; `stage_missing_files(path=None, *, allow_download=False, downloader=None) -> list[str]`.

## Runtime

- Pins: `torch==2.14.0`, `torchvision==0.29.0`, `torchaudio==2.11.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`, `huggingface-hub==0.36.2`; Python 3.12.
- Precision: float32; preprocessing resizes the image to 352×352 (aspect ratio not preserved, ImageNet mean/std; `ViTImageProcessor`, snapshot defaults) and tokenises each phrase with the CLIP BPE tokenizer; the 352×352 logits are passed through a sigmoid and resampled bilinearly (`align_corners=False`) to the input size; the mask is `probability >= threshold`.
- Measured 2026-09-14 in the Windows venv (`torch 2.14.0+cu130`) with `CUDA_VISIBLE_DEVICES=-1` and `HF_HUB_OFFLINE=1`, device `cpu`: `verify_snapshot` 0.32 s (8 files, 603 MB); load 4.95 s; `segment` on a synthetic 640×480 scene (red circle, blue square, yellow triangle, green ground band on off-white, drawn with Pillow) with six phrases at `threshold=0.5` in 0.82 s: `a red circle` area 0.082, max 0.90, IoU 0.964; `a blue square` 0.151, 0.96, 0.960; `a yellow triangle` 0.053, 0.95, 0.908; `green grass` 0.324, 0.85, 0.958; `a cat` 0.000, 0.00; `the sky` 0.000, 0.01 — `miou` 0.947, verdict `sample-sanity`; at 0.3 the four IoUs were 0.924 / 0.931 / 0.810 / 0.956 and at 0.7 0.871 / 0.903 / 0.852 / 0.833; one phrase 0.15 s; blank 640×480 white → `a red circle` empty (max 0.02), `a cat` empty (0.00); 4096×4096 blank → empty in 0.42 s; uniform noise → both empty (max ≤ 0.02).
- Tutorial execution: `tutorials/clipseg_segmentation_colab.ipynb` ran top-to-bottom in a fresh local kernel (all 8 code cells, 65.4 s including the 603 MB staging, same four IoUs and `miou` 0.947 as the smoke run, both absent phrases empty); recorded in `docs/release-verification.md` as pre-flight, not supported-runtime evidence.
- Tests: `pytest -q -o addopts= tests` — offline, no weights required; `ruff check src tests tools` clean.
- Not executed: CUDA path, photographs (only drawn shapes, blank images and noise), the image-prompt mode, any mean-IoU measurement against labelled masks, phrases about people, non-English phrases.

## References

- Lüddecke, Ecker. Image Segmentation Using Text and Image Prompts. CVPR 2022. https://arxiv.org/abs/2112.10003
- Wu et al. PhraseCut: Language-based Image Segmentation in the Wild. CVPR 2020. https://arxiv.org/abs/2008.01187
- Radford et al. Learning Transferable Visual Models From Natural Language Supervision (CLIP). ICML 2021. https://arxiv.org/abs/2103.00020
- Upstream code: https://github.com/timojl/clipseg
- Upstream card: https://huggingface.co/CIDAS/clipseg-rd64-refined
- Transformers `CLIPSeg` documentation: https://huggingface.co/docs/transformers/model_doc/clipseg
