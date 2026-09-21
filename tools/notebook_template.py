"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E segmentation-adaptation workflow: the pinned CIDAS/clipseg-rd64-refined snapshot is
digest-verified and loaded, 800 Apache-2.0 FoodSeg103 food photographs with pixel-wise ingredient masks are fetched
as eight digest-pinned parquet row groups over HTTPS range requests and turned into (image, phrase, mask) records,
the records are validated and split by image, a synthetic scene of drawn shapes is segmented through the inference
contract, the frozen model is scored over the held-out records (mean IoU, micro IoU, Dice, pixel precision and
recall at the threshold) beside an empty-mask and a full-mask baseline, a bounded fine-tuning of the CLIPSeg decoder
runs on cached CLIP activations with validation-mIoU epoch selection, the held-out split is scored again, six
held-out records and the drawn scene are re-run with the adapted model, and the adapter is exported and reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "clipseg_segmentation_pipeline",
    "repo_name": "clipseg-segmentation-pipeline",
    "stem": "clipseg_segmentation",
    "notebook_name": "clipseg_segmentation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "pipeline_class": "ClipSegSegmentationPipeline",
    "weights_key": "clipseg-rd64-refined",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "runtime_imports": ["torch", "transformers", "PIL"],
    "title": "CLIPSeg rd64-refined — DIMER E2E text-prompted segmentation adaptation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/clipseg-segmentation-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/tutorials/clipseg_segmentation_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-CIDAS%2Fclipseg--rd64--refined-ffcc4d?style=flat",
            "https://huggingface.co/CIDAS/clipseg-rd64-refined",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-timojl%2Fclipseg-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/timojl/clipseg",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2112.10003-b31b1b.svg", "https://arxiv.org/abs/2112.10003"),
    ],
    "capability": "zero-shot (text-prompted) image segmentation — one image plus 1–16 free-text phrases → one binary mask and probability map per phrase — and bounded supervised fine-tuning of the CLIPSeg decoder on labelled (image, phrase, mask) records, using the pinned `CIDAS/clipseg-rd64-refined` weights",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned `CIDAS/clipseg-rd64-refined` snapshot (a 603 MB `model.safetensors`; no pickle is opened anywhere), fetches "
        "the first eight row groups of the FoodSeg103 validation shard from the Hugging Face Hub at an immutable revision "
        "with HTTPS range requests (about 43 MB; each row group refused on any SHA-256 or byte-total mismatch), turns each "
        "of the 800 images into one (image, phrase, mask) record and splits them by image into 600 / 60 / 140, segments a "
        "synthetic scene of drawn shapes through the inference contract with an input manifest and a rejection probe, "
        "scores the frozen model over the 140 held-out records (mean IoU, micro IoU, Dice, pixel precision and recall at "
        "the threshold) beside an empty-mask and a full-mask baseline, runs a bounded fine-tuning of the CLIPSeg decoder on "
        "cached CLIP activations with validation-mIoU epoch selection, scores the held-out records again, re-runs six "
        "held-out records and the drawn scene with the adapted model, exports the adapter as safetensors with a manifest, "
        "and reloads that artifact into a fresh pipeline to verify mask parity. The default path needs no repository clone, "
        "no DIMER worker or service, no credential, no upload dialog and no configuration edit (NOTEBOOK_SPEC 2.0 §5). On a "
        "Tesla T4 the default path took about 2 minutes of cell time (eight epochs 59 s, frozen "
        "scoring of 140 records 7 s); a CUDA runtime is used automatically when present, and the path is "
        "practical on CPU too (the build venv cached the 660 training and validation records in about "
        "two to three minutes and trained the decoder in seconds per epoch)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to upload one zip "
        "of images and mask images plus a `masks.csv` (`file`, `mask`, `prompt`, optional `id`; one row per image, at least "
        "eight images; a mask image's non-zero pixels are the mask). The records pass through the same validation, "
        "image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells as the "
        "FoodSeg103 sample. Uploaded files stay inside this runtime. BYOD is optional and never part of the default path."
    ),
    "intro": (
        "CLIPSeg is a frozen CLIP ViT-B/16 image encoder and CLIP text encoder joined by a small transformer decoder: the "
        "decoder reads the image tower's activations at layers 3, 6 and 9, conditions them on the phrase's CLIP embedding "
        "through FiLM, and produces one 352 × 352 logit map per phrase; the carried module passes the logits through a "
        "sigmoid, resamples the probability map to the input size and thresholds it into a mask under a caller-owned "
        "`threshold` (150,747,746 parameters in all, published under the **Apache-2.0** licence). The output is **an "
        "uncalibrated per-pixel sigmoid**: the same map can be a tight or a generous mask depending on the threshold, and "
        "**the threshold is a caller-owned request parameter**.\n\n"
        "What this notebook adds to inference is **adaptation of the decoder on labelled (image, phrase, mask) records**. "
        "The records are food photographs from FoodSeg103, each paired with the name of the ingredient that covers the most "
        "pixels (`bread`, `chicken duck`, `steak`, `pie`, …) and that ingredient's pixel mask — a phrase vocabulary and a "
        "boundary convention far from the PhraseCut phrases the decoder was trained on, and on them the frozen model "
        "already finds the right region roughly: a mean IoU of **0.637** on the 140 held-out records in the "
        "build record (the full-mask baseline scores 0.283). So the honest question is narrow: does a bounded "
        "fine-tuning of the 1,127,009-parameter decoder on 600 records — the CLIP towers frozen, exactly as the upstream "
        "authors trained it — move the held-out **mean IoU**, **Dice**, **pixel precision** and **pixel recall** on an "
        "image-disjoint test split past the frozen model and two **non-adapted baselines**, and what does it do to the "
        "drawn shapes the same decoder segments? Nothing here is a claim about your images or your phrases: it is one "
        "seeded split of one small labelled set.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (an 8-file manifest with the tokenizer and processor "
        "files) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files before the "
        "processor or the model is constructed. The pipeline runs in **float32 on every device**: the adapter is trained in "
        "float32 and overlays without a cast, and CPU, Tesla-class and consumer GPUs then run the same arithmetic."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried package guarantees; stage and digest-verify the immutable "
        "upstream snapshot; fetch a digest-pinned labelled mask set, turn it into (image, phrase, mask) records, validate "
        "it and split it by image without leakage; segment a drawn scene through the public API and read the output "
        "contract correctly (an uncalibrated sigmoid, a caller-owned threshold, a `sample-sanity` report only against masks "
        "you drew yourself); measure the frozen model's held-out mean IoU, Dice, pixel precision and recall beside two "
        "non-adapted baselines; run a bounded fine-tuning of the decoder with the per-pixel binary cross-entropy, explicit "
        "hyperparameters and validation-based epoch selection; evaluate on an image-disjoint test split; look at the adapted "
        "masks next to the frozen ones and the references, and at what the drawn scene does after the shared decoder was "
        "tuned; and export a safetensors adapter that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "instance or panoptic segmentation (one binary mask per phrase; masks of different phrases are independent), "
        "phrase vocabularies beyond one phrase per record in the adaptation sample, threshold tuning (the threshold is "
        "fixed at `MASK_THRESHOLD` for every measurement here), fine-tuning of the CLIP image or text towers, evaluation on "
        "PhraseCut or a segmentation benchmark proper (only one seeded 800-record sample is scored here), and any claim "
        "that ingredient masks stand in for your images. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Kaggle, Python 3.12; CPU or CUDA). The default path uses CUDA automatically when present. The CLIP towers run `EVAL_BATCH_SIZE` records per forward and the build record measured 7 s to score 140 records and 59 s for the eight epochs (caching the tower activations for 600 + 60 records took 30 s) on a Tesla T4, about 2 minutes of cell time for the whole path including the pinned install and the downloads. The pinned `torch==2.14.0` install and the 603 MB checkpoint are the large downloads of the run; the row groups are about 43 MB. The activation cache holds about 1.3 GB of half-precision tensors on the host for 600 records.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a per-pixel sigmoid is and why thresholding it is a decision the caller owns; what intersection-over-union and Dice measure and why 140 records from one draw give no dispersion; why a self-drawn scene is a plumbing check while a held-out split of one labelled set is a measurement of that set only.",
        "- **Data contract:** records are `{id, image, prompt, mask}` — `image` a PIL image (or a file decodable by Pillow) with sides within 16..4,096 px, `prompt` one phrase of at most 64 characters (normalised like a query), `mask` a boolean height × width array or a mask image whose non-zero pixels are the mask, with at least one true pixel. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; a dataset needs 8..5,000 records; splitting de-duplicates by decoded pixels so no image lands in two splits. BYOD accepts one zip (or directory) of images and mask images plus a `masks.csv` in the layout named above.",
        "- **Validation is structural, not semantic:** every image and mask is decoded and every phrase checked, but nothing checks that a mask outlines what its phrase names — a mislabelled set is fine-tuned on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there. The default path uploads nothing.",
        "- **External access (data):** besides the model snapshot, the default path reads eight row groups of `default/validation/0000.parquet` from `https://huggingface.co/datasets/EduardoPacheco/FoodSeg103/resolve/<revision>/` at the immutable parquet-conversion revision `176acc3e…` with HTTPS range requests (the parquet footer plus about 43 MB of row-group bytes out of a 115 MB shard), each row group pinned by SHA-256 and byte total in the carried `samples.py` and refused on any mismatch. FoodSeg103 is published under the Apache-2.0 licence (LARC-CMU-SMU; Wu et al. 2021); nothing is redistributed by this repository.",
    ],
    "cells": [
        {
            "md": (
                "## 4. FoodSeg103 records, the phrases and the split\n\n"
                "`fetch_corpus` returns the eight pinned row groups from the cache under `weights/foodseg103/` or the Hub at the "
                "pinned parquet-conversion revision — `pyarrow` reads the shard's footer and exactly those row groups over "
                "HTTPS range requests; every cached file is re-hashed and every fetched row group refused on any SHA-256 or "
                "byte-total mismatch — and `read_corpus` turns each row into a record: the photograph, the ingredient class "
                "that covers the most pixels as the phrase (`largest_class`; background and *other ingredients* never "
                "qualify) and that class's pixels as the mask. `build_sample_dataset` draws a seeded image-level split "
                "(600 / 60 / 140). `validate_dataset` then checks every record against the contract, `check_split_disjoint` "
                "asserts no image (by decoded-pixel digest) is shared, and the training split's summary table is written to "
                "`outputs/{stem}_train.csv`.\n\n"
                "Look for: 800 records over some sixty phrases with masks covering about a quarter of their images, three "
                "digests, and four refusal probes — a duplicate id, an empty mask, a mask of the wrong shape, and a dataset "
                "too small to use — each rejected before the model does anything."
            ),
            "code": (
                "import hashlib\n"
                "import json\n"
                "import time\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_zip = Path('work') / 'byod.zip'\n"
                "    byod_zip.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_zip.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_zip)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records)}}\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_groups = fetch_corpus(cache_dir='weights/foodseg103')\n"
                "    corpus = read_corpus(corpus_groups)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} @ {{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'row_groups': len(corpus_groups), 'images': sum(len(v) for v in corpus_groups.values()), 'records': len(corpus), 'bytes': sum(len(r['image']) + len(r['label']) for v in corpus_groups.values() for r in v), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'prompts': manifest['n_prompts'], 'mask_area': {{k: round(v, 3) for k, v in manifest['mask_area_fraction'].items()}}, 'width': manifest['image_width'], 'height': manifest['image_height'], 'digest': manifest['digest'][:16] + '...'}}}})\n\n\n"
                "def overlay(image, mask, colour=(220, 40, 40)):\n"
                "    base = np.asarray(image.convert('RGB'), dtype=np.float32)\n"
                "    out = base.copy()\n"
                "    out[mask] = 0.45 * out[mask] + 0.55 * np.array(colour, dtype=np.float32)\n"
                "    return Image.fromarray(out.round().astype(np.uint8))\n\n\n"
                "example = train_records[0]\n"
                "overlay(example['image'], example['mask']).save('outputs/{stem}_example_record.png')\n"
                "print({{'example': {{'id': example['id'], 'image': list(example['image'].size), 'prompt': example['prompt'], 'mask_area_fraction': round(float(example['mask'].mean()), 3)}}}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:8]],\n"
                "    'empty mask': [{{**train_records[0], 'mask': np.zeros_like(train_records[0]['mask'])}}, *train_records[1:8]],\n"
                "    'mask of the wrong shape': [{{**train_records[0], 'mask': np.ones((8, 8), dtype=bool)}}, *train_records[1:8]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Segment a drawn scene through the inference contract\n\n"
                "The inference contract is exercised as the inference-only tutorial exercised it: a 640 × 480 scene drawn in "
                "code — a red circle, a blue square, a yellow triangle and a green ground band — with the exact masks the "
                "shapes were drawn from and two absent phrases (`a cat`, `the sky`) on purpose; a different image family "
                "from the food photographs, and a scene the adapted model will segment again in Section 9. `validate_inputs` "
                "applies exactly the checks `segment` applies (one image with sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE`, "
                "1..`MAX_PROMPTS` distinct phrases of at most `MAX_PROMPT_CHARS` characters, a threshold in [0, 1]) and "
                "returns an input manifest; a duplicate-phrase request is validated too and its rejection recorded as a "
                "finding. `segment` returns one mask, probability map, area fraction, tight box and maximum probability per "
                "phrase — **the probabilities are an uncalibrated sigmoid** and the threshold is a **caller-owned request "
                "parameter**; an absent phrase still yields a map. `evaluation_report` with the drawn masks is "
                "`sample-sanity`: one `mask_iou` per drawn phrase and their mean, plumbing evidence for one drawing — a "
                "segmentation benchmark needs labelled masks, which Section 6 supplies. The inference-only card recorded a "
                "mean IoU of 0.86 on the four drawn shapes."
            ),
            "code": (
                "def synthetic_scene(width=640, height=480):\n"
                "    \"\"\"Coloured shapes drawn with Pillow (no text); returns image + {{phrase: boolean reference mask}}.\"\"\"\n"
                "    image = Image.new('RGB', (width, height), (245, 245, 240))\n"
                "    d = ImageDraw.Draw(image)\n"
                "    shapes = [\n"
                "        ('green grass', 'rectangle', [0, 320, 640, 480], (60, 179, 75)),\n"
                "        ('a red circle', 'ellipse', [80, 80, 260, 260], (220, 40, 40)),\n"
                "        ('a blue square', 'rectangle', [340, 90, 560, 300], (40, 70, 200)),\n"
                "        ('a yellow triangle', 'polygon', [(200, 460), (320, 330), (440, 460)], (250, 200, 30)),\n"
                "    ]\n"
                "    masks = {{}}\n"
                "    for phrase, kind, geometry, colour in shapes:\n"
                "        getattr(d, kind)(geometry, fill=colour)\n"
                "        reference = Image.new('1', image.size)\n"
                "        getattr(ImageDraw.Draw(reference), kind)(geometry, fill=1)\n"
                "        masks[phrase] = np.array(reference, dtype=bool)\n"
                "    return image, masks\n\n\n"
                "scene, scene_masks = synthetic_scene()\n"
                "scene_prompts = list(scene_masks) + ['a cat', 'the sky']  # two absent phrases on purpose\n"
                "scene_name = 'synthetic_shapes_640x480'\n"
                "scene_sha256 = hashlib.sha256(np.asarray(scene).tobytes()).hexdigest()\n"
                "THRESHOLD = MASK_THRESHOLD\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'LOGIT_SIZE': LOGIT_SIZE, 'MAX_PROMPTS': MAX_PROMPTS, 'MAX_PROMPT_CHARS': MAX_PROMPT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MASK_THRESHOLD': MASK_THRESHOLD, 'EXTRACT_LAYERS': list(EXTRACT_LAYERS), 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'EVAL_BATCH_SIZE': EVAL_BATCH_SIZE, 'device': pipe.device}}}})\n"
                "input_manifest = validate_inputs(scene, scene_prompts, threshold=THRESHOLD, names=[scene_name])\n"
                "try:\n"
                "    validate_inputs(scene, ['a red circle', 'A red circle.'])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'duplicate-phrase-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print({{'scene': scene_name, 'sha256': scene_sha256[:16] + '...', 'manifest_verdict': input_manifest['verdict'], 'findings': len(input_manifest['findings'])}})\n\n\n"
                "def segment_scene(pipeline, label):\n"
                "    started = time.perf_counter()\n"
                "    result = pipeline.segment(scene, scene_prompts, threshold=THRESHOLD)\n"
                "    seconds = round(time.perf_counter() - started, 3)\n"
                "    checks = {{\n"
                "        'one_segment_per_phrase': [s['prompt'] for s in result['segments']] == result['queries'] and len(result['queries']) == len(scene_prompts),\n"
                "        'probabilities_in_unit_interval': all(0.0 <= s['probability'].min() and s['probability'].max() <= 1.0 for s in result['segments']),\n"
                "        'masks_at_input_resolution': all(s['mask'].shape == (scene.height, scene.width) for s in result['segments']),\n"
                "        'identity_reported': result['model_id'] == MODEL_ID and result['model_revision'] == MODEL_REVISION,\n"
                "    }}\n"
                "    if not all(checks.values()):\n"
                "        raise RuntimeError(f'segment output failed a sanity check: {{checks}}')\n"
                "    report = evaluation_report(result, scene_masks, sample_kind='synthetic (drawn in this notebook)')\n"
                "    summary = {{s['prompt']: {{'area_fraction': round(s['area_fraction'], 3), 'max_probability': round(s['max_probability'], 3), 'bbox': s['bbox']}} for s in result['segments']}}\n"
                "    ious = {{m['reference']: round(m['value'], 3) for m in report['metrics'] if m['id'] == 'mask_iou'}}\n"
                "    miou = next((m['value'] for m in report['metrics'] if m['id'] == 'miou'), None)\n"
                "    with open(f'outputs/{stem}_scene_{{label}}.json', 'w', encoding='utf-8') as handle:\n"
                "        json.dump({{'segments': summary, 'report': report}}, handle, indent=2, ensure_ascii=False)\n"
                "    palette = [(220, 40, 40), (40, 70, 200), (250, 200, 30), (60, 179, 75), (160, 60, 200), (0, 170, 170)]\n"
                "    sheet = np.asarray(scene, dtype=np.float32).copy()\n"
                "    for index, s in enumerate(result['segments']):\n"
                "        sheet[s['mask']] = 0.45 * sheet[s['mask']] + 0.55 * np.array(palette[index % len(palette)], dtype=np.float32)\n"
                "    Image.fromarray(sheet.round().astype(np.uint8)).save(f'outputs/{stem}_scene_{{label}}.png')\n"
                "    print({{label: {{'seconds': seconds, 'checks': checks, 'mask_iou': ious, 'miou': None if miou is None else round(miou, 3), 'absent_phrases': {{p: summary[p]['area_fraction'] for p in ('a cat', 'the sky')}}, 'verdict': report['verdict']}}}})\n"
                "    return summary, seconds, checks, report\n\n\n"
                "frozen_scene, frozen_scene_seconds, frozen_scene_checks, frozen_scene_report = segment_scene(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "## 6. Baselines and the frozen model on the test records\n\n"
                "Two non-adapted baselines frame the adaptation, each scored by `segmentation_metrics` (carried in "
                "`metrics.py`): the **mean IoU** (the per-record intersection-over-union of the thresholded mask and the "
                "reference, averaged — the measure the epoch is selected on), the **micro IoU** (intersection over union "
                "pooled over every pixel of the split, so large masks weigh more), **Dice**, and **pixel precision** and "
                "**pixel recall** pooled over the split. The **empty-mask** baseline predicts no pixel and scores 0 by "
                "construction — the floor any segmenter must beat to do better than silence. The **full-mask** baseline "
                "predicts every pixel: its IoU is the reference's area fraction, what \"the ingredient is somewhere in the "
                "photograph\" buys without looking. The **frozen model** is scored by `pipe.evaluate`, which segments every "
                "record's phrase on its image in batches of `EVAL_BATCH_SIZE`, thresholds the sigmoid at `THRESHOLD` and "
                "scores the mask. Expect the frozen model **well above both baselines** — it is a phrase-conditioned "
                "segmenter and these are nameable things: the build record measured **0.637** mean IoU on the 140 "
                "held-out records (the frozen decoder already finds most dishes — 63 of the 140 held-out records score above 0.8 IoU — but misses 24 almost entirely (IoU under 0.2), typically ingredient names it does not ground at all (`pie`, `lamb`, `garlic`, `shellfish` at 0.0), with precision 0.835 well above recall 0.736: it under-segments); read four records' scores under their phrases."
            ),
            "code": (
                "METRICS = ('miou', 'iou_micro', 'dice', 'pixel_precision', 'pixel_recall')\n\n"
                "baseline_empty = empty_baseline(test_records)\n"
                "baseline_full = full_baseline(test_records)\n"
                "print({{'empty_baseline': {{k: round(baseline_empty[k], 3) for k in METRICS}}, 'n': baseline_empty['n'], 'note': baseline_empty['baseline']}})\n"
                "print({{'full_baseline': {{k: round(baseline_full[k], 3) for k in METRICS}}, 'note': baseline_full['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, threshold=THRESHOLD, batch_size=EVAL_BATCH_SIZE)\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'n': frozen_test['n'], 'predicted_area_fraction': round(frozen_test['predicted_area_fraction'], 3), 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for row in frozen_test['rows'][:4]:\n"
                "    print({{'id': row['id'], 'prompt': row['prompt'], 'iou': round(row['iou'], 3), 'dice': round(row['dice'], 3), 'reference_pixels': row['reference'], 'predicted_pixels': row['predicted']}})"
            ),
        },
        {
            "md": (
                "## 7. Bounded fine-tuning of the decoder\n\n"
                "`pipe.adapt` trains only the CLIPSeg decoder — the three transformer layers over the reduced activations, "
                "the FiLM conditioning, the reduce projections and the transposed convolution: 1,127,009 of 150,747,746 "
                "parameters — while the CLIP image and text towers stay frozen, exactly the split the upstream authors "
                "trained with. The loss is the **per-pixel binary cross-entropy** between the 352 × 352 decoder logits and "
                "the reference mask resampled to that grid — the upstream training objective. Because the towers are "
                "frozen, their outputs — the image activations at layers 3, 6 and 9 and the phrase embedding — are computed "
                "once per record under no gradient and cached in half precision on the host (the **frozen-tower cache**), "
                "and each step runs only the decoder on those cached activations: the logits equal the full model's "
                "exactly, at a fraction of the cost. AdamW without weight decay at a fixed learning rate, gradient clipping "
                "at 1.0, seeded shuffling, no scheduler, no augmentation. Epoch 0 records the frozen model's validation "
                "rates; every epoch is scored on the 60 validation records at `THRESHOLD`, and the epoch with the **highest "
                "validation mean IoU** (the earliest on ties) is kept.\n\n"
                "Watch the validation mean IoU rise from 0.603 to 0.855 (epoch 4 in the build "
                "record) while the loss drops from about 0.183 to 0.060: the adapted decoder's masks overlap the references by 0.84 mean IoU (frozen 0.64), predicting 0.29 of the image area against 0.25 frozen and the references' 0.28."
            ),
            "code": (
                "EPOCHS = 8  # @param {{type:\"integer\"}}\n"
                "LEARNING_RATE = 3e-4  # @param {{type:\"number\"}}\n"
                "BATCH_SIZE = 8  # @param {{type:\"integer\"}}\n\n\n"
                "def report(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'train_loss': None if entry['train_loss'] is None else round(entry['train_loss'], 4)}}\n"
                "    if entry.get('val'):\n"
                "        row.update({{'val_' + k: round(entry['val'][k], 3) for k in METRICS}})\n"
                "    if 'note' in entry:\n"
                "        row['note'] = entry['note']\n"
                "    print(row)\n\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, threshold=THRESHOLD, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'threshold': adapt_result['threshold'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'extract_layers': adapt_result['extract_layers'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'cache_seconds': adapt_result['cache_seconds'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test records were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted model is scored exactly as the frozen model was in Section 6 and the four systems are put side by "
                "side. Read it in this order: **mean IoU** first (the measure the epoch was selected on — the build record "
                "measured 0.637 → **0.842**), then **Dice** (0.715 → 0.904), then "
                "pixel **precision** and **recall** together (0.835 / 0.736 → 0.881 / 0.927 — a gain in one at the cost of the "
                "other is a moved threshold, not a better segmenter), then the predicted area against the reference area. "
                "The cell asserts the adapted mean IoU is at least the frozen one and above the full-mask baseline. One "
                "hundred and forty records from one seeded split give **no dispersion estimate**; the deltas are "
                "sample-sanity evidence that the adaptation contract works, not a benchmark, and a result on one food "
                "dataset's largest-ingredient masks says nothing about other phrases, other images or your data until you "
                "measure them."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records, threshold=THRESHOLD, batch_size=EVAL_BATCH_SIZE)\n"
                "adapted_val = pipe.evaluate(val_records, threshold=THRESHOLD, batch_size=EVAL_BATCH_SIZE)\n"
                "comparison = {{metric: {{'empty': round(baseline_empty[metric], 3), 'full': round(baseline_full[metric], 3), 'frozen': round(frozen_test[metric], 3), 'adapted': round(adapted_test[metric], 3)}} for metric in METRICS}}\n"
                "comparison['delta_vs_frozen'] = {{metric: round(adapted_test[metric] - frozen_test[metric], 3) for metric in METRICS}}\n"
                "comparison['area'] = {{'reference_pixels': adapted_test['reference_pixels'], 'frozen_predicted_pixels': frozen_test['predicted_pixels'], 'adapted_predicted_pixels': adapted_test['predicted_pixels'], 'frozen_predicted_area_fraction': round(frozen_test['predicted_area_fraction'], 3), 'adapted_predicted_area_fraction': round(adapted_test['predicted_area_fraction'], 3)}}\n"
                "for key, row in comparison.items():\n"
                "    print({{key: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'threshold': THRESHOLD,\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'baselines': {{'empty': {{k: v for k, v in baseline_empty.items() if k != 'rows'}}, 'full': {{k: v for k, v in baseline_full.items() if k != 'rows'}}}},\n"
                "    'frozen_test': {{k: v for k, v in frozen_test.items() if k != 'rows'}},\n"
                "    'validation_metrics': {{k: v for k, v in adapted_val.items() if k != 'rows'}},\n"
                "    'test_metrics': {{k: v for k, v in adapted_test.items() if k != 'rows'}},\n"
                "    'per_record': [{{**frozen_row, 'adapted_iou': adapted_row['iou'], 'adapted_dice': adapted_row['dice'], 'adapted_predicted': adapted_row['predicted']}} for frozen_row, adapted_row in zip(frozen_test['rows'], adapted_test['rows'], strict=True)],\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['miou'] >= frozen_test['miou']\n"
                "assert adapted_test['miou'] > baseline_full['miou']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_both_baselines': adapted_test['miou'] > max(baseline_empty['miou'], baseline_full['miou'])}})"
            ),
        },
        {
            "md": (
                "## 9. Look at the masks, segment the scene again, export the adapter and reload it\n\n"
                "Six held-out records are written as panels (`outputs/{stem}_examples/`: the photograph with the reference "
                "mask, the frozen mask and the adapted mask side by side, the phrase and both IoUs beneath) so the numbers "
                "can be checked by eye. The drawn scene from Section 5 is then segmented again by the adapted model — the "
                "decoder that was tuned serves every phrase, so this is a small look at what the adaptation did *outside* "
                "its phrase vocabulary and its corpus: the build record measured before adaptation `green grass` 0.96, `a red circle` 0.96, `a blue square` 0.96, `a yellow triangle` 0.91 (mean IoU 0.95); absent phrases' area fraction `a cat` 0.000, `the sky` 0.000; after adaptation `green grass` 0.97, `a red circle` 0.97, `a blue square` 0.96, `a yellow triangle` 0.93 (mean IoU 0.96); absent phrases' area fraction `a cat` 0.000, `the sky` 0.000 — one drawing of evidence, "
                "not a measurement.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the decoder, about 4.5 MB in float32 — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, "
                "the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the threshold the "
                "epoch was selected at, the training configuration and the epoch history (OUT8). "
                "`ClipSegSegmentationPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, its "
                "digest and its exact tensor set **before** deserialising, refuses any tensor outside the decoder, and overlays "
                "the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell "
                "asserts identical masks on eight test records (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "caption_font = ImageFont.load_default(size=18)\n"
                "adapted_items = pipe.segment_batch([(r['image'], r['prompt']) for r in test_records[:6]], threshold=THRESHOLD)\n"
                "for record, frozen_row, adapted_row, adapted_item in zip(test_records[:6], frozen_test['rows'][:6], adapted_test['rows'][:6], adapted_items, strict=True):\n"
                "    thumb = record['image'].convert('RGB')\n"
                "    scale = 320 / max(thumb.size)\n"
                "    size = (max(1, round(thumb.width * scale)), max(1, round(thumb.height * scale)))\n"
                "    small = thumb.resize(size)\n"
                "    ref_small = np.asarray(Image.fromarray(record['mask'].astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127\n"
                "    ada_small = np.asarray(Image.fromarray(adapted_item['mask'].astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127\n"
                "    panels = [overlay(small, ref_small, (60, 179, 75)), overlay(small, ada_small, (220, 40, 40))]\n"
                "    sheet = Image.new('RGB', (sum(p.width for p in panels) + 8, panels[0].height + 56), (255, 255, 255))\n"
                "    x = 0\n"
                "    for panel in panels:\n"
                "        sheet.paste(panel, (x, 0))\n"
                "        x += panel.width + 8\n"
                "    marker = ImageDraw.Draw(sheet)\n"
                "    marker.text((8, panels[0].height + 6), f\"{{record['prompt']}} — reference (green) | adapted (red); IoU frozen {{frozen_row['iou']:.3f}} -> adapted {{adapted_row['iou']:.3f}}\", fill=(20, 20, 20), font=caption_font)\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'panels': ['reference overlay', 'adapted overlay']}})\n\n"
                "adapted_scene, adapted_scene_seconds, adapted_scene_checks, adapted_scene_report = segment_scene(pipe, 'adapted')\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...', 'threshold': artifact_manifest['adapter']['threshold'], 'best_epoch': artifact_manifest['adapter']['best_epoch']}})\n\n"
                "reloaded = ClipSegSegmentationPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "before = [item['mask'] for item in pipe.segment_batch([(r['image'], r['prompt']) for r in test_records[:8]], threshold=THRESHOLD)]\n"
                "after = [item['mask'] for item in reloaded.segment_batch([(r['image'], r['prompt']) for r in test_records[:8]], threshold=THRESHOLD)]\n"
                "parity = {{'identical_masks': sum(bool(np.array_equal(a, b)) for a, b in zip(before, after, strict=True)), 'of': len(before)}}\n"
                "print({{'reload_parity': parity, 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['identical_masks'] == parity['of']\n\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': pipe.weight_sha256}},\n"
                "    'data_source': data_source,\n"
                "    'threshold': THRESHOLD,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'repo': CORPUS_REPO, 'revision': CORPUS_REVISION, 'file': CORPUS_FILE, 'license': CORPUS_LICENSE, 'row_groups': sorted(ROW_GROUP_PINS), 'shard_bytes': CORPUS_BYTES, 'excluded_class_ids': list(EXCLUDED_CLASS_IDS)}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'scene': {{'name': scene_name, 'sha256': scene_sha256, 'prompts': scene_prompts}}, 'frozen': {{'segments': frozen_scene, 'seconds': frozen_scene_seconds, 'checks': frozen_scene_checks, 'report': frozen_scene_report}}, 'adapted': {{'segments': adapted_scene, 'seconds': adapted_scene_seconds, 'checks': adapted_scene_checks, 'report': adapted_scene_report}}, 'output_files': ['outputs/{stem}_scene_frozen.json', 'outputs/{stem}_scene_adapted.json', 'outputs/{stem}_scene_frozen.png', 'outputs/{stem}_scene_adapted.png']}},\n"
                "    'comparison': comparison,\n"
                "    'examples': 'outputs/{stem}_examples',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors'])}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'pillow': PIL.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "A phrase-conditioned segmenter trained on PhraseCut already finds the largest ingredient in a food photograph "
        "roughly when asked by name: the frozen model scores a mean IoU of 0.637 on the FoodSeg103 records. A "
        "bounded fine-tuning of its decoder on 600 records moves that to 0.842 mean IoU and 0.904 Dice "
        "in the build record, with a 4.5 MB adapter that reloads mask-for-mask. That is the claim: the adaptation contract "
        "works end to end on a text-prompted segmenter with a real labelled set, and the numbers it produces are read as "
        "mean and micro IoU, Dice, precision and recall at one stated threshold, against two non-adapted baselines and the "
        "frozen model, with the predicted area beside them rather than in isolation. The Tesla T4 run reproduced the CPU sweep exactly (0.842 / 0.904 at epoch 4, plateau 0.849–0.855 after). The row has no sibling on this corpus — CLIPSeg is the fleet's only text-prompted segmenter — so the comparison is the frozen decoder against the tuned one on the same 140 records: the gain is almost all recall (0.736 → 0.927 at precision 0.835 → 0.881), the records under 0.2 IoU fall from 24 to 1 and those above 0.8 rise from 63 to 111, one record (`garlic`) stays at 0.0, and the drawn scene outside the vocabulary moved by a hundredth (0.947 → 0.958 mean IoU, the two absent phrases still empty): a decoder of 1.1 M parameters learned FoodSeg103's annotation convention, not new visual concepts.\n\n"
        "The test split is 140 records from one seeded draw of one 800-record sample, the validation split that picks the "
        "epoch is 60, and every rate is at the one threshold `MASK_THRESHOLD` — not a benchmark, not a threshold sweep, not "
        "a measure of phrases the sample never asks (one phrase per image, the largest ingredient only). So a result here "
        "says the contract works on food photographs' largest ingredients, not that the adapted model handles other "
        "phrases, other image families or your masks. The decoder that was tuned serves every phrase: the drawn scene "
        "re-segmented in Section 9 is one drawing of evidence about what the tuning did outside its vocabulary "
        "(before adaptation `green grass` 0.96, `a red circle` 0.96, `a blue square` 0.96, `a yellow triangle` 0.91 (mean IoU 0.95); absent phrases' area fraction `a cat` 0.000, `the sky` 0.000; after adaptation `green grass` 0.97, `a red circle` 0.97, `a blue square` 0.96, `a yellow triangle` 0.93 (mean IoU 0.96); absent phrases' area fraction `a cat` 0.000, `the sky` 0.000), not a measurement, and a deployment that segments other phrases must measure them after "
        "adapting. The towers were not adapted: what the image encoder cannot see stays unsegmented, and **the "
        "probabilities remain an uncalibrated sigmoid**.\n\n"
        "Three things to carry to real data. **Baselines first:** the empty and full-mask rates on *your* masks, and the "
        "frozen model's predicted area, are the numbers to read before any adapted one. **Precision and recall together:** a "
        "gain in mean IoU that comes with a collapse of one of them is a moved threshold, and the threshold is yours to "
        "set on a validation split, not the test split. **Leakage:** keep every image in one split (the contract "
        "de-duplicates by decoded pixels) and split by source, session or scene when your images come from few sources.\n\n"
        "Successful execution proves that the recorded repository revision's package, carried in this standalone notebook, "
        "can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real labelled mask set, validate "
        "the demonstrated dataset contract without leakage, execute the inference contract for a drawn scene and a bounded "
        "fine-tuning of the decoder with the upstream objective, evaluate against two non-adapted baselines and the frozen "
        "model on an image-disjoint split, and emit the shown machine-readable artifacts — without the repository being "
        "reachable. It does **not** establish benchmark superiority, segmentation quality on any other phrase vocabulary "
        "or image family, calibration of the sigmoid, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** raise `EPOCHS` and watch the validation mean IoU "
        "pick the epoch; change `LEARNING_RATE` by a factor of ten in either direction and read the curve; set `THRESHOLD` "
        "to `0.3` or `0.7` before Section 6 and read how precision and recall trade against each other for both the frozen "
        "and the adapted model; change `SPLIT_SEED` and read how much 140 records move; or bring your own masks through "
        "BYOD and read the two baselines before the adapted number.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: the "
        "pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the top. "
        "`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file is "
        "incomplete or altered — delete it from `weights/clipseg-rd64-refined/` and rerun Section 3. A `sha256` `ValueError` "
        "naming a parquet row group in Section 4: a cached `weights/foodseg103/validation-rg*.parquet` is incomplete — delete "
        "it and rerun Section 4.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/timojl/clipseg\n"
        "- Image Segmentation Using Text and Image Prompts (Lüddecke and Ecker, CVPR 2022): https://arxiv.org/abs/2112.10003\n"
        "- FoodSeg103 (Apache-2.0): https://huggingface.co/datasets/EduardoPacheco/FoodSeg103 — Wu, Fu, Liu, Lim, Hoi, Sun. A Large-Scale Benchmark for Food Image Segmentation (ACM MM 2021): https://arxiv.org/abs/2105.05409\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
