"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "clipseg_segmentation_pipeline",
    "repo_name": "clipseg-segmentation-pipeline",
    "stem": "clipseg_segmentation",
    "notebook_name": "clipseg_segmentation_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "mode": "GUIDED",
    "pipeline_class": "ClipSegSegmentationPipeline",
    "weights_key": "clipseg-rd64-refined",
    "runtime_imports": ["torch", "transformers"],
    "title": "CLIPSeg rd64-refined — DIMER text-prompted image segmentation tutorial (standalone)",
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
    "capability": "zero-shot (text-prompted) image segmentation — one image plus 1–16 free-text phrases → one binary mask and probability map per phrase — using the pinned `CIDAS/clipseg-rd64-refined` weights",
    "intro": (
        "At inference the CLIPSeg model (a frozen CLIP ViT-B/16 image encoder and CLIP text encoder joined by a small "
        "transformer decoder with 64-dimensional reduced activations and a refined transposed convolution head; about 151M "
        "parameters, the decoder trained on PhraseCut phrase–mask pairs) conditions the decoder on each text phrase and "
        "produces one 352×352 logit map per phrase; the carried module passes the logits through a sigmoid, resamples the "
        "probability map to the input size, and thresholds it into a binary mask under a caller-owned `threshold`. **No "
        "adaptation occurs:** no training, fine-tuning, in-context conditioning, or preprocessing fitting happens in this "
        "notebook — the upstream checkpoint supplies the weights, processor and tokenizer, and the carried module adds "
        "snapshot verification, the input contract (image side ceilings, 1–16 distinct phrases up to 64 characters, a "
        "threshold in [0, 1]), a fixed output contract (mask, probability map, area fraction, tight box and maximum "
        "probability per phrase), and the `mask_iou`, `validate_inputs` and `evaluation_report` helpers. The default sample "
        "is a flat scene of coloured shapes drawn in code with the exact masks they were drawn from, so the per-phrase "
        "`mask_iou` and their mean are demonstration (plumbing) evidence for one drawing, not a PhraseCut benchmark."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic scene with known masks (or upload your own photograph and type "
        "your own phrases) and validate it into an input manifest, choose a threshold, run the supported task, read the "
        "masks correctly (an uncalibrated sigmoid per pixel, independent per phrase, no class exclusivity), exercise an "
        "optional BYOD path, produce an evaluation report that is `sample-sanity` with `mask_iou` and `miou` only when "
        "reference masks exist and `not-measurable` otherwise, and export the masks, an overlay and provenance."
    ),
    "exclusions": (
        "Instance separation (one mask per phrase, even when several objects match it), panoptic or semantic labelling of "
        "every pixel (masks are independent per phrase and may overlap or leave pixels unassigned), the image-prompt "
        "(one-shot) conditioning mode the upstream model also supports, high-resolution boundary accuracy (the decoder works "
        "at 352×352 and the mask is resampled), batch throughput, evaluation on PhraseCut, Pascal-5i or COCO (not bundled; "
        "only drawn shapes are scored here), and any training. The model was trained on photographs with phrase "
        "annotations; flat drawings, documents, medical and satellite imagery, and non-English phrases are outside what "
        "this notebook measures, and a confident-looking mask carries no signal."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. CPU is adequate: the repository's model card records 5.0 s to load and 0.8 s for six phrases on a 640×480 drawn scene (0.15 s for one) in the Windows venv (Intel Core Ultra 9 275HX). The pinned `torch==2.14.0` install and the 603 MB checkpoint are the large downloads of the run.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a per-pixel sigmoid is and why thresholding it is a decision the caller owns; what intersection-over-union of masks measures and why a few drawn shapes are not a benchmark.",
        "- **Data:** the default sample is a deterministic 640×480 scene drawn in code with Pillow (a red circle, a blue square, a yellow triangle and a green ground band on an off-white background; no text rendering, so its digest is stable across Pillow builds) with the exact boolean masks the shapes were drawn from, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, sides between 16 and 4096 px, plus your own phrases typed into the form field; no reference masks exist for uploads, so their report is `not-measurable`. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic scene or optional BYOD\n\n"
                "The default sample is **synthetic** and carries its own references: a red circle, a blue square and a yellow "
                "triangle on an off-white background above a green ground band are drawn with Pillow at 640×480, and the same "
                "drawing calls produce the boolean reference mask for each shape — the same scene the repository's smoke run "
                "used. Four phrases name the four regions and two more (`a cat`, `the sky`) name things that are not there, so "
                "the notebook shows both a mask that should be found and one that should stay empty. The reference masks are "
                "the references for the `mask_iou` sanity check later. They are not a labelled dataset, so nothing here is a "
                "PhraseCut measurement. The image digest is printed for the record. BYOD is optional and disabled by default; "
                "when enabled, upload one image and type your phrases (one per line) — no reference masks exist for them, so "
                "the evaluation report will be `not-measurable`.\n\n"
                "The mask threshold is a **caller-owned request parameter**: `threshold` cuts the per-pixel sigmoid "
                "(`MASK_THRESHOLD = 0.5` is the package default, the natural cut of a sigmoid, not a calibration — the smoke run "
                "recorded IoU 0.81–0.96 at 0.3, 0.91–0.96 at 0.5 and 0.83–0.90 at 0.7 on this scene). Nothing is validated in "
                "this cell — the next section hands the image and the phrases to the pipeline's own validation stage, which is "
                "the only checker. Look for a dictionary naming the sample kind, the image size and digest, the threshold and "
                "the phrases."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "byod_prompts = 'a person\\na dog'  # @param {{type:\"string\"}}\n"
                "threshold = 0.5  # @param {{type:\"number\"}}\n\n\n"
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
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    image_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[image_name]))\n"
                "    image.load()\n"
                "    prompts = [line.strip() for line in byod_prompts.splitlines() if line.strip()]\n"
                "    reference_masks = None\n"
                "    sample_kind = 'BYOD'\n"
                "else:\n"
                "    # Deterministic drawing: no randomness and no text rendering, so no seed is needed and the digest is stable.\n"
                "    image, reference_masks = synthetic_scene()\n"
                "    prompts = list(reference_masks) + ['a cat', 'the sky']  # two absent phrases on purpose\n"
                "    image_name = 'synthetic_shapes_640x480.png'\n"
                "    sample_kind = 'synthetic'\n\n"
                "image_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample_kind': sample_kind, 'name': image_name, 'mode': image.mode, 'size': image.size, 'rgb_sha256': image_sha256, 'threshold': threshold, 'prompts': prompts, 'has_reference_masks': reference_masks is not None}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the request → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: it applies exactly the checks `segment` applies — "
                "image type and sides `MIN_IMAGE_SIDE`..`MAX_IMAGE_SIDE` px, 1..`MAX_PROMPTS` distinct non-empty phrases of at "
                "most `MAX_PROMPT_CHARS` characters (normalised by `format_prompts`), and a threshold in `[0, 1]` — and returns "
                "an **input manifest** naming the schema (including the 352×352 resize that does not preserve aspect ratio, "
                "the sigmoid and the resampling), the input's observed mode and size, the normalised phrases, the threshold and "
                "the verdict. The manifest is written to `outputs/{stem}_input_manifest.json`. To show what rejection looks "
                "like, the cell also validates a duplicated phrase and records the pipeline's own error message as a finding. "
                "Inside the pipeline the image is converted to RGB and resized; nothing else is dropped or altered. The "
                "pipeline cannot tell whether a phrase names anything in the image: that contract is the caller's, and an "
                "absent phrase simply yields an empty (or spurious) mask."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'LOGIT_SIZE': LOGIT_SIZE, 'MAX_PROMPTS': MAX_PROMPTS, 'MAX_PROMPT_CHARS': MAX_PROMPT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MASK_THRESHOLD': MASK_THRESHOLD}}}})\n"
                "input_manifest = validate_inputs(image, prompts, threshold=threshold, names=[image_name])\n"
                "# Demonstrate rejection on a request that breaks the contract; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(image, ['a red circle', 'A red circle.'])\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'duplicate-phrase-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Segment the phrases and read the output correctly\n\n"
                "`segment` returns one entry per phrase with `mask` (a boolean array at input resolution), `probability` (the "
                "float32 sigmoid map it was cut from), `area_fraction`, `max_probability`, a tight `bbox` around the mask (or "
                "`None` when it is empty), plus the normalised `queries`, the `threshold`, the image size and the model "
                "identity. **The probabilities are uncalibrated sigmoids**: a 0.9 pixel is not 90 % likely to belong to the "
                "phrase, and the masks of different phrases are independent — they can overlap, and pixels can belong to none. "
                "Greedy thresholding is deterministic on a fixed device and dtype; CUDA kernels can shift probabilities slightly, "
                "so GPU and CPU masks need not match at the boundary. Every phrase costs one decoder pass over the same encoded "
                "image (about 0.15 s each on the reference CPU). As recorded in the model card, the repository's CPU smoke on "
                "this same scene found all four shapes with IoU 0.91–0.96 at 0.5, left `a cat` and `the sky` empty (maximum "
                "probability ≤ 0.01), and on a blank white image or uniform noise left `a red circle` empty too (maximum 0.02) "
                "— an absent phrase usually yields an empty mask here, but that is an observation, not a guarantee."
            ),
            "code": (
                "import time\n\n"
                "t0 = time.time()\n"
                "result = pipe.segment(image, prompts, threshold=threshold)\n"
                "elapsed = round(time.time() - t0, 2)\n"
                "print({{'device': pipe.device, 'seconds': elapsed, 'n_prompts': len(result['queries']), 'threshold': result['threshold']}})\n"
                "for segment in result['segments']:\n"
                "    print(f\"{{segment['prompt']:20s}} area={{segment['area_fraction']:.3f}}  max_p={{segment['max_probability']:.2f}}  bbox={{segment['bbox']}}\")"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. No accuracy is "
                "reported by default: segmentation quality needs labelled masks on images from the deployment domain with a "
                "matching phrase vocabulary, and this repository ships none (PhraseCut is not bundled). The repository's metric "
                "helper is `mask_iou` — intersection-over-union of two boolean masks — and when reference masks are supplied the "
                "report carries one `mask_iou` entry per phrase (with the reference and predicted area fractions) and their mean "
                "`miou`, with the verdict `sample-sanity`. On the synthetic path those references are shapes **you drew "
                "yourself**, so a high IoU proves only that the input contract, resize, decoder, sigmoid, resampling and "
                "thresholding round-trip. On BYOD no reference masks exist, the verdict is `not-measurable`, and the report "
                "states what would make the task measurable. The report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, reference_masks, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps({{k: v for k, v in report.items() if k != 'metrics'}}, indent=2))\n"
                "for metric in report['metrics']:\n"
                "    print(f\"{{metric['id']:10}} {{metric['value']:.3f}}  {{metric.get('reference', '')}}  ({{metric['estimation']}})\")\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No reference masks exist for these phrases, so nothing is scored; inspect the overlay yourself.')"
            ),
        },
        {
            "md": (
                "## 8. Export outputs and provenance\n\n"
                "Machine-readable JSON preserves the per-phrase statistics (area fraction, maximum probability, tight box), the "
                "queries and threshold, the evaluation report, the input manifest, the sample identity and digest, the "
                "notebook's source (repository, revision, embedded module digest, generator), the model identifier, the immutable "
                "model revision, the model licence, and the runtime identity (Python, `torch`, `transformers`, device); the "
                "masks themselves are written as one 8-bit PNG per phrase (0/255) and the probability maps as one `.npz`, "
                "because arrays do not belong in JSON. An overlay PNG tints each phrase's mask in its own colour on the image "
                "for visual inspection (a supplement to, not a replacement for, the machine-readable files). No credentials are "
                "recorded."
            ),
            "code": (
                "import re\n\n"
                "palette = [(220, 40, 40), (40, 70, 200), (250, 200, 30), (60, 179, 75), (160, 60, 200), (0, 170, 170)]\n"
                "base = np.asarray(image.convert('RGB'), dtype=np.float32)\n"
                "overlay = base.copy()\n"
                "for index, segment in enumerate(result['segments']):\n"
                "    colour = np.array(palette[index % len(palette)], dtype=np.float32)\n"
                "    overlay[segment['mask']] = 0.45 * overlay[segment['mask']] + 0.55 * colour\n"
                "Image.fromarray(overlay.round().astype(np.uint8)).save('outputs/{stem}_overlay.png')\n"
                "mask_files = {{}}\n"
                "for segment in result['segments']:\n"
                "    slug = re.sub(r'[^a-z0-9]+', '_', segment['prompt']).strip('_')\n"
                "    path = f'outputs/{stem}_mask_{{slug}}.png'\n"
                "    Image.fromarray((segment['mask'].astype(np.uint8) * 255)).save(path)\n"
                "    mask_files[segment['prompt']] = path\n"
                "np.savez_compressed('outputs/{stem}_probabilities.npz', **{{segment['prompt']: segment['probability'] for segment in result['segments']}})\n"
                "payload = {{\n"
                "    'segments': [{{k: v for k, v in segment.items() if k not in ('mask', 'probability')}} for segment in result['segments']],\n"
                "    'queries': result['queries'],\n"
                "    'threshold': result['threshold'],\n"
                "    'mask_files': mask_files,\n"
                "    'evaluation_report': report,\n"
                "    'input_manifest': input_manifest,\n"
                "    'sample': {{'kind': sample_kind, 'name': image_name, 'size': list(image.size), 'rgb_sha256': image_sha256, 'prompts': prompts, 'has_reference_masks': reference_masks is not None}},\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The masks are per-pixel sigmoid cuts of a decoder conditioned on your phrase; nothing in the output scores a mask as "
        "a whole, the probabilities are uncalibrated, and masks of different phrases are independent. On the drawn scene the "
        "`mask_iou` values in the evaluation report compare the masks with shapes you drew yourself and the verdict is "
        "`sample-sanity`, which proves only that the input contract, resize, decoder, sigmoid, resampling and thresholding work "
        "(the repository's smoke run scored IoU 0.91–0.96 on the four shapes and left the two absent phrases empty); they say "
        "nothing about photographs, cluttered scenes, thin or small objects (the decoder works at 352×352), phrases naming "
        "attributes or relations, or non-English prompts, and a BYOD result is a single-image observation with the verdict "
        "`not-measurable`. **An absent phrase is not guaranteed an empty mask** and a present one is not guaranteed a full "
        "one: the threshold trades area for precision (0.3 grew every mask, 0.7 shrank them in the smoke run), so choose it on "
        "your own labelled masks. The pipeline provides no instance separation, no exhaustive pixel labelling, no image-prompt "
        "mode, no benchmark evaluation and no training capability.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, can "
        "acquire and digest-verify the pinned model, validate the demonstrated request, execute the public pipeline path, and "
        "emit the shown machine-readable outputs in the tested runtime — without the repository being reachable. It does **not** "
        "establish benchmark superiority, deployment calibration, safety for high-consequence decisions, or production fitness on "
        "an unseen domain.\n\n"
        "**Next experiments:** set `threshold` to 0.3 and 0.7 and watch the IoUs move; rename `a blue square` to `a blue "
        "rectangle` or `a blue box`; ask for `a shape` and see which pixels the decoder assigns; enable `USE_BYOD` with a "
        "photograph you know, then build your own boolean reference masks and pass them to `evaluation_report` to see the "
        "verdict switch to `sample-sanity`.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/timojl/clipseg\n"
        "- Image Segmentation Using Text and Image Prompts (Lüddecke and Ecker, 2021): https://arxiv.org/abs/2112.10003\n"
        "- PhraseCut: Language-based Image Segmentation in the Wild (Wu et al., 2020): https://arxiv.org/abs/2008.01187"
    ),
}
