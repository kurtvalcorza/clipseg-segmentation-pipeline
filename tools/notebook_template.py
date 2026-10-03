"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.2 §4 standalone carrier).

Only the task-specific prose and stage cells live here. The isolated runtime, the embedded package (three modules,
carried verbatim in dependency order), and the model pin/stage/verify cells are produced by the generator from
repository sources so they cannot drift from the package.

This template configures an E2E segmentation-adaptation workflow: the pinned CIDAS/clipseg-rd64-refined snapshot is
digest-verified and loaded, 800 Apache-2.0 FoodSeg103 food photographs with pixel-wise ingredient masks are fetched
as eight digest-pinned parquet row groups over HTTPS range requests and turned into (image, phrase, mask) records,
the records are validated and split by image, a synthetic scene of drawn shapes is segmented through the inference
contract, the frozen model is scored over the held-out records (mean IoU, micro IoU, Dice, pixel precision and
recall at the threshold) beside an empty-mask and a full-mask baseline, a bounded fine-tuning of the CLIPSeg decoder
runs on cached CLIP activations with validation-mIoU epoch selection, the held-out split is scored again, six
held-out records and the drawn scene are re-run with the adapted model, the adapter is exported and reloaded, and a
change-one-thing activity reads the adapted model's validation masks at a second threshold.

Review fixes (2026-10-02 review, CLS-M1..M5, CLS-m1..m5) live here and in the generator, never in src/: the
image-segmentation workshop notebook carries src/ byte for byte, so the BYOD loader the notebook uses
(`load_byod_records`, keyed by relative path) is defined in the Section 4 cell and the package's
`load_byod_dataset` is left unchanged.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

# The BYOD loader used by Section 4 (CLS-M4, CLS-m5). Plain Python, inserted into the cell with its braces escaped
# for str.format, so it can be read and tested here as written.
_BYOD_HELPERS = r'''BYOD_IGNORED_NAMES = ('.DS_Store', 'Thumbs.db', 'desktop.ini')
BYOD_LOSSY_MASK_SUFFIXES = ('.jpg', '.jpeg', '.jfif')


def _byod_key(name, where):
    """A member's path relative to the upload root, '/'-separated; absolute paths and '..' are refused."""
    raw = str(name).replace('\\', '/').strip()
    if not raw:
        raise ValueError(f'{where}: an empty path')
    if raw.startswith('/') or re.match(r'^[A-Za-z]:', raw):
        raise ValueError(f'{where}: {raw!r} is an absolute path; use a path relative to the folder that holds masks.csv')
    parts = [part for part in raw.split('/') if part not in ('', '.')]
    if '..' in parts:
        raise ValueError(f'{where}: {raw!r} leaves the upload folder (..)')
    return '/'.join(parts)


def _byod_ignored(key):
    """Operating-system metadata that a zip tool adds on its own: macOS __MACOSX/ and ._ files, .DS_Store, Thumbs.db."""
    parts = key.split('/')
    return '__MACOSX' in parts[:-1] or parts[-1].startswith('._') or parts[-1] in BYOD_IGNORED_NAMES


def load_byod_records(path):
    """BYOD records from a .zip or a directory holding masks.csv (columns file, mask, prompt and optionally id).

    Every member is keyed by its full relative path (never by its file name alone), so images/0001.png and
    masks/0001.png stay two files; the file and mask columns are resolved relative to the folder that holds
    masks.csv. Refused with a named message: two members with the same path, a row whose image and mask are the same
    file, a listed file that is missing, a file no row names, a JPEG mask (lossy compression turns into mask pixels)
    and an undecodable image. Returns (records, report)."""
    source = Path(path)
    members = {}
    if source.is_dir():
        for file in sorted(source.rglob('*')):
            if file.is_file():
                members[_byod_key(file.relative_to(source).as_posix(), 'BYOD directory')] = file.read_bytes()
    elif zipfile.is_zipfile(source):
        with zipfile.ZipFile(source) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                key = _byod_key(info.filename, 'BYOD zip')
                if key in members:
                    raise ValueError(f'the zip holds two members with the same path {key!r}; keep one of them')
                members[key] = archive.read(info)
    else:
        raise ValueError(f'{source} is neither a .zip file nor a directory')
    ignored = sorted(key for key in members if _byod_ignored(key))
    for key in ignored:
        del members[key]
    tables = sorted(key for key in members if key.rsplit('/', 1)[-1] == 'masks.csv')
    if len(tables) != 1:
        raise ValueError(f'BYOD data must hold exactly one masks.csv (with the columns file, mask and prompt); found {len(tables)}: {tables[:3]}')
    table = tables[0]
    root = table.rsplit('/', 1)[0] + '/' if '/' in table else ''
    rows = list(csv.DictReader(io.StringIO(members[table].decode('utf-8-sig'))))
    if not rows or any(column not in rows[0] for column in ('file', 'mask', 'prompt')):
        raise ValueError('masks.csv must have the columns file, mask and prompt')
    records, listed, non_binary = [], {table}, []
    for number, row in enumerate(rows, start=2):
        where = f'masks.csv line {number}'
        image_key = root + _byod_key(row.get('file', ''), where + ' (file)')
        mask_key = root + _byod_key(row.get('mask', ''), where + ' (mask)')
        if image_key == mask_key:
            raise ValueError(f'{where}: the image and the mask are the same file {image_key!r}; the mask must be a separate image')
        for needed in (image_key, mask_key):
            if needed not in members:
                same_name = [key for key in members if key.rsplit('/', 1)[-1] == needed.rsplit('/', 1)[-1]]
                hint = f'; the upload has {same_name[:3]} (paths in masks.csv are relative to the folder holding it)' if same_name else ''
                raise ValueError(f'{where}: masks.csv names a missing file {needed!r}{hint}')
        if mask_key.lower().endswith(BYOD_LOSSY_MASK_SUFFIXES):
            raise ValueError(f'{where}: the mask {mask_key!r} is a JPEG; JPEG compression adds stray non-zero pixels that would count as mask. Save masks as PNG (lossless) with 0 for background.')
        listed.update((image_key, mask_key))
        try:
            image = Image.open(io.BytesIO(members[image_key]))
            image.load()
            mask = Image.open(io.BytesIO(members[mask_key]))
            mask.load()
        except Exception as exc:  # noqa: BLE001
            raise ValueError(f'{where}: not a decodable image: {image_key} / {mask_key}') from exc
        if len(np.unique(np.asarray(mask.convert('L')))) > 2:
            non_binary.append(mask_key)
        default_id = re.sub(r'[^A-Za-z0-9_.:-]', '_', image_key.rsplit('.', 1)[0])[-64:]
        records.append({'id': str(row.get('id', '') or '').strip() or default_id, 'image': image.convert('RGB'), 'prompt': str(row.get('prompt', '')), 'mask': mask})
    unlisted = sorted(key for key in members if key not in listed)
    if unlisted:
        raise ValueError(f'{len(unlisted)} file(s) in the upload are not named by any masks.csv row, e.g. {unlisted[0]!r}: delete them from the upload or add a row for each (macOS __MACOSX/ and ._ files, .DS_Store and Thumbs.db are ignored automatically)')
    return records, {'files': len(members), 'ignored_os_metadata': len(ignored), 'non_binary_masks': len(non_binary), 'non_binary_examples': non_binary[:3]}


def byod_minimum_images(val_fraction=0.15, test_fraction=0.2):
    """The smallest number of distinct images split_dataset turns into a training split of MIN_RECORDS and a
    non-empty validation and test split (split_dataset keeps round(20 %) for test, at least one, and round(15 %)
    for validation)."""
    for n in range(MIN_RECORDS, MAX_RECORDS + 1):
        n_test = max(1, round(n * test_fraction))
        n_val = round(n * val_fraction)
        if n_val >= 1 and n - n_test - n_val >= MIN_RECORDS:
            return n
    raise ValueError('no BYOD size satisfies the split minimums')


def describe_byod_source(source, file_name):
    """File name, byte count and SHA-256 of the upload (a directory: SHA-256 over the sorted relative paths and file digests)."""
    source = Path(source)
    if source.is_dir():
        files = sorted(p for p in source.rglob('*') if p.is_file())
        listing = ''.join(f'{p.relative_to(source).as_posix()}\t{hashlib.sha256(p.read_bytes()).hexdigest()}\n' for p in files)
        return {'name': file_name, 'kind': 'directory', 'files': len(files), 'bytes': sum(p.stat().st_size for p in files), 'sha256': hashlib.sha256(listing.encode('utf-8')).hexdigest(), 'sha256_of': 'sorted relative path + TAB + file SHA-256 lines'}
    payload = source.read_bytes()
    return {'name': file_name, 'kind': 'zip', 'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest()}


BYOD_MIN_IMAGES = byod_minimum_images()
'''


def _esc(source: str) -> str:
    """Escape braces so the generator's str.format leaves the code as written."""
    return source.replace("{", "{{").replace("}", "}}")


TEMPLATE = {
    "package": "clipseg_segmentation_pipeline",
    "repo_name": "clipseg-segmentation-pipeline",
    "stem": "clipseg_segmentation",
    "notebook_name": "clipseg_segmentation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    # GDL11 (NOTEBOOK_SPEC 2.2 §3.5): Sections 1-3 labelled Infrastructure and the carried module source collapsed.
    "infrastructure_labels": True,
    "isolated_runtime": True,
    # The fleet's uv isolated-environment mechanism (bart-mnli-zero-shot-classification-pipeline ee128d2, generator /2.1):
    # managed CPython, a size- and SHA-256-verified uv wheel (the same wheel as this repository's workshop notebook), and a
    # lock compiled from the pyproject pins with `uv pip compile pyproject.toml --python-version 3.12 --python-platform
    # x86_64-manylinux_2_28 --generate-hashes --only-binary :all: -o tutorials/requirements-colab.lock.txt`, transitive
    # versions constrained to the workshop's carried requirements.txt.
    "managed_python": "3.12.12",
    "uv": {
        "version": "0.12.15",
        "url": "https://files.pythonhosted.org/packages/1e/fd/432451d732917c49152a291de3ef171aa6b0f1a22d39780fb2c1f085ca4c/uv-0.12.15-py3-none-manylinux_2_17_x86_64.manylinux2014_x86_64.whl",
        "bytes": 20081404,
        "sha256": "aee9802f46bae436bd91751bb33ddeb379ef1596b5c19df193219d545d244b60",
    },
    "lock": "tutorials/requirements-colab.lock.txt",
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
        "Selecting **Run all** in a fresh supported runtime builds an isolated environment from the hash-locked pins (the "
        "kernel's own packages are left alone, so no restart is needed), stages and digest-verifies the "
        "pinned `CIDAS/clipseg-rd64-refined` snapshot (a 603 MB `model.safetensors`; no pickle is opened anywhere), fetches "
        "the first eight row groups of the FoodSeg103 validation shard from the Hugging Face Hub at an immutable revision "
        "with HTTPS range requests (about 43 MB; each row group refused on any SHA-256 or byte-total mismatch), turns each "
        "of the 800 images into one (image, phrase, mask) record and splits them by image into 600 / 60 / 140, segments a "
        "synthetic scene of drawn shapes through the inference contract with an input manifest and a rejection probe, "
        "scores the frozen model over the 140 held-out records (mean IoU, micro IoU, Dice, pixel precision and recall at "
        "the threshold) beside an empty-mask and a full-mask baseline, runs a bounded fine-tuning of the CLIPSeg decoder on "
        "cached CLIP activations with validation-mIoU epoch selection, scores the held-out records again, re-runs six "
        "held-out records and the drawn scene with the adapted model, exports the adapter as safetensors with a manifest, "
        "reloads that artifact into a fresh pipeline to verify mask parity, and ends with a change-one-thing threshold "
        "activity on the validation records. The default path needs no repository clone, no DIMER worker or service, no "
        "credential, no upload dialog, no restart and no configuration edit (NOTEBOOK_SPEC 2.2 §5). Measured times name "
        "their environment: on a Kaggle Tesla T4 (2026-09-21, the previous notebook version, which installed the pins "
        "into the kernel) the cells after the install took 176 s — the snapshot download 66 s, caching the tower "
        "activations of the 600 + 60 records 30 s, the eight epochs 59 s, scoring the 140 frozen records 7 s. Building the "
        "isolated environment adds the download and install of the locked packages (PyTorch with its CUDA libraries); "
        "no hosted run of this version is recorded yet. A CUDA runtime is used automatically when present and is "
        "recommended; on CPU the same path is practical but slower (see the Prerequisites)."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4, select that cell and choose "
        "**Runtime → Run after**. Section 4 reads one `.zip` (or a directory) named in `BYOD_PATH`, or, when `BYOD_PATH` "
        "is empty, the file you choose in the Colab upload dialog: images and mask images plus a `masks.csv` (`file`, "
        "`mask`, `prompt`, optional `id`; one row per image; paths relative to the folder holding `masks.csv`, so "
        "`images/0001.png` and `masks/0001.png` stay two files; masks lossless PNG, non-zero pixels = mask). It needs "
        "**at least 12 distinct images** (the seeded split keeps about 20 % for test and 15 % for validation, and "
        "training needs 8) and refuses a smaller set before splitting, naming your count. Sections 5, 6 and 7 reload the "
        "pretrained model first, so the frozen rows and epoch 0 are the untouched model's. The records pass through the "
        "same validation, image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and "
        "reload-parity cells as the FoodSeg103 sample. Uploaded files stay inside this runtime. BYOD is optional and never "
        "part of the default path."
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
        "already finds the right region roughly: a mean IoU of **0.637** on the 140 held-out records (Kaggle T4 run of "
        "2026-09-21; the full-mask baseline scores 0.283). So the honest question is narrow: does a bounded "
        "fine-tuning of the 1,127,009-parameter decoder on 600 records — the CLIP towers frozen, exactly as the upstream "
        "authors trained it — move the held-out **mean IoU**, **Dice**, **pixel precision** and **pixel recall** on an "
        "image-disjoint test split past the frozen model and two **non-adapted baselines**, and what does it do to the "
        "drawn shapes the same decoder segments? Nothing here is a claim about your images or your phrases: it is one "
        "seeded split of one small labelled set.\n\n"
        "**Snapshot note:** the pinned revision ships `model.safetensors` (an 8-file manifest with the tokenizer and processor "
        "files) — no pickle is opened anywhere in this notebook. Section 3 stages and digest-verifies those files before the "
        "processor or the model is constructed. The pipeline runs in **float32 on every device**: the adapter is trained in "
        "float32 and overlays without a cast, and CPU, Tesla-class and consumer GPUs then run the same arithmetic.\n\n"
        "**Who this is for.** A learner who knows basic Python and NumPy, has met the idea of a neural network, and wants "
        "to see how a pretrained segmentation model is measured on new data and adapted to it without fooling themselves. "
        "No prior experience with CLIPSeg, segmentation metrics or fine-tuning is assumed; each term is explained where it "
        "is first used and again in the **Glossary** at the end.\n\n"
        "**Input → Model → Output.**\n\n"
        "| | Inference | Adaptation |\n"
        "|---|---|---|\n"
        "| Input | one RGB image (sides 16..4,096 px) and 1–16 short phrases | 600 / 60 training and validation records, each a photograph, one phrase and that phrase's pixel mask |\n"
        "| Model | CLIPSeg rd64-refined: frozen CLIP image and text towers + a small decoder | the same model; only the 1.1 M-parameter decoder is trained, on cached tower activations |\n"
        "| Output | per phrase: a probability map (an uncalibrated sigmoid), a mask at `THRESHOLD`, its area and box | held-out mean IoU / Dice / precision / recall against two baselines and the frozen model, and a safetensors adapter that reloads with parity |\n\n"
        "**How to use this notebook.** Choose a runtime (a GPU runtime is much faster; CPU works), then **Runtime → Run "
        "all**. Sections 1–3 are **infrastructure** — the isolated environment, the carried code (collapsed) and the "
        "model verification — and can be run without study. The learning path starts in Section 4. Form fields "
        "(`# @param`) are the only values meant to be edited; the defaults reproduce the recorded path. Each stage states "
        "what it does and asks you to **predict** before it runs; the next cell opens with **What to notice** and a "
        "collapsible **Check your reasoning** block with a worked answer from a recorded run. Sections 5, 6 and 7 always "
        "start from the pretrained model, so any re-run (BYOD or an experiment) compares against the untouched model. "
        "Section 10 is a **change-one-thing activity** that runs after the default path without changing it; each "
        "optional experiment names the field to change and the cell to re-run from (**Runtime → Run after**). "
        "**Troubleshooting**, a **Glossary** and a **Conclusion** template are at the end. Your notes are optional and "
        "are not required submissions.\n\n"
        "**Roadmap:** 4 fetch, validate and split the FoodSeg103 records by image → 5 segment a drawn scene through the "
        "inference contract → 6 two baselines and the frozen model on the test records → 7 fine-tune the decoder, the "
        "epoch selected on validation → 8 compare on the held-out records → 9 look at the masks, export and reload the "
        "adapter → 10 **change one thing: the threshold** → conclude."
    ),
    "learning_objectives": (
        "by the end you should be able to (1) explain *image + phrase → frozen CLIP towers → decoder → per-pixel sigmoid "
        "→ mask at a threshold* and say why the probabilities are not calibrated (Section 5); (2) explain why the split "
        "is made by image and checked for leakage (Section 4); (3) read mean IoU, micro IoU, Dice, pixel precision and "
        "pixel recall, and explain from Section 6's numbers why the empty-mask and full-mask baselines frame the frozen "
        "model's score; (4) predict and then check which epoch the validation split keeps and what the adaptation buys "
        "on held-out records (Sections 7–8); (5) compare the reference, frozen and adapted masks of the same records by "
        "eye (Section 9); (6) predict and then measure how a different threshold trades precision against recall "
        "(Section 10); and (7) check that an exported adapter reproduces the evaluated model (reload parity, Section 9). "
        "Along the way the notebook stages and digest-verifies the immutable upstream snapshot and fetches a "
        "digest-pinned labelled mask set."
    ),
    "exclusions": (
        "instance or panoptic segmentation (one binary mask per phrase; masks of different phrases are independent), "
        "phrase vocabularies beyond one phrase per record in the adaptation sample, threshold tuning (every measurement "
        "in Sections 5–9 is at the one `THRESHOLD`, `MASK_THRESHOLD` by default; Section 10 only reads a second threshold "
        "on the validation records), fine-tuning of the CLIP image or text towers, evaluation on PhraseCut or a "
        "segmentation benchmark proper (only one seeded 800-record sample is scored here), and any claim that ingredient "
        "masks stand in for your images. The repository exposes none of these."
    ),
    "prerequisites": [
        "- **Learner:** basic Python and NumPy, and Colab or Jupyter familiarity; no prior experience with segmentation or fine-tuning. The notebook explains the sigmoid and the threshold, IoU, Dice, precision and recall, the two baselines, the decoder, the frozen-tower cache and epoch 0 where they are first used; the Glossary repeats them.",
        "- **Runtime:** a fresh supported runtime (Google Colab, Kaggle or Linux Jupyter — **Linux x86_64 only**; the notebook builds its own isolated Python 3.12.12 environment, and a Windows or macOS kernel is not supported). The default path uses CUDA automatically when present, float32 on every device; a GPU runtime is recommended. Measured, each figure with its environment: on a Kaggle Tesla T4 (2026-09-21, previous notebook version) scoring the 140 frozen records took 7 s, caching the tower activations of 600 + 60 records 30 s and the eight epochs 59 s, and the cells after the install 176 s in all (66 s of it the snapshot download); in a local CPU run of this version (2026-10-04, 24-thread Windows workstation, CPython 3.12.10, files pre-staged) the tower cache took 132 s and the eight epochs 206 s. No hosted CPU run is recorded; on a 2-vCPU hosted CPU runtime expect several times the workstation figures (an estimate). The locked install (PyTorch 2.14.0 with its CUDA libraries) and the 603 MB checkpoint are the large downloads of the run; the row groups are about 43 MB. The activation cache holds about 1.3 GB of half-precision tensors in host memory for 600 records.",
        "- **Knowledge:** basic Python, NumPy and PIL; what a per-pixel sigmoid is and why thresholding it is a decision the caller owns; what intersection-over-union and Dice measure and why 140 records from one draw give no dispersion; why a self-drawn scene is a plumbing check while a held-out split of one labelled set is a measurement of that set only.",
        "- **Data contract:** records are `{id, image, prompt, mask}` — `image` a PIL image (or a file decodable by Pillow) with sides within 16..4,096 px, `prompt` one phrase of at most 64 characters (normalised like a query), `mask` a boolean height × width array or a mask image whose non-zero pixels are the mask, with at least one true pixel. Ids match `[A-Za-z0-9_.:-]{1,64}` and are unique; `validate_dataset` accepts 8..5,000 records per call, and splitting de-duplicates by decoded pixels so no image lands in two splits. **BYOD** accepts one `.zip` or directory (the Colab upload dialog, or `BYOD_PATH` on any runtime) holding a `masks.csv` with the columns `file`, `mask`, `prompt` (optional `id`); its paths are relative to the folder holding `masks.csv`, every file must be named by a row (macOS `__MACOSX/` and `._` files, `.DS_Store` and `Thumbs.db` are ignored), masks must be lossless (PNG; a JPEG mask is refused) and a mask with grey levels is counted and reported (every non-zero pixel is mask). A BYOD set needs **at least 12 distinct images**: the seeded split keeps about 20 % for test and 15 % for validation (at least one each), and training needs 8 records; Section 4 refuses a smaller set before splitting and names the count.",
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
                "too small to use — each rejected before the model does anything.\n\n"
                "**Bring your own data (optional).** With `USE_BYOD = True` the cell reads your set from `BYOD_PATH` (a `.zip` "
                "or a directory) or, when that is empty, from the Colab upload dialog, with `load_byod_records` (defined in "
                "this cell). Every file is keyed by its path relative to the folder holding `masks.csv`, so `images/0001.png` "
                "and `masks/0001.png` are two files; a row whose image and mask are the same file, two zip members with the "
                "same path, a file no row names, a missing file and a JPEG mask are each refused with a message naming the "
                "path. Before splitting, the cell refuses a set with fewer than 12 distinct images and names your count. The "
                "validation and test splits of a small set may hold only a few records; they are validated with a minimum of "
                "one, the training split with `MIN_RECORDS`. After changing the fields, select this cell and choose "
                "**Runtime → Run after**.\n\n"
                "**Predict before running:** the split is made by *image*. Could the same photograph appear in the training "
                "and the test split if it were listed twice under two ids? Which of the four refusal probes do you expect the "
                "validator to accept?"
            ),
            "code": (
                "import csv\n"
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import re\n"
                "import time\n"
                "import zipfile\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw, ImageFont\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "BYOD_PATH = ''  # @param {{type:\"string\"}}\n"
                "SPLIT_SEED = 42  # @param {{type:\"integer\"}}\n\n\n"
                + _esc(_BYOD_HELPERS)
                + "\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "byod_upload = None\n"
                "if USE_BYOD:\n"
                "    if BYOD_PATH.strip():\n"
                "        byod_source = Path(BYOD_PATH.strip()).expanduser()\n"
                "        if not byod_source.exists():\n"
                "            raise FileNotFoundError(f'BYOD_PATH {{str(byod_source)!r}} does not exist; set it to a .zip or a directory holding masks.csv')\n"
                "        file_name = byod_source.name\n"
                "    else:\n"
                "        try:\n"
                "            from google.colab import files\n"
                "        except ImportError:\n"
                "            raise RuntimeError('USE_BYOD is True and BYOD_PATH is empty, but this runtime has no Colab upload dialog: set BYOD_PATH to a .zip or a directory holding masks.csv.') from None\n"
                "        uploaded = files.upload()\n"
                "        if len(uploaded) != 1:\n"
                "            raise ValueError(f'Upload exactly one .zip file (received {{len(uploaded)}}): run this cell again and choose the file, or set BYOD_PATH to a .zip or a directory.')\n"
                "        file_name, payload = next(iter(uploaded.items()))\n"
                "        byod_source = Path('work') / 'byod.zip'\n"
                "        byod_source.parent.mkdir(parents=True, exist_ok=True)\n"
                "        byod_source.write_bytes(payload)\n"
                "    byod_upload = describe_byod_source(byod_source, file_name)\n"
                "    records, byod_report = load_byod_records(byod_source)\n"
                "    distinct_images = len({{image_digest(r['image']) for r in records}})\n"
                "    if distinct_images < BYOD_MIN_IMAGES:\n"
                "        raise ValueError(f'Your upload holds {{len(records)}} records ({{distinct_images}} distinct images); BYOD needs at least {{BYOD_MIN_IMAGES}} distinct images: the seeded split keeps about 20 % for test and 15 % for validation, and training needs {{MIN_RECORDS}}. Add images and run this cell again.')\n"
                "    if byod_report['non_binary_masks']:\n"
                "        print({{'warning': f\"{{byod_report['non_binary_masks']}} mask(s) have grey levels, e.g. {{byod_report['non_binary_examples']}}: every non-zero pixel counts as mask. Save masks with exactly two values (0 and 255) if that is not what you meant.\"}})\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_rows = {{'byod': len(records), 'distinct_images': distinct_images, 'minimum_distinct_images': BYOD_MIN_IMAGES, **byod_report}}\n"
                "else:\n"
                "    t0 = time.perf_counter()\n"
                "    corpus_groups = fetch_corpus(cache_dir='weights/foodseg103')\n"
                "    corpus = read_corpus(corpus_groups)\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} @ {{CORPUS_REVISION[:12]}} ({{CORPUS_LICENSE}})'\n"
                "    raw_rows = {{'row_groups': len(corpus_groups), 'images': sum(len(v) for v in corpus_groups.values()), 'records': len(corpus), 'bytes': sum(len(r['image']) + len(r['label']) for v in corpus_groups.values() for r in v), 'seconds': round(time.perf_counter() - t0, 1)}}\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=MIN_RECORDS if name == 'train' else 1) for name, part in splits.items()}}\n"
                "splits = {{name: manifest['records'] for name, manifest in dataset_manifests.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw_rows': raw_rows, 'splits': disjoint}})\n"
                "if byod_upload is not None:\n"
                "    print({{'byod_upload': byod_upload}})\n"
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
                "**What to notice:** 600 / 60 / 140 records, three different digests, and all four refusal probes rejected.\n\n"
                "<details><summary>Check your reasoning</summary>No: the split de-duplicates by the decoded pixels before it "
                "draws, so a photograph listed twice under two ids is kept once and lands in one split; `check_split_disjoint` "
                "would raise if an image appeared in two splits. A duplicate in training and test would make the test score "
                "optimistic, because the model would be scored on a picture it was trained on. None of the probes is accepted: "
                "in the Kaggle T4 run of 2026-09-21 and in a local CPU run of this version (2026-10-04) the duplicate id, the "
                "empty mask, the 8 × 8 mask and the three-record set were each rejected by `validate_dataset` before the model "
                "ran.</details>\n\n"
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
                "parameter**, set here as the form field `THRESHOLD` (default `MASK_THRESHOLD`, 0.5) and used by every "
                "measurement in Sections 5–9; an absent phrase still yields a map. `evaluation_report` with the drawn masks is "
                "`sample-sanity`: one `mask_iou` per drawn phrase and their mean, plumbing evidence for one drawing — a "
                "segmentation benchmark needs labelled masks, which Section 6 supplies. At the default threshold the frozen "
                "model scored a mean IoU of **0.947** on the four drawn shapes (Kaggle T4 run of 2026-09-21 and local CPU runs "
                "alike).\n\n"
                "**Every pass starts from the pretrained model.** Section 7 changes the model in memory. If this cell, "
                "Section 6 or Section 7 runs again after Section 7 (a BYOD run or one of the optional experiments), it "
                "first reloads the pretrained model from the verified snapshot with `reset_to_pretrained()`, so a *frozen* "
                "number is never read from the adapted decoder.\n\n"
                "**Predict before running:** will `a cat` and `the sky` — phrases with nothing to find in the drawing — come "
                "back with an empty mask, a small one, or a mask of the whole picture? Which drawn shape do you expect the "
                "lowest IoU for?"
            ),
            "code": (
                "import gc\n\n"
                "THRESHOLD = 0.5  # @param {{type:\"number\"}}\n\n\n"
                "def reset_to_pretrained():\n"
                "    \"\"\"Sections 5-7 start from the pretrained model: if an earlier pass adapted `pipe`, reload it from the verified snapshot.\"\"\"\n"
                "    global pipe\n"
                "    if pipe.adapter is None:\n"
                "        return\n"
                "    pipe = None\n"
                "    gc.collect()\n"
                "    if torch.cuda.is_available():\n"
                "        torch.cuda.empty_cache()\n"
                "    pipe = ClipSegSegmentationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
                "    print({{'reloaded': 'the pretrained model, from the verified snapshot', 'reason': 'an earlier pass had adapted the decoder in memory'}})\n\n\n"
                "reset_to_pretrained()\n\n\n"
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
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'LOGIT_SIZE': LOGIT_SIZE, 'MAX_PROMPTS': MAX_PROMPTS, 'MAX_PROMPT_CHARS': MAX_PROMPT_CHARS, 'MAX_TEXT_TOKENS': MAX_TEXT_TOKENS, 'MASK_THRESHOLD': MASK_THRESHOLD, 'EXTRACT_LAYERS': list(EXTRACT_LAYERS), 'MIN_RECORDS': MIN_RECORDS, 'MAX_RECORDS': MAX_RECORDS, 'EVAL_BATCH_SIZE': EVAL_BATCH_SIZE, 'device': pipe.device}}, 'threshold': THRESHOLD}})\n"
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
                "    print({{label: {{'seconds': seconds, 'checks': checks, 'mask_iou': ious, 'miou': None if miou is None else round(miou, 3), 'absent_phrases': {{p: summary[p]['area_fraction'] for p in ('a cat', 'the sky')}}, 'verdict': report['verdict'], 'adapted': pipeline.adapter is not None}}}})\n"
                "    return summary, seconds, checks, report\n\n\n"
                "frozen_scene, frozen_scene_seconds, frozen_scene_checks, frozen_scene_report = segment_scene(pipe, 'frozen')"
            ),
        },
        {
            "md": (
                "**What to notice:** four `True` checks, a mean IoU near 0.95, and an area fraction of 0.0 for both absent "
                "phrases; `adapted` is `False`.\n\n"
                "<details><summary>Check your reasoning</summary>Empty: the sigmoid stays below 0.5 everywhere for `a cat` "
                "and `the sky`, so their masks have no pixel (area fraction 0.000), though each still has a probability map. "
                "The triangle is hardest — in the Kaggle T4 run of 2026-09-21 the frozen IoUs were `green grass` 0.96, `a red "
                "circle` 0.96, `a blue square` 0.96 and `a yellow triangle` 0.91, mean **0.947** — thin corners are where a "
                "352 × 352 logit grid resampled to 640 × 480 loses pixels. One drawing is plumbing evidence, not a "
                "measurement.</details>\n\n"
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
                "scores the mask. The cell reloads the pretrained model first if an earlier pass adapted it, refuses to "
                "report an adapted model as frozen, prints `adapted` with the scores, and keeps the frozen masks of six test "
                "records for the panels in Section 9. Scoring the 140 records took 7 s on a Kaggle T4 (2026-09-21).\n\n"
                "**Predict before running:** order the three — empty mask, full mask, frozen model — from best to worst mean "
                "IoU. Will the frozen model's pixel precision be higher or lower than its pixel recall (does it draw masks "
                "that are too small or too large)?"
            ),
            "code": (
                "METRICS = ('miou', 'iou_micro', 'dice', 'pixel_precision', 'pixel_recall')\n\n"
                "reset_to_pretrained()\n"
                "baseline_empty = empty_baseline(test_records)\n"
                "baseline_full = full_baseline(test_records)\n"
                "print({{'empty_baseline': {{k: round(baseline_empty[k], 3) for k in METRICS}}, 'n': baseline_empty['n'], 'note': baseline_empty['baseline']}})\n"
                "print({{'full_baseline': {{k: round(baseline_full[k], 3) for k in METRICS}}, 'note': baseline_full['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "frozen_test = pipe.evaluate(test_records, threshold=THRESHOLD, batch_size=EVAL_BATCH_SIZE)\n"
                "if frozen_test['adapted']:\n"
                "    raise RuntimeError('Section 6 must score the frozen (pretrained) model, but the model in memory is adapted. Restart the session and choose Run all.')\n"
                "print({{'frozen_model_test': {{k: round(frozen_test[k], 3) for k in METRICS}}, 'adapted': frozen_test['adapted'], 'n': frozen_test['n'], 'predicted_area_fraction': round(frozen_test['predicted_area_fraction'], 3), 'verdict': frozen_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'definitions': frozen_test['definitions']}})\n"
                "for row in frozen_test['rows'][:4]:\n"
                "    print({{'id': row['id'], 'prompt': row['prompt'], 'iou': round(row['iou'], 3), 'dice': round(row['dice'], 3), 'reference_pixels': row['reference'], 'predicted_pixels': row['predicted']}})\n"
                "frozen_example_masks = [item['mask'] for item in pipe.segment_batch([(r['image'], r['prompt']) for r in test_records[:6]], threshold=THRESHOLD)]"
            ),
        },
        {
            "md": (
                "**What to notice:** empty 0.000, full about 0.28, frozen about 0.64 mean IoU with `adapted: False`, and "
                "precision above recall.\n\n"
                "<details><summary>Check your reasoning</summary>Frozen model, then full mask, then empty mask. In the Kaggle "
                "T4 run of 2026-09-21 (and identically in local CPU runs) the empty mask scored 0.000 by construction, the "
                "full mask 0.283 — the references cover about 28 % of their photographs — and the frozen model **0.637** "
                "mean IoU, Dice 0.715. Its pixel precision 0.835 is well above its recall 0.736: when it marks a pixel it is "
                "usually right, but it marks too few — it under-segments. 63 of the 140 records score above 0.8 IoU, while 24 "
                "are missed almost entirely (IoU under 0.2), typically ingredient names the decoder does not ground at all "
                "(`pie`, `lamb`, `garlic`, `shellfish` at 0.0).</details>\n\n"
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
                "validation mean IoU** (the earliest on ties) is kept. The cell reloads the pretrained model first if an "
                "earlier pass adapted it, and refuses to train on top of an adapted decoder, so epoch 0 is always the frozen "
                "model. On a Kaggle T4 (2026-09-21) the tower cache took 30 s and the eight epochs 59 s.\n\n"
                "**Predict before running:** will the validation mean IoU rise at every one of the eight epochs, or peak "
                "and then flatten? Will the kept epoch be the last one?"
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
                "reset_to_pretrained()\n"
                "if pipe.adapter is not None:\n"
                "    raise RuntimeError('Section 7 must start from the pretrained decoder; restart the session and choose Run all.')\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, epochs=EPOCHS, lr=LEARNING_RATE, batch_size=BATCH_SIZE, threshold=THRESHOLD, progress=report)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "print({{'threshold': adapt_result['threshold'], 'trainable_parameters': adapt_result['n_trainable'], 'total_parameters': adapt_result['n_total'], 'extract_layers': adapt_result['extract_layers'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'loss': adapt_result['loss'], 'cache_seconds': adapt_result['cache_seconds'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "**What to notice:** epoch 0 labelled `frozen model` at about 0.60 validation mean IoU, the loss falling, and "
                "`best_epoch`.\n\n"
                "<details><summary>Check your reasoning</summary>It peaks and then flattens. In the Kaggle T4 run of "
                "2026-09-21 the validation mean IoU rose from 0.603 at epoch 0 to **0.855 at epoch 4** and stayed on a "
                "plateau of 0.849–0.855 afterwards, while the training loss kept falling from about 0.183 to 0.060; the "
                "earliest best epoch, 4, was kept, not the last. A falling training loss with a flat validation score is "
                "the signal that further epochs fit the 600 training records rather than the task. A local CPU run of this "
                "version (2026-10-04) printed the same curve and kept epoch 4 again (0.603 → 0.855, the same values to three decimals).</details>\n\n"
                "## 8. Held-out evaluation\n\n"
                "The test records were never used for training or epoch selection, and no image appears in two splits. The "
                "adapted model is scored exactly as the frozen model was in Section 6 and the four systems are put side by "
                "side. Read it in this order: **mean IoU** first (the measure the epoch was selected on), then **Dice**, then "
                "pixel **precision** and **recall** together (a gain in one at the cost of the other is a moved threshold, "
                "not a better segmenter), then the predicted area against the reference area. The cell prints a verdict — "
                "whether the adapted model beat the frozen one and both baselines — instead of stopping: on your own data or "
                "with other settings the honest answer can be *no*, and the export and reload in Section 9 still run. It also "
                "adds a row to `run_history`, so an optional experiment can be compared with the default run. One hundred and "
                "forty records from one seeded split give **no dispersion estimate**; the deltas are sample-sanity evidence "
                "that the adaptation contract works, not a benchmark, and a result on one food dataset's largest-ingredient "
                "masks says nothing about other phrases, other images or your data until you measure them.\n\n"
                "**Predict before running:** will the adapted model's held-out mean IoU land nearer the validation peak or "
                "nearer the frozen 0.637? Which will move more, precision or recall?"
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
                "improved = adapted_test['miou'] > frozen_test['miou']\n"
                "beats_baselines = adapted_test['miou'] > max(baseline_empty['miou'], baseline_full['miou'])\n"
                "run_history = globals().get('run_history', [])\n"
                "run_history.append({{'data_source': data_source, 'threshold': THRESHOLD, 'epochs': EPOCHS, 'learning_rate': LEARNING_RATE, 'split_seed': SPLIT_SEED, 'best_epoch': adapt_result['best_epoch'], 'frozen_miou': round(frozen_test['miou'], 3), 'adapted_miou': round(adapted_test['miou'], 3), 'delta_miou': round(adapted_test['miou'] - frozen_test['miou'], 3)}})\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json', 'adapted_beats_frozen': improved, 'adapted_beats_both_baselines': beats_baselines, 'verdict': 'improved on the held-out records' if improved and beats_baselines else 'not improved: read the epoch history and the per-record rows before trusting this adapter'}})\n"
                "for row in run_history:\n"
                "    print({{'run_history': row}})"
            ),
        },
        {
            "md": (
                "**What to notice:** the four-system table, the `delta_vs_frozen` row, the predicted area against the "
                "reference area, and the verdict.\n\n"
                "<details><summary>Check your reasoning</summary>Near the validation peak, and recall moves most. In the Kaggle "
                "T4 run of 2026-09-21 the held-out mean IoU went from 0.637 to **0.842** (Dice 0.715 → 0.904), close to the "
                "0.855 validation peak; precision rose modestly (0.835 → 0.881) while recall jumped (0.736 → 0.927), and the "
                "predicted area moved from 0.25 to 0.29 of the image against the references' 0.28. The records under 0.2 IoU "
                "fell from 24 to 1 and those above 0.8 rose from 63 to 111. The adapted decoder learned to draw FoodSeg103's "
                "masks as large as its annotators did. A local CPU run of this version (2026-10-04) printed the same 0.637 → 0.842 mean IoU and 0.715 → 0.904 Dice.</details>\n\n"
                "## 9. Look at the masks, segment the scene again, export the adapter and reload it\n\n"
                "Six held-out records are written as panels (`outputs/{stem}_examples/`: the photograph with the reference "
                "mask, the frozen mask and the adapted mask side by side — green, blue and red overlays — the phrase and both "
                "IoUs beneath) so the numbers can be checked by eye; the frozen masks are the ones Section 6 kept. The drawn "
                "scene from Section 5 is then segmented again by the adapted model — the decoder that was tuned serves every "
                "phrase, so this is a small look at what the adaptation did *outside* its phrase vocabulary and its corpus. "
                "In the Kaggle T4 run of 2026-09-21 the drawn shapes scored `green grass` 0.96, `a red circle` 0.96, `a blue "
                "square` 0.96, `a yellow triangle` 0.91 (mean IoU 0.947) before adaptation and 0.97, 0.97, 0.96, 0.93 (mean "
                "IoU 0.958) after, the absent phrases' area fraction 0.000 both times — one drawing of evidence, not a "
                "measurement.\n\n"
                "`pipe.save_artifact` writes the trained tensors — the decoder, about 4.5 MB in float32 — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and revision, "
                "the digest of the base `model.safetensors`, the tensor names, the file size and SHA-256, the threshold the "
                "epoch was selected at, the training configuration and the epoch history (OUT8). "
                "`ClipSegSegmentationPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest, its "
                "digest and its exact tensor set **before** deserialising, refuses any tensor outside the decoder, and overlays "
                "the tensors onto a freshly loaded base — a new object from files, not the in-memory model (VER2). The cell "
                "compares the masks of up to eight test records and stops with the cause if any differ (VER4). For a BYOD run "
                "`result.json` records your upload's name, size and SHA-256 instead of the FoodSeg103 corpus block.\n\n"
                "**Predict before running:** in the panels, where will the frozen (blue) and adapted (red) masks differ most — "
                "at the edges of an ingredient, or on whole ingredients the frozen model missed? Will the adapted decoder still "
                "leave `a cat` and `the sky` empty on the drawing?"
            ),
            "code": (
                "import shutil\n\n"
                "examples_dir = Path('outputs/{stem}_examples')\n"
                "shutil.rmtree(examples_dir, ignore_errors=True)\n"
                "examples_dir.mkdir(parents=True)\n"
                "caption_font = ImageFont.load_default(size=18)\n"
                "adapted_items = pipe.segment_batch([(r['image'], r['prompt']) for r in test_records[:6]], threshold=THRESHOLD)\n"
                "for record, frozen_row, adapted_row, adapted_item, frozen_mask in zip(test_records[:6], frozen_test['rows'][:6], adapted_test['rows'][:6], adapted_items, frozen_example_masks, strict=True):\n"
                "    thumb = record['image'].convert('RGB')\n"
                "    scale = 320 / max(thumb.size)\n"
                "    size = (max(1, round(thumb.width * scale)), max(1, round(thumb.height * scale)))\n"
                "    small = thumb.resize(size)\n"
                "    ref_small = np.asarray(Image.fromarray(record['mask'].astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127\n"
                "    fro_small = np.asarray(Image.fromarray(frozen_mask.astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127\n"
                "    ada_small = np.asarray(Image.fromarray(adapted_item['mask'].astype(np.uint8) * 255).resize(size, Image.NEAREST)) > 127\n"
                "    panels = [overlay(small, ref_small, (60, 179, 75)), overlay(small, fro_small, (40, 70, 200)), overlay(small, ada_small, (220, 40, 40))]\n"
                "    sheet = Image.new('RGB', (sum(p.width for p in panels) + 8 * (len(panels) - 1), panels[0].height + 56), (255, 255, 255))\n"
                "    x = 0\n"
                "    for panel in panels:\n"
                "        sheet.paste(panel, (x, 0))\n"
                "        x += panel.width + 8\n"
                "    marker = ImageDraw.Draw(sheet)\n"
                "    marker.text((8, panels[0].height + 6), f\"{{record['prompt']}} — reference (green) | frozen (blue) | adapted (red); IoU frozen {{frozen_row['iou']:.3f}} -> adapted {{adapted_row['iou']:.3f}}\", fill=(20, 20, 20), font=caption_font)\n"
                "    sheet.save(examples_dir / f\"{{record['id']}}.png\")\n"
                "print({{'examples': sorted(p.name for p in examples_dir.iterdir()), 'panels': ['reference overlay', 'frozen overlay', 'adapted overlay']}})\n\n"
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
                "if parity['identical_masks'] != parity['of']:\n"
                "    raise RuntimeError(f'Reload parity failed: {{parity}}. The saved adapter does not reproduce the evaluated model; re-run from Section 7 or restart the session and choose Run all.')\n"
                "reloaded = None\n\n"
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
                "    'run_history': run_history,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'pillow': PIL.__version__, 'device': pipe.device, 'dtype': 'float32'}},\n"
                "}}\n"
                "if byod_upload is not None:\n"
                "    # A BYOD run records the upload, not the FoodSeg103 sample it did not use.\n"
                "    del result_payload['corpus']\n"
                "    result_payload['byod_upload'] = byod_upload\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
        {
            "md": (
                "**What to notice:** six three-panel sheets, the adapted scene line, a 64-tensor adapter of about 4.5 MB, and "
                "`reload_parity` 8 of 8.\n\n"
                "<details><summary>Check your reasoning</summary>Mostly on whole regions the frozen model left out: the "
                "frozen masks are too small (recall 0.736), and adaptation mainly grows them to the annotated extent (recall "
                "0.927), with comparatively small changes at the edges. On the drawing the adapted decoder still leaves `a "
                "cat` and `the sky` empty and moves the shapes by about a hundredth (0.947 → 0.958 mean IoU in the Kaggle T4 "
                "run of 2026-09-21): 1.1 M decoder parameters learned FoodSeg103's annotation convention, not new visual "
                "concepts. The adapter held 64 tensors (4,514,484 bytes) and the reloaded pipeline gave identical masks on "
                "8 of 8 test records.</details>\n\n"
                "## 10. Your turn — change one thing: the threshold\n\n"
                "Optional, and it runs after the default path without changing it: this cell reads the **adapted** model's "
                "masks on the 60 **validation** records at two thresholds — `THRESHOLD` (the one every earlier number used) and "
                "`ACTIVITY_THRESHOLD` — and writes `outputs/{stem}_threshold_activity.json`. It does not train, does not touch "
                "the test records and does not change the exported adapter. **Predict → Change → Run → Observe → Explain:**\n\n"
                "1. **Predict:** at `ACTIVITY_THRESHOLD = 0.3` (below the default 0.5), will pixel precision rise or fall? "
                "Pixel recall? Will the mean IoU move by more or less than 0.05?\n"
                "2. **Change:** the field is already set to `0.3`. After the first run, set it to `0.7`.\n"
                "3. **Run:** run this cell only (it is safe to re-run on its own).\n"
                "4. **Observe:** the two rows of the table, and the `predicted_area_fraction` beside the reference area.\n"
                "5. **Explain:** in one sentence, why does lowering the threshold trade precision for recall? Why is the "
                "threshold chosen on validation records and never on the test records?\n\n"
                "**Predict before running:** write your guess for step 1 before you run the cell."
            ),
            "code": (
                "ACTIVITY_THRESHOLD = 0.3  # @param {{type:\"number\"}}\n\n"
                "if not 0.0 <= ACTIVITY_THRESHOLD <= 1.0:\n"
                "    raise ValueError('ACTIVITY_THRESHOLD must be between 0 and 1')\n"
                "activity = {{'model': 'adapted' if pipe.adapter is not None else 'frozen', 'records': 'validation', 'n': len(val_records), 'rows': {{}}}}\n"
                "for cut in sorted({{float(THRESHOLD), float(ACTIVITY_THRESHOLD)}}):\n"
                "    measured = pipe.evaluate(val_records, threshold=cut, batch_size=EVAL_BATCH_SIZE)\n"
                "    activity['rows'][str(cut)] = {{**{{k: round(measured[k], 3) for k in METRICS}}, 'predicted_area_fraction': round(measured['predicted_area_fraction'], 3)}}\n"
                "activity['reference_area_fraction'] = round(dataset_manifests['validation']['mask_area_fraction']['mean'], 3)\n"
                "with open('outputs/{stem}_threshold_activity.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(activity, handle, indent=2)\n"
                "print({{'model': activity['model'], 'records': activity['records'], 'n': activity['n'], 'reference_area_fraction': activity['reference_area_fraction']}})\n"
                "for cut, row in activity['rows'].items():\n"
                "    print({{'threshold': float(cut), **row}})"
            ),
        },
    ],
    "closing": (
        "**What to notice:** two rows, one per threshold; the lower threshold predicts a larger area.\n\n"
        "<details><summary>Check your reasoning</summary>Precision falls, recall rises, and the mean IoU barely moves. In a local CPU run of this version (2026-10-04, default settings) the adapted model on the 60 validation records scored, at 0.5, mean IoU 0.855, precision 0.903, recall 0.911 and a predicted area of 0.298; at 0.3, 0.851, 0.871, 0.933 and 0.315; at 0.7, 0.844, 0.927, 0.873 and 0.282 — against a reference area of 0.298. The mean IoU moved by less than 0.02 either way, because after adaptation most of each mask lies far from the cut. Lowering the threshold admits every pixel whose "
        "sigmoid lies between the two cuts, so the masks grow: more of the reference is covered (recall up) and more "
        "background is marked too (precision down). Choosing the threshold on the test records would make the reported "
        "test score optimistic, because the test split would then have been used to tune the model; the validation "
        "split exists to make that choice.</details>\n\n"
        "## Interpretation and limits\n\n"
        "Start from your own run (Section 8's verdict and `run_history`): did the adapted decoder beat the frozen model "
        "and both baselines on the held-out records, and was the gain in precision, in recall, or both?\n\n"
        "<details><summary>Reference answer from the recorded runs</summary>A phrase-conditioned segmenter trained on "
        "PhraseCut already finds the largest ingredient in a food photograph roughly when asked by name: the frozen model "
        "scored a mean IoU of 0.637 on the FoodSeg103 test records. A bounded fine-tuning of its decoder on 600 records "
        "moved that to 0.842 mean IoU and 0.904 Dice in the Kaggle T4 run of 2026-09-21, which reproduced a CPU sweep "
        "exactly (0.842 / 0.904 at epoch 4, plateau 0.849–0.855 after), with a 4.5 MB adapter that reloads "
        "mask-for-mask. The gain is almost all recall (0.736 → 0.927 at precision 0.835 → 0.881), the records under 0.2 "
        "IoU fall from 24 to 1 and those above 0.8 rise from 63 to 111, one record (`garlic`) stays at 0.0, and the drawn "
        "scene outside the vocabulary moved by about a hundredth (0.947 → 0.958 mean IoU, the two absent phrases still "
        "empty): a decoder of 1.1 M parameters learned FoodSeg103's annotation convention, not new visual concepts. On "
        "BYOD data or with other settings your numbers will differ, and the verdict can be *not improved*.</details>\n\n"
        "That is the claim: the adaptation contract works end to end on a text-prompted segmenter with a real labelled "
        "set, and the numbers it produces are read as mean and micro IoU, Dice, precision and recall at one stated "
        "threshold, against two non-adapted baselines and the frozen model, with the predicted area beside them rather "
        "than in isolation. The row has no sibling on this corpus — CLIPSeg is the fleet's only text-prompted segmenter — "
        "so the comparison is the frozen decoder against the tuned one on the same 140 records.\n\n"
        "The test split is 140 records from one seeded draw of one 800-record sample, the validation split that picks the "
        "epoch is 60, and every rate in Sections 5–9 is at the one threshold `THRESHOLD` — not a benchmark, not a "
        "threshold sweep, not a measure of phrases the sample never asks (one phrase per image, the largest ingredient "
        "only). So a result here says the contract works on food photographs' largest ingredients, not that the adapted "
        "model handles other phrases, other image families or your masks. The decoder that was tuned serves every phrase: "
        "the drawn scene re-segmented in Section 9 is one drawing of evidence about what the tuning did outside its "
        "vocabulary, not a measurement, and a deployment that segments other phrases must measure them after adapting. The "
        "towers were not adapted: what the image encoder cannot see stays unsegmented, and **the probabilities remain an "
        "uncalibrated sigmoid**.\n\n"
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
        "**Optional experiments (they do not affect the default path).** Each names the field to change and the cell to "
        "re-run from with **Runtime → Run after**; Sections 5, 6 and 7 reload the pretrained model first, so the frozen "
        "rows and epoch 0 are always the untouched model's, and Section 8 adds a row to `run_history` to compare with the "
        "default run.\n\n"
        "- **More epochs:** set `EPOCHS = 16` in Section 7 and **Run after** from Section 7. Does validation keep a later "
        "epoch? <details><summary>Reference answer</summary>Inferred from the default run's plateau (0.849–0.855 after "
        "epoch 4 on the Kaggle T4): the earliest best epoch stays near 4 unless a later epoch edges above 0.855, and the "
        "held-out gain barely changes; not re-measured at 16 epochs.</details>\n"
        "- **Learning rate:** set `LEARNING_RATE = 3e-5` or `3e-3` in Section 7 and **Run after** from Section 7. Read the "
        "epoch-0 row (it must equal the frozen 0.603 again) and the curve.\n"
        "- **Threshold for every stage:** set `THRESHOLD = 0.3` or `0.7` in Section 5 and **Run after** from Section 5. "
        "Read how precision and recall trade against each other for both the frozen and the adapted model; the adapter is "
        "then selected and saved at that threshold.\n"
        "- **Another split:** set `SPLIT_SEED` in Section 4 and **Run after** from Section 4. How much do 140 records move "
        "the frozen and adapted numbers?\n"
        "- **Your own masks:** BYOD (Section 4, then **Run after**); read the two baselines and the frozen row before the "
        "adapted number.\n\n"
        "## Troubleshooting\n\n"
        "- **Section 1 stops with \"needs a Linux x86_64 runtime\".** The locked environment is built from manylinux "
        "wheels. Use Google Colab, Kaggle or a Linux Jupyter server; Windows and macOS kernels are not supported.\n"
        "- **Section 1 fails while downloading** (`uv` wheel, Python build or packages). The runtime needs PyPI and the "
        "python-build-standalone download; run the cell again once the network is back. A \"size/SHA-256\" or \"does not "
        "match its digest\" error means a file was altered: do not edit the cell, regenerate the notebook from the "
        "repository.\n"
        "- **A restart prompt.** This notebook never needs a runtime restart: nothing is installed into the kernel. If "
        "the runtime restarted anyway (you chose it, or it was reset), choose **Run all** again from the top; files "
        "already downloaded and verified are reused while they are still there.\n"
        "- **\"The isolated environment's Python process exited\".** Usually the runtime ran out of memory: the activation "
        "cache holds about 1.3 GB in host memory for 600 records. Restart the session and choose **Run all**; on a small "
        "CPU runtime close other notebooks first.\n"
        "- **No GPU.** The notebook runs on CPU with the same arithmetic, only slower (Prerequisites). In Colab choose "
        "**Runtime → Change runtime type → T4 GPU** before **Run all**.\n"
        "- **`FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3.** A staged file is "
        "incomplete or altered — delete it from `weights/clipseg-rd64-refined/` and run Section 3 again.\n"
        "- **A `sha256` `ValueError` naming a parquet row group in Section 4.** A cached "
        "`weights/foodseg103/validation-rg*.parquet` is incomplete — delete it and run Section 4 again.\n"
        "- **BYOD is refused in Section 4.** The message names what to fix: fewer than 12 distinct images, a missing "
        "`masks.csv` or column, a path `masks.csv` names that is not in the upload (paths are relative to the folder "
        "holding `masks.csv`), an image and mask that are the same file, two zip members with the same path, a file no "
        "row names, a JPEG mask, a mask of the wrong size or with no pixel, a phrase over 64 characters or a duplicate id. "
        "Fix the set and run Section 4 with **Run after** again. An empty or cancelled upload asks you to choose one file "
        "or set `BYOD_PATH`.\n"
        "- **Section 8 says \"not improved\".** The adapted decoder did not beat the frozen model on your test records: "
        "read the epoch history (did validation keep epoch 0?) and the per-record rows in the evaluation report.\n"
        "- **Reload parity fails in Section 9.** The saved adapter does not reproduce the evaluated model. Re-run from "
        "Section 7 (it reloads the pretrained model first), or restart the session and choose **Run all**.\n"
        "- **Numbers differ slightly from the recorded runs.** GPU arithmetic is not bit-for-bit the CPU's; a difference "
        "in the third decimal between a T4 and a CPU run is expected. Larger differences at the default settings mean a "
        "form field was changed.\n\n"
        "## Glossary\n\n"
        "- **Phrase (prompt):** the free text that says what to segment; CLIPSeg encodes it with the CLIP text tower.\n"
        "- **Sigmoid / probability map:** the per-pixel output squashed into 0..1; *uncalibrated* means 0.7 is not a 70 % "
        "chance.\n"
        "- **Threshold:** the cut that turns the probability map into a mask (pixel ≥ threshold); a caller-owned choice.\n"
        "- **IoU (intersection over union):** overlap of the predicted and reference masks divided by their union; "
        "**mean IoU** averages it over records, **micro IoU** pools every pixel of the split.\n"
        "- **Dice:** 2 × overlap / (predicted + reference pixels); higher is better, like IoU.\n"
        "- **Pixel precision / recall:** of the marked pixels, the share that is reference; of the reference pixels, the "
        "share that is marked.\n"
        "- **Empty-mask / full-mask baseline:** predict nothing / everything; the floor and the \"somewhere in the "
        "photograph\" score.\n"
        "- **CLIP towers:** the frozen image and text encoders; **decoder:** the small trained part that turns their "
        "activations into the logit map, conditioned on the phrase through **FiLM** (a learned scale and shift).\n"
        "- **Frozen-tower cache:** the towers' outputs computed once and reused at every epoch, because the towers do not "
        "change.\n"
        "- **Frozen model:** the pretrained decoder, before any training here.\n"
        "- **Epoch / epoch 0:** one pass over the training records; epoch 0 is the frozen model, scored before training "
        "so it can win.\n"
        "- **Training / validation / test split:** records used to train, to choose the epoch, and to report the final "
        "result once — split by image.\n"
        "- **Adapter / reload parity:** the saved decoder tensors, and the check that a pipeline rebuilt from them gives "
        "the same masks.\n\n"
        "## Conclusion (your notes)\n\n"
        "Optional personal notes, not a required submission. Complete each sentence from your own run:\n\n"
        "- On these test records the empty mask scored mean IoU ___, the full mask ___, the frozen model ___ and the "
        "adapted model ___ (best epoch ___).\n"
        "- The adaptation moved precision from ___ to ___ and recall from ___ to ___, so it mainly ___.\n"
        "- At threshold ___ (Section 10) the validation masks covered ___ of the image against the references' ___, "
        "because ___.\n"
        "- Before trusting an adapted segmenter on my own images I would measure ___ on ___.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/timojl/clipseg\n"
        "- Image Segmentation Using Text and Image Prompts (Lüddecke and Ecker, CVPR 2022): https://arxiv.org/abs/2112.10003\n"
        "- FoodSeg103 (Apache-2.0): https://huggingface.co/datasets/EduardoPacheco/FoodSeg103 — Wu, Fu, Liu, Lim, Hoi, Sun. A Large-Scale Benchmark for Food Image Segmentation (ACM MM 2021): https://arxiv.org/abs/2105.05409\n"
        "- DIMER Notebook Specification 2.2 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
