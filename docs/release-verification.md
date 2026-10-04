# Release verification

`tutorials/clipseg_segmentation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.2 (REL8). This file is the durable release-gate record for the
notebook. The earlier `TASK-INFERENCE` notebook's Kaggle CPU run (2026-09-14, retained below) is history for a
superseded blob, not evidence for this one.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one generated tutorial notebook (the WORKSHOP notebook is declared and checked separately; see the workshop section below), named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.2`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; exactly two kernel cells — the isolated-environment
  install (pinned `uv` wheel checked by size and SHA-256, managed CPython 3.12.12, the carried hash lock
  `tutorials/requirements-colab.lock.txt` installed with `--require-hashes --only-binary :all:`, Linux x86_64 only) and
  the router that runs every later cell there; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the FoodSeg103
  parquet-conversion revision `176acc3edd2432ee126bda6fb01469eadeb018df` is the one other 40-hex revision the
  documents may cite);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ClipSegSegmentationPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus`, `build_sample_dataset(corpus, seed=SPLIT_SEED)` / the notebook's `load_byod_records` (relative-path
  keys, collision and same-file refusals, the 12-image floor checked before splitting, `BYOD_PATH`) + `split_dataset`,
  `validate_dataset` per split (training `MIN_RECORDS`, validation and test a minimum of one), `check_split_disjoint`, `write_dataset_csv`, the four dataset refusal probes, the
  ceiling print, `validate_inputs` with the duplicate-phrase refusal probe, `segment` on the drawn scene with the
  structural checks and the `evaluation_report` against the drawn masks, `empty_baseline`, `full_baseline`,
  `reset_to_pretrained()` before Sections 5, 6 and 7 with Section 6 refusing an adapted model, `pipe.evaluate` on the
  frozen model, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the
  validation and test splits after adaptation with a printed verdict and `run_history` (no result-dependent assertion), the drawn scene re-segmented after
  adaptation, the reference / frozen / adapted example panels, `pipe.save_artifact`,
  `ClipSegSegmentationPipeline.from_artifact` and the mask-parity check that raises with its cause, the result fields
  `weight_file` / `weight_format` / `weight_sha256`, the `corpus` block replaced by `byod_upload` on a BYOD run, and the
  Section 10 threshold activity), the eight expected `outputs/` paths, the guided layer (audience, input → model →
  output, how to use, roadmap, at least seven predictions with worked answers, the change-one-thing activity,
  troubleshooting, glossary, conclusion template, three Infrastructure labels), a list of stale learner-facing text
  that must not return (the restart instruction, the 0.86 scene value, the eight-image BYOD minimum, the two-minute
  estimate), no bare `assert` in learner cells, the learner-facing statements (Apache-2.0 weights, the uncalibrated
  sigmoid, the caller-owned threshold, adaptation of the decoder on labelled records, the two non-adapted baselines,
  mean and micro IoU, Dice, pixel precision and recall, the empty-mask and full-mask baselines, the per-pixel binary
  cross-entropy, the frozen-tower cache, highest validation mean IoU, no dispersion estimate, float32 on every
  device, the leakage guidance, the precision-and-recall guidance, the excluded instance/panoptic scope, the snapshot
  note, the troubleshooting section) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git
  clone` / `github.com/kurtvalcorza` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `CLIPSegForImageSegmentation` / `CLIPSegProcessor` / `torch.sigmoid(` /
  `torch.inference_mode(` / `from huggingface_hub import` / `urllib.request` / `pyarrow` / `safetensors` imports /
  `torch.optim` / `.backward(` / `requires_grad` / `pipe._model` / `pipe._processor` / `extractall(` use **outside the
  carried module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(`, `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI also installs the pinned CPU-only torch wheel plus `transformers`, `huggingface-hub`, `safetensors`, `numpy`,
`pillow` and `pyarrow`, the package with `--no-deps`, runs `ruff check src tests tools`, `tools/build_notebook.py
--check`, and the unit suite (`tests/`, including `test_adaptation.py`, `test_import_boundary.py`,
`test_role_helpers.py`, `test_notebook_parity.py`; injected runner and parquet opener, no weights —
`tests/test_model_backed.py` is skipped without the snapshot). These are source/provenance and unit checks. They are
**not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU or GPU runtime (CUDA used automatically when present) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; promotion evidence |
| Kaggle script kernel (pre-flight only) | Fresh GPU container that clones the candidate branch, installs the pins and runs `tests/test_model_backed.py` plus the package-API recipe probe | Builder pre-flight to catch defects and fix the recipe before spending a notebook run; **not** promotion evidence for the notebook blob |
| Local Windows-venv CPU run (pre-flight only) | Workstation venv `dimer-next16`, `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1` | Builder pre-flight of the package API on the cached row groups (the recipe sweep and the model-backed suite); **not** a supported runtime and not promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/clipseg-rd64-refined/` or the row-group cache `weights/foodseg103/` (the standalone path writes
   the manifest itself, stages the missing files from the Hub and reads the pinned row groups over range requests,
   so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `BYOD_PATH = ''`, `SPLIT_SEED = 42`, `THRESHOLD = 0.5`, `EPOCHS = 8`, `LEARNING_RATE = 3e-4`,
   `BATCH_SIZE = 8`, `ACTIVITY_THRESHOLD = 0.3`) on a Linux x86_64 runtime;
4. verify that Section 1 builds the isolated environment (the printed dictionary names the isolated Python 3.12.12
   and the locked package count), that the runtime-record cell reports `NOTEBOOK_SOURCE.repository_revision` equal
   to the revision recorded in `metadata.dimer.generated_from` and imported versions equal to the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`, `pillow==11.3.0`, `pyarrow==25.0.1`, and that the whole notebook completes in **one pass with no
   restart** — a run that needed a restart is not a one-pass `Run all` and is not promotion evidence (REL11);
5. verify every default-path stage completes:
   - the locked runtime installed into the isolated environment from the carried hash lock with no GitHub access;
   - the three carried module cells execute (defining `ClipSegSegmentationPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `format_prompts`, `mask_iou`, `mask_bbox`,
     `segmentation_metrics`, `mask_metrics`, `empty_baseline`, `full_baseline`, `fetch_corpus`, `read_corpus`,
     `largest_class`, `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv`, the class vocabulary and the ceilings) with no import of the
     repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting all 8 manifest entries fetched from `CIDAS/clipseg-rd64-refined` at the
     immutable revision on a clean runtime, `verify_snapshot` returning its dict (8 files, the 603 MB
     `model.safetensors` re-hashed), and `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified
     directory in float32;
   - Section 4: `fetch_corpus` reading the eight pinned row groups over HTTPS range requests with every SHA-256 and
     byte total matching (800 images, about 43 MB), `read_corpus` making 800 records; the seeded split into
     600 / 60 / 140 with `check_split_disjoint` reporting no shared image and the three dataset digests printed;
     `outputs/…_train.csv` and `outputs/…_example_record.png` written; the four dataset refusal probes each raising
     `ValueError`;
   - Section 5: the ceilings surfaced; the drawn scene rendered; the input manifest written to
     `outputs/…_input_manifest.json` (verdict `accepted`, one recorded rejection finding from the duplicate-phrase
     probe); the six phrases segmented with every structural check `True`, `outputs/…_scene_frozen.json` and `.png`
     written and the `evaluation_report` verdict `sample-sanity` (the inference-only card recorded `miou` 0.947 on
     the four drawn shapes — an observation, not an assertion);
   - Section 6: the empty-mask baseline (mean IoU 0.0 exactly), the full-mask baseline (≈ 0.283) and the
     frozen model's test rates (≈ 0.637 mean IoU / 0.715 Dice in the Tesla T4 build record —
     the frozen decoder already finds most dishes — 63 of the 140 held-out records score above 0.8 IoU — but misses 24 almost entirely (IoU under 0.2), typically ingredient names it does not ground at all (`pie`, `lamb`, `garlic`, `shellfish` at 0.0), with precision 0.835 well above recall 0.736: it under-segments) with four records' scores printed under their phrases;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 1,127,009 trainable of 150,747,746 parameters,
     and an eight-epoch history with the validation mean IoU rising (build record: 0.603 → 0.822 / 0.840 / 0.851 / 0.855 / 0.849 / 0.854 / 0.851 / 0.853, `best_epoch`
     4);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, the predicted area
     and `outputs/…_evaluation_report.json` written; the printed verdict (adapted against frozen and both baselines —
     0.842 against 0.637 in the build record, Dice 0.715 → 0.904) and one `run_history` row;
   - Section 9: six three-panel example sheets (reference, frozen, adapted) under `outputs/…_examples/`; the drawn scene re-segmented by the adapted model
     with `outputs/…_scene_adapted.json` and `.png` (build record: before adaptation `green grass` 0.96, `a red circle` 0.96, `a blue square` 0.96, `a yellow triangle` 0.91 (mean IoU 0.947); absent phrases' area fraction `a cat` 0.000, `the sky` 0.000; after adaptation `green grass` 0.97, `a red circle` 0.97, `a blue square` 0.96, `a yellow triangle` 0.93 (mean IoU 0.958); absent phrases' area fraction `a cat` 0.000, `the sky` 0.000 — a recorded observation, not an
     assertion); `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (the decoder,
     about 4.5 MB) and `ClipSegSegmentationPipeline.from_artifact` reloading it with 8/8 identical masks on eight
     test records (the cell raises with the cause if any differ); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity
     and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block, the
     inference-contract records before and after adaptation, the comparison, the artifact digest, the reload
     parity, `run_history`, the runtime versions, device and dtype;
   - Section 10: the adapted model's validation rates at `THRESHOLD` and `ACTIVITY_THRESHOLD` and
     `outputs/…_threshold_activity.json`;
6. verify the exports exist and the interpretation section matches the observed path; then, in the same session,
   exercise the reuse journeys (REL12): set `USE_BYOD = True` with a representative set of at least 12 distinct images
   and **Run after** from Section 4 (Section 6 must print `adapted: False` and the frozen rows of a freshly loaded
   model; the run must reach Section 9's reload parity and `result.json` must carry `byod_upload` and no `corpus`),
   and once with 11 images (refused in Section 4, naming the count and the minimum);
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the row-group cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `clipseg_segmentation_colab.ipynb` (`E2E`, previous blob) | `2c999ae` / `100adc4f` | 2026-09-21 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-clipseg-segmentation` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | **Not a one-pass `Run all`** — attempt 1 stopped in the install cell with the restart `RuntimeError` (`cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`) after 237.2 s; 11/11 code cells ok on attempt 2 after a manual restart (198.1 s). Not promotion evidence (REL11; review CLS-M1, 2026-10-02); the metrics below are observations of that run: 26 files, 647 MB staged from the Hub into a clean cache; comparison {miou: {empty: 0, full: 0.283, frozen: 0.637, adapted: 0.842}, iou_micro: {empty: 0, full: 0.254, frozen: 0.642, adapted: 0.824}, dice: {empty: 0, full: 0.426, frozen: 0.715, adapted: 0.904}, pixel_precision: {empty: 0, full: 0.254, frozen: 0.835, adapted: 0.881}, pixel_recall: {empty: 0, full: 1, frozen: 0.736, adapted: 0.927}, delta_vs_frozen: {miou: 0.205, iou_micro: 0.181, dice: 0.189, pixel_precision: 0.046, pixel_recall: 0.191}, area: {reference_pixels: 11411484, frozen_predicted_pixels: 10047557, adapted_predicted_pixels: 12006659, frozen_predicted_area_fraction: 0.248, adapted_predicted_area_fraction: 0.291}}; drawing / scene / page check frozen vs adapted {frozen: {miou: 0.947}, adapted: {miou: 0.958}}; reload parity {identical_masks: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-clipseg-segmentation/v3/evidence/` in the workspace |
| `clipseg_segmentation_colab.ipynb` (`TASK-INFERENCE`, superseded) | `69dc7ee` / `2fd1160bfd0d` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-clipseg-segmentation` v1) | PASSED — 8/8 code cells, 18 files, 605 MB staged, 224.8 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/clipseg_segmentation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/clipseg_segmentation_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-04 | `a110af3` (`a110af347943da9042ee60fd20116249cd6f4280`, PR #10 head) / blob `fce863b3` (`fce863b3a4cd35e247c9746d2466152e61beb0a0`; downloaded from GitHub at that commit and blob-verified before the session). Evidence under `docs/execution-evidence/2026-10-04/`: executed notebook `clipseg_segmentation_colab_a110af3_colab-cli-t4.ipynb` SHA-256 `d3dc5545aaba8afe29857255eb262f2bb1b799263bf0f90ccd11307dfceaa10a`, `clipseg_segmentation_colab_a110af3_exec.log` SHA-256 `e45e39cd9884fe3105f0046ccb79a4f949e043ebadce5bac31f70a920247c644`, `clipseg_segmentation_colab_a110af3_run_summary.json` SHA-256 `053a4818d2f0c12ab0ddec98cb2630896de8cd5e3c5bcc408ba6f2c8144f5fcc` | Colab CLI 0.7.4 sequential execution, fresh Colab **Tesla T4** VM, via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`; session started 2026-10-04T01:16:15Z and stopped after the run). Isolated `uv` Python 3.12.12 environment (48 locked packages, built in 50 s; kernel Python 3.13.15); `torch 2.14.0+cu130`, `transformers 4.57.6`, `PIL 11.3.0`, `cuda:0`, float32; `repository_revision` `58e7cd95` | Default path only, form parameters at their defaults, no repository checkout, clean runtime (all 8 snapshot files fetched from the Hub at `999e0328` and re-verified; the eight row groups read over range requests, 800 records, 42,297,825 bytes). Not a browser `Run all`: the CLI records no execution counts, so order is evidenced by `exec.log` (`Executing cell 1/14` … `14/14` in order). Not exercised: the BYOD journeys (12 and 11 images), the rerun, the results download | 212.5 s CLI execution wall | **One pass, no restart, 0 errors** — 14/14 code cells; cells 4–6 (the three carried module definitions) print nothing by design. Split 600 / 60 / 140; four dataset refusal probes rejected; drawn scene frozen mean IoU 0.947 (`sample-sanity`). Test: empty 0.0; full 0.283 / Dice 0.426; frozen 0.637 mean IoU / 0.642 micro / 0.715 Dice / precision 0.835 / recall 0.736; adapted 0.842 / 0.824 / 0.904 / 0.881 / 0.927 (Δ mean IoU +0.205). Validation mean IoU by epoch 0.603 → 0.822 / 0.840 / 0.851 / 0.855 / 0.849 / 0.854 / 0.851 / 0.853, `best_epoch` 4; 1,127,009 trainable of 150,747,746 parameters. Drawn scene after adaptation 0.958; adapter 64 tensors, 4,514,484 bytes; reload parity 8/8 identical masks. Validation at `ACTIVITY_THRESHOLD` 0.3: mean IoU 0.851, at 0.5: 0.855. **Compared with the 2026-09-21 Kaggle T4 run of `100adc4f`:** every test rate, delta, area count, drawn-scene score and the reload parity are equal; the only stderr lines are the upstream slow-image-processor and ignored-`padding` warnings. Status stays **Candidate** |
| 2026-10-04 | review-fix head of PR #10 / blob `fce863b3` (pre-flight; the executed copy's code cells are identical to this blob, four markdown cells then received the numbers this run produced) | Windows venv `dimer-next16` (`torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12.10, `cpu`, float32, 24 threads), `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`; the two kernel cells (isolated install and router) skipped and the runtime-record cell run with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`; snapshot and row groups pre-staged (re-hashed by the notebook) | every learner cell verbatim from the notebook JSON: (a) default path Sections 4–10; (b) then **Run after** from Section 5 with the fields unchanged and Section 7 at `EPOCHS = 1`; (c) a separate session: BYOD by `BYOD_PATH`, an `images/` + `masks/` zip with a macOS `__MACOSX/` entry and `.DS_Store`, 11 then 12 distinct images, Sections 4–10 at `EPOCHS = 1` | (a) Section 7 337.1 s (tower cache 131.6 s); (b) 582.5 s session in all; (c) 19.4 s | Pre-flight, not promotion evidence. (a) 600 / 60 / 140; scene 0.947; empty 0.000, full 0.283, frozen 0.637 / Dice 0.715 / precision 0.835 / recall 0.736 (`adapted: False`); validation 0.603 → 0.822 / 0.840 / 0.851 / **0.855** / 0.848 / 0.853 / 0.851 / 0.854, epoch 4 kept; adapted 0.842 / 0.904 / 0.881 / 0.927; scene after 0.958; panels 976 px wide (three overlays); 64 tensors, 4,514,484 bytes; reload parity 8/8; activity on validation: 0.5 → mean IoU 0.855, precision 0.903, recall 0.911, area 0.298; 0.3 → 0.851 / 0.871 / 0.933 / 0.315; 0.7 → 0.844 / 0.927 / 0.873 / 0.282 (reference area 0.298). (b) the reload message printed, scene 0.947 again, Section 6 frozen rates equal to (a) within 1e-6 (identical) with `adapted: False`, Section 7 epoch 0 0.6033 = (a)'s. (c) 11 images refused in Section 4: `Your upload holds 11 records (11 distinct images); BYOD needs at least 12 distinct images …`; 12 images accepted (2 OS-metadata files ignored, images distinct from their masks), split 8 / 2 / 2, through Section 9 with reload parity 2/2 and Section 10; validation kept epoch 0 and Section 8 printed *not improved* without stopping; `result.json` had no `corpus` and `byod_upload` {name, bytes 12268, sha256 equal to the zip's}. The uv isolated-environment path itself is Linux-only and was not executed here |
| 2026-09-21 | `d7b8652` / `c74a3b90` (pre-flight: the build-record placeholders still unfilled in the prose, code identical) | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-clipseg-segmentation` v2) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout | 302.6 s | Pre-flight; 11/11 code cells ok only after a manual restart following the install cell — not a one-pass `Run all`; 26 files, 647 MB staged; the metrics the build record quotes |
| 2026-09-21 | `2c999ae` / `100adc4f` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-clipseg-segmentation` v3; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`, float32) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 435.4 s (237.2 s attempt 1 + 198.1 s attempt 2) | **Not a one-pass `Run all`** — attempt 1 stopped in the install cell with the restart `RuntimeError` (`cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`) after 237.2 s; 11/11 code cells ok on attempt 2 after a manual restart (198.1 s). Not promotion evidence (REL11; review CLS-M1, 2026-10-02); the metrics below are observations of that run: 26 files, 647 MB staged from the Hub into a clean cache; comparison {miou: {empty: 0, full: 0.283, frozen: 0.637, adapted: 0.842}, iou_micro: {empty: 0, full: 0.254, frozen: 0.642, adapted: 0.824}, dice: {empty: 0, full: 0.426, frozen: 0.715, adapted: 0.904}, pixel_precision: {empty: 0, full: 0.254, frozen: 0.835, adapted: 0.881}, pixel_recall: {empty: 0, full: 1, frozen: 0.736, adapted: 0.927}, delta_vs_frozen: {miou: 0.205, iou_micro: 0.181, dice: 0.189, pixel_precision: 0.046, pixel_recall: 0.191}, area: {reference_pixels: 11411484, frozen_predicted_pixels: 10047557, adapted_predicted_pixels: 12006659, frozen_predicted_area_fraction: 0.248, adapted_predicted_area_fraction: 0.291}}; drawing / scene / page check frozen vs adapted {frozen: {miou: 0.947}, adapted: {miou: 0.958}}; reload parity {identical_masks: 8, of: 8}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-clipseg-segmentation/v3/evidence/` in the workspace |
| 2026-09-21 | package API at `d7b8652` (pre-flight, not the notebook blob) | Kaggle Tesla T4 script kernel (`kurtvalcorza/dimer-probe-clipseg-e2e` v3 — v1 was never run and v2 died after its green pytest on an import-path slip in the probe script, not in the row; `torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12, `cuda:0`, float32), branch cloned, pins installed, snapshot staged from the Hub | `tests/test_model_backed.py` (7 passed, 19 warnings in 43.48s) and the recipe probe: the eight pinned row groups read over range requests (800 records, digest match), empty and full baselines, frozen model on the 140 test records, `adapt(epochs=8, lr=3e-4, batch_size=8)` with validation-mIoU selection, adapted evaluation, artifact round trip | 398 s | **PASS** — 7 passed, 19 warnings in 43.48s; the notebook's 8 code cells re-executed through the package API in 130 s with peak CUDA memory 1.61 GB; the metrics it produced are the ones the notebook run above recorded (same seed, same split, same recipe) |
| 2026-09-20 | package API at the working tree of `feat/e2e-segmentation-adaptation` (pre-flight, not the notebook blob) | Windows venv `dimer-next16` (`torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12.10, `cpu`, float32), `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, row groups cached | the CPU recipe sweep on the default split (600 / 60 / 140): frozen 0.637 mean IoU (Dice 0.715, precision 0.835, recall 0.736; empty 0.000, full 0.283); 8 epochs, batch 8, validation mean IoU per epoch (epoch 0 = frozen 0.603) → test mean IoU / Dice at the kept epoch: lr 3e-5 → 0.777 … 0.820 (epoch 7 kept) → 0.804 / 0.875; lr 1e-4 → 0.813 … 0.844, still rising at epoch 8 (kept) → 0.831 / 0.896; **lr 3e-4 → 0.822, 0.840, 0.851, 0.855 (epoch 4 kept), then 0.848–0.854 plateau → 0.842 / 0.904**; lr 1e-3 → 0.848, 0.834, 0.850, 0.856, 0.858 (epoch 5 kept), 0.855, 0.849, 0.856 → 0.850 / 0.910. 3e-4 and 1e-3 are within noise of each other; 3e-4 plateaus by epoch 4 without the epoch-2 dip and is the default. Decoder-on-cache parity max abs 7.0e-4 (fp16 cache); artifact 4,514,484 bytes; reload parity 8/8 | ~19 min (tower cache 129–157 s per arm, adapt 307–369 s per arm incl. the eight validation passes, frozen test 35.8 s; the box was shared with another CPU job) | PASS — pre-flight only; fixed the recipe at lr 3e-4 × 8; not promotion evidence |
| 2026-09-14 | `69dc7ee` / `2fd1160bfd0d` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-clipseg-segmentation` v1) | Default sample path, `Run all` from a fresh interpreter, no repository checkout | 224.8 s | PASSED — 8/8 code cells, 18 files, 605 MB staged; not evidence for the `E2E` blob |

## Current status

**Candidate** — the `E2E` notebook `tutorials/clipseg_segmentation_colab.ipynb` was regenerated on 2026-10-04 for the 2026-10-02 review findings (CLS-M1..M5, CLS-m1..m5): it now builds an isolated `uv` environment from a hash-locked lock instead of installing into the kernel (no restart), and its blob `fce863b3` passed a hosted run in one pass on 2026-10-04 (Colab CLI 0.7.4 sequential execution on a fresh Colab Tesla T4 at `a110af3`, default path, 14/14 code cells in order, no restart, 0 errors; first row of Recorded executions). That run is CLI sequential execution, not a browser `Run all`. The 2026-09-21 Kaggle T4 run of the previous blob `100adc4f` needed a manual restart after the install cell, so it was not a one-pass `Run all` and is not promotion evidence. The pre-flight rows above (the 2026-10-04 local CPU run of this version, the package-API probe, the notebook pre-flight of an earlier blob) and the superseded TASK-INFERENCE run are history, not promotion evidence. Remaining before promotion: in one hosted session, the BYOD and rerun journeys of step 6 (REL12) on the current blob, and an integrator's promotion decision.

Facts a reviewer should weigh: the sample is food photographs with the largest ingredient's mask, a phrase vocabulary
(`bread`, `chicken duck`, `steak`, …) and a boundary convention the PhraseCut-trained decoder never saw, but the CLIP
towers know the words, which is why the frozen model already lands well above the full-mask baseline
(the frozen decoder already finds most dishes — 63 of the 140 held-out records score above 0.8 IoU — but misses 24 almost entirely (IoU under 0.2), typically ingredient names it does not ground at all (`pie`, `lamb`, `garlic`, `shellfish` at 0.0), with precision 0.835 well above recall 0.736: it under-segments) and why the gain is a boundary refinement of one decoder on one convention, not a repair of a domain
gap; every rate is at one threshold (`MASK_THRESHOLD`) over one reference mask per record and the notebook says so; the
60-record validation split selects the epoch; the towers are frozen, so what the image encoder cannot resolve at
352 × 352 stays unsegmented; the decoder that was tuned serves every phrase, and the drawn scene re-segmented after
adaptation is the only evidence about what happened outside the vocabulary. The forward pass is deterministic on a
fixed device and dtype, but the training of the decoder is not bit-reproducible across GPUs, and 140 records make a
few hundredths of mean IoU the expected spread between two runs, not a finding. The Tesla T4 run reproduced the CPU sweep exactly (0.842 / 0.904 at epoch 4, plateau 0.849–0.855 after). The row has no sibling on this corpus — CLIPSeg is the fleet's only text-prompted segmenter — so the comparison is the frozen decoder against the tuned one on the same 140 records: the gain is almost all recall (0.736 → 0.927 at precision 0.835 → 0.881), the records under 0.2 IoU fall from 24 to 1 and those above 0.8 rise from 63 to 111, one record (`garlic`) stays at 0.0, and the drawn scene outside the vocabulary moved by a hundredth (0.947 → 0.958 mean IoU, the two absent phrases still empty): a decoder of 1.1 M parameters learned FoodSeg103's annotation convention, not new visual concepts.

## Image-segmentation workshop notebook

`tutorials/DIMER_MultiModel_Image_Segmentation_Workshop.ipynb` (`E2E` / `WORKSHOP`, DIMER Notebook Specification 2.2) is a **Candidate**. It is recorded separately from `clipseg_segmentation_colab.ipynb` (also a Candidate), whose status it does not change. It carries the CLIPSeg package modules, the SAM (`kurtvalcorza/sam-vit-segmentation-pipeline@ed74a93`) and SAM 2 (`kurtvalcorza/sam2-segmentation-pipeline@df023e1`) reference modules, their manifests and licences, a runner, a frozen sample manifest and a hash-pinned dependency lock. These are installed into an isolated `uv` Python 3.12.12 environment. Its design is `docs/image-segmentation-workshop-spec.md`.

| Check | Automatic (every pull request) | Manual (before promotion) |
|---|---|---|
| Metadata, opening declaration, no persisted outputs, every code cell plain Python | `tools/validate_release_assets.py` | — |
| Each carried file matches `CARRIED_HASHES`; carried `clipseg_reference/` and CLIPSeg manifest equal the package; `source.json` agrees with the metadata; `vendor_provenance.json` digests equal the carried SAM / SAM 2 files | `tools/validate_release_assets.py`, `tests/test_workshop_notebook.py` | — |
| Default `Run all` on a fresh Colab T4 runtime without a restart, with total time, peak GPU memory and disk recorded | — | recorded 2026-09-27 for blob `6e455fb1` (PASS, 393 s; peak memory and disk not captured); blob `3f94b41f0870` executed in order by the Colab CLI on a fresh T4 2026-10-03 (PASS, 372 s from setup to report; not a browser `Run all`; peak memory and disk not captured) |
| Optional bring-your-own-data and unlabelled-image branches | — | not yet exercised |

| Date (UTC) | Notebook source | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-10-03 | `a93bf44` / blob `3f94b41f0870` (`3f94b41f0870846275675cad766d61c52e511a44`; downloaded from GitHub at the PR #9 head and blob-verified before the session). Executed file: `docs/execution-evidence/2026-10-03/DIMER_MultiModel_Image_Segmentation_Workshop_a93bf44_colab-cli-t4.ipynb`, SHA-256 `057bcebb24d3e96124a02ceddf17b3a2ad7e419c4bc94fe43f4a372f08ec4641` | Google Colab CLI 0.7.4 on a fresh Colab **Tesla T4** session (reported by cell 1) via the workspace `colab-cli-serial-test-suite` (`colab new --gpu T4`, `colab exec -f`, `colab stop`; session started 2026-10-03T00:15:41Z and stopped after the run); isolated `uv` Python 3.12.12 environment built from the carried hash-pinned lock | Default path. Code cells ran in order in one kernel; this is not a browser Run all, and the CLI records no execution counts, so order is evidenced by its `Executing cell k/N` log (cells 1–12 of 12). Stages: `prepare` (120 / 40 / 40 / 12), `clipseg` (4 epochs, 120 optimizer steps), `reload`, `sam`, `sam2`, `report`. The optional BYOD and unlabelled-image branches were left at their defaults (off) and were not run; the results bundle was written but not downloaded | 371.9 s from setup to report (notebook's own timer); 384 s CLI execution wall | **PASSED** 12/12 code cells, no errors (the only stderr is the upstream `sam2_video` → `sam2` model-type warning). CLIPSeg validation mIoU 0.6175 (frozen) → 0.7919 (epoch 4, selected). Test mean target IoU / Dice: frozen 0.6173 / 0.6988, reloaded 0.8201 / 0.8850, empty 0 / 0; frozen → reloaded per image 33 improved, 1 unchanged, 6 worsened of 40. Reload parity passed on 92 records in a new process (max probability difference 0.0, identical masks, metric parity). SAM point / box 0.5062 / 0.7594; SAM 2 point / box 0.5293 / 0.7418; oracle filled box 0.5467. SAM 2 box expansion: 3 improved / 9 worsened at +10%, 1 / 11 at +25% (3 whole-image boxes at +25%). `segmentation_results.zip` was written. **Compared with the 2026-09-27 Colab run of blob `6e455fb1`:** every figure that row records is equal (validation 0.6175 → 0.7919 at epoch 4; test 0.6173 → 0.8201; empty 0; SAM 0.5062 / 0.7594; SAM 2 0.5293 / 0.7418; oracle 0.5467; reload max difference 0.0; 1 / 11 at +25%); the +10% counts equal the CPU pre-flight; Dice and the 33 / 1 / 6 split were not recorded in that row; setup-to-report time is 21.4 s shorter (371.9 s vs 393.3 s). Boundary: the saved outputs were inspected; peak GPU memory and disk were not captured; the BYOD, unlabelled-image and bundle-download journeys remain open. Status stays **Candidate** |
| 2026-09-27 | `54ae910` / blob `6e455fb153b7` (the executed file's code cells equal this blob; Colab added only a `# @title` line to cell 3, and the carried files and hashes are unchanged) | Google Colab, fresh **Tesla T4** runtime (reported by cell 2); isolated `uv` Python 3.12.12 environment built from the carried hash-pinned lock | Default `Run all` without a restart: all 12 code cells executed in order (execution counts 1–12, no errors). Stages: `prepare` (120 / 40 / 40 / 12), `clipseg` (4 epochs, 120 optimizer steps), `reload`, `sam`, `sam2`, `report`. The optional branches were left at their defaults (off) | 393.3 s from setup to report | **PASS**. CLIPSeg validation mIoU went from 0.6175 (frozen) to 0.7919 (epoch 4, selected). On test, mean target IoU went from 0.6173 (frozen) to 0.8201 (reloaded); the empty baseline is 0. Reload parity passed with identical masks (max probability difference 0.0). SAM point / box: 0.5062 / 0.7594. SAM 2 point / box: 0.5293 / 0.7418. The oracle filled box scored 0.5467. The SAM 2 box-expansion deltas equal the CPU pre-flight (1 improved / 11 worsened at +25%). `segmentation_results.zip` was exported. Peak GPU memory and disk use were not captured in the visible outputs |
| 2026-09-27 | This branch (carried `workshop.py` sha256 `51129c5383dc…`) | Builder pre-flight in a Linux container, CPU only (4 cores). The exact hash-pinned lock was installed with `uv` 0.12.15 into managed Python 3.12.12 (torch 2.14.0+cu130, transformers 4.57.6). The runner's CUDA-only lines (the GPU-required `device()`, synchronize, device name, peak memory) were patched in a copy for CPU | `prepare` (120 train / 40 validation / 40 test / 12 activity), `clipseg` (4 epochs, 120 optimizer steps), `reload` (new process), `sam`, `sam2` (including the box-expansion activity), `report`. The notebook's display, metric-demo and report cells were then run against those outputs; the optional branches were left at their defaults (off) | prepare 49 s, clipseg 120 s, reload 38 s, sam 784 s, sam2 242 s, report 5 s (CPU float32) | **PASS**. CLIPSeg validation mIoU went from 0.618 (frozen) to 0.792 (epoch 4, selected). On test, mean target IoU went from 0.617 (frozen) to 0.820 (reloaded); the empty baseline is 0. Reload parity passed on all 92 records (max probability difference 0.0). SAM point / box: 0.506 / 0.759. SAM 2 point / box: 0.529 / 0.742. In the SAM 2 box-expansion activity, 3 improved / 9 worsened at +10% and 1 / 11 at +25%: boxes expanded to the image edges make SAM 2 select the inverse region (plate and background), which is model behaviour the activity is designed to expose. `segmentation_results.zip` was exported. **Not a supported runtime and not promotion evidence**: CPU float32 differs from the T4 path, so Colab figures may differ |

`vendor_provenance.json` previously recorded SHA-256 digests computed on CRLF checkouts for `sam_reference.py`, the SAM manifest and both LICENSE files. Those digests matched no carried file. They now record the LF digests, which equal the files at the pinned SAM and SAM 2 revisions. A fresh Colab T4 `Run all` of the committed blob is recorded in the 2026-09-27 Colab row of the workshop table. The notebook stays **Candidate** until a reviewer confirms that run against the blob under review and an integrator promotes it; the optional branches remain unexercised.

**Revision 0.2.0-candidate (2026-10-02).** The Notebook Review Framework v1 review of `bf3ae25` (`docs/reviews/2026-10-02-notebook-review/`) found 2 Major and 9 Minor issues; they are fixed in the notebook and the carried `workshop.py` (sha256 `51129c5383dc…` → `bc97f0ad87c4…`), recorded in `metadata.dimer.review_revisions`. The 2026-09-27 Colab run above covers the previous blob `6e455fb1` only. Before promotion, a fresh Colab T4 run of the new revision must cover the default `Run all`, the BYOD branch with a valid directory and one rejected input (REL12), the unlabelled-image branch, and downloading the results bundle. `tests/test_seg_review_fixes.py` exercises the changed bookkeeping, messages and displays on CPU with stand-in models; that is not clean-runtime evidence.

**Notebook source layout change (2026-10-03).** The workshop's `CARRIED_FILES` literal in cell `code-03` was one 415,188-character line. `tools/split_workshop_carrier.py` rewrote it as parenthesised runs of short string pieces (at most 1,000 characters each), and the change is logged in `metadata.dimer.generated_from.post_generation_revisions`. Python joins the pieces back into the same text: the carried files, `CARRIED_HASHES`, the carried `source.json` and `generated_from.files` are unchanged, and no cell line is now longer than 2,000 characters (`python tools/split_workshop_carrier.py --check`). The notebook blob changes from `26a3a74e0a8c` (revision 0.2.0-candidate, which has no hosted run) to `3f94b41f0870`. The 2026-09-27 Colab run covers blob `6e455fb1`. A hosted run of the new blob passed on 2026-10-03 (Colab CLI on a fresh T4, default path, 12/12 code cells in order, no errors, every recorded figure equal to the 2026-09-27 run; first row of the workshop table). Status stays **Candidate**.
