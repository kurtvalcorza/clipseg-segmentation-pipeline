# Release verification

`tutorials/clipseg_segmentation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until the
exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation, code-cell
compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but are **not**
runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate record for the
notebook. The earlier `TASK-INFERENCE` notebook's Kaggle CPU run (2026-09-14, retained below) is history for a
superseded blob, not evidence for this one.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 8-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the FoodSeg103
  parquet-conversion revision `176acc3edd2432ee126bda6fb01469eadeb018df` is the one other 40-hex revision the
  documents may cite);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `ClipSegSegmentationPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus`, `build_sample_dataset(corpus, seed=SPLIT_SEED)` / `load_byod_dataset` + `split_dataset`,
  `validate_dataset` per split, `check_split_disjoint`, `write_dataset_csv`, the four dataset refusal probes, the
  ceiling print, `validate_inputs` with the duplicate-phrase refusal probe, `segment` on the drawn scene with the
  structural checks and the `evaluation_report` against the drawn masks, `empty_baseline`, `full_baseline`,
  `pipe.evaluate` on the frozen model, `pipe.adapt` with its explicit hyperparameters, `pipe.evaluate` on the
  validation and test splits after adaptation with the two mean-IoU assertions, the drawn scene re-segmented after
  adaptation, the example panels, `pipe.save_artifact`, `ClipSegSegmentationPipeline.from_artifact` and the
  mask-parity assertion, and the result fields `weight_file` / `weight_format` / `weight_sha256`, the `corpus`
  block), the seven expected `outputs/` paths, the learner-facing statements (Apache-2.0 weights, the uncalibrated
  sigmoid, the caller-owned threshold, adaptation of the decoder on labelled records, the two non-adapted baselines,
  mean and micro IoU, Dice, pixel precision and recall, the empty-mask and full-mask baselines, the per-pixel binary
  cross-entropy, the frozen-tower cache, highest validation mean IoU, no dispersion estimate, float32 on every
  device, the leakage guidance, the precision-and-recall guidance, the excluded instance/panoptic scope, the snapshot
  note, the troubleshooting block) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git
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
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `EPOCHS = 8`, `LEARNING_RATE = 3e-4`, `BATCH_SIZE = 8`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`, `safetensors==0.8.0`,
   `numpy==2.5.3`, `pillow==11.3.0`, `pyarrow==25.0.1` (an interpreter restart after the install is expected where
   the runtime's preinstalled torch or numpy differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
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
   - Section 6: the empty-mask baseline (mean IoU 0.0 exactly), the full-mask baseline (≈ @P:FULL_MIOU@) and the
     frozen model's test rates (≈ @P:FROZEN_MIOU@ mean IoU / @P:FROZEN_DICE@ Dice in the Tesla T4 build record —
     @P:FROZEN_READ@) with four records' scores printed under their phrases;
   - Section 7: `pipe.adapt` printing epoch 0 as the frozen model, 1,127,009 trainable of 150,747,746 parameters,
     and an eight-epoch history with the validation mean IoU rising (build record: @P:VAL_CURVE@, `best_epoch`
     @P:BEST_EPOCH@);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, the predicted area
     and `outputs/…_evaluation_report.json` written (the cell asserts the adapted test mean IoU is at least the
     frozen one and above the full-mask baseline — @P:ADAPTED_MIOU@ against @P:FROZEN_MIOU@ in the build record,
     Dice @P:FROZEN_DICE@ → @P:ADAPTED_DICE@);
   - Section 9: six example panels under `outputs/…_examples/`; the drawn scene re-segmented by the adapted model
     with `outputs/…_scene_adapted.json` and `.png` (build record: @P:DRAWING_AFTER@ — a recorded observation, not an
     assertion); `pipe.save_artifact` writing `outputs/…_adapter/{adapter.safetensors,manifest.json}` (the decoder,
     about 4.5 MB) and `ClipSegSegmentationPipeline.from_artifact` reloading it with 8/8 identical masks on eight
     test records (the cell asserts it); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity
     and licence, the snapshot block (`weight_file`, `weight_format`, `weight_sha256`), the `corpus` block, the
     inference-contract records before and after adaptation, the comparison, the artifact digest, the reload
     parity, the runtime versions, device and dtype;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model
   identifier and immutable revision, whether the model cache, the weights directory and the row-group cache were
   clean, outcome, produced outputs, the observed metrics (as observations, not a benchmark) and any warning or
   applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `clipseg_segmentation_colab.ipynb` (`E2E`) | pending | — | — | pending: the supported-runtime run of the exact candidate blob is not yet recorded |
| `clipseg_segmentation_colab.ipynb` (`TASK-INFERENCE`, superseded) | `69dc7ee` / `2fd1160bfd0d` | 2026-09-14 | Kaggle CPU (`kurtvalcorza/dimer-nb2-clipseg-segmentation` v1) | PASSED — 8/8 code cells, 18 files, 605 MB staged, 224.8 s; evidence for the earlier inference-only notebook, not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/clipseg_segmentation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/clipseg_segmentation_colab.ipynb`). Wall times, when recorded, are the sum of
per-cell times reported by the executor and include installs and the model download; they are measurements for the
stated runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-20 | package API at `@P:PROBE_SHA@` (pre-flight, not the notebook blob) | Kaggle Tesla T4 script kernel (`kurtvalcorza/dimer-probe-clipseg-e2e` v1; `torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12, `cuda:0`, float32), branch cloned, pins installed, snapshot staged from the Hub | `tests/test_model_backed.py` (@P:MB_RESULT@) and the recipe probe: the eight pinned row groups read over range requests (800 records, digest match), empty and full baselines, frozen model on the 140 test records, `adapt(epochs=8, lr=3e-4, batch_size=8)` with validation-mIoU selection, adapted evaluation, artifact round trip | @P:PROBE_WALL@ | @P:PROBE_OUTCOME@ |
| 2026-09-20 | package API at the working tree of `feat/e2e-segmentation-adaptation` (pre-flight, not the notebook blob) | Windows venv `dimer-next16` (`torch 2.14.0+cu130`, `transformers 4.57.6`, Python 3.12.10, `cpu`, float32), `CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`, row groups cached | the CPU recipe sweep on the default split (600 / 60 / 140): frozen 0.637 mean IoU (Dice 0.715, precision 0.835, recall 0.736; empty 0.000, full 0.283); 8 epochs, batch 8, validation mean IoU per epoch (epoch 0 = frozen 0.603) → test mean IoU / Dice at the kept epoch: lr 3e-5 → 0.777 … 0.820 (epoch 7 kept) → 0.804 / 0.875; lr 1e-4 → 0.813 … 0.844, still rising at epoch 8 (kept) → 0.831 / 0.896; **lr 3e-4 → 0.822, 0.840, 0.851, 0.855 (epoch 4 kept), then 0.848–0.854 plateau → 0.842 / 0.904**; lr 1e-3 → 0.848, 0.834, 0.850, 0.856, 0.858 (epoch 5 kept), 0.855, 0.849, 0.856 → 0.850 / 0.910. 3e-4 and 1e-3 are within noise of each other; 3e-4 plateaus by epoch 4 without the epoch-2 dip and is the default. Decoder-on-cache parity max abs 7.0e-4 (fp16 cache); artifact 4,514,484 bytes; reload parity 8/8 | ~19 min (tower cache 129–157 s per arm, adapt 307–369 s per arm incl. the eight validation passes, frozen test 35.8 s; the box was shared with another CPU job) | PASS — pre-flight only; fixed the recipe at lr 3e-4 × 8; not promotion evidence |
| 2026-09-14 | `69dc7ee` / `2fd1160bfd0d` (`TASK-INFERENCE`, superseded) | Kaggle CPU (`kurtvalcorza/dimer-nb2-clipseg-segmentation` v1) | Default sample path, `Run all` from a fresh interpreter, no repository checkout | 224.8 s | PASSED — 8/8 code cells, 18 files, 605 MB staged; not evidence for the `E2E` blob |

## Current status

**Candidate.** The `E2E` notebook has not yet executed in a supported runtime; the registry stays **Candidate** until a
clean run of the exact candidate blob is recorded above and an integrator promotes it. What exists: static validation
(`tools/validate_release_assets.py`), the generator parity checks (`--check` OK), the offline unit suite, the CPU run of
the model-backed suite (6 of 7, the CUDA test skipped), the CPU recipe sweep and the Tesla T4 pre-flight of the package
API (table above).

Facts a reviewer should weigh: the sample is food photographs with the largest ingredient's mask, a phrase vocabulary
(`bread`, `chicken duck`, `steak`, …) and a boundary convention the PhraseCut-trained decoder never saw, but the CLIP
towers know the words, which is why the frozen model already lands well above the full-mask baseline
(@P:FROZEN_READ@) and why the gain is a boundary refinement of one decoder on one convention, not a repair of a domain
gap; every rate is at one threshold (`MASK_THRESHOLD`) over one reference mask per record and the notebook says so; the
60-record validation split selects the epoch; the towers are frozen, so what the image encoder cannot resolve at
352 × 352 stays unsegmented; the decoder that was tuned serves every phrase, and the drawn scene re-segmented after
adaptation is the only evidence about what happened outside the vocabulary. The forward pass is deterministic on a
fixed device and dtype, but the training of the decoder is not bit-reproducible across GPUs, and 140 records make a
few hundredths of mean IoU the expected spread between two runs, not a finding. @P:SIBLING_COMPARISON@
