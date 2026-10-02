# CLIPSeg Text-Prompted Segmentation E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 2 October 2026  
**Repository:** `kurtvalcorza/clipseg-segmentation-pipeline`  
**Notebook:** `tutorials/clipseg_segmentation_colab.ipynb`  
**Reviewed commit:** `bf3ae25850c9f8e94050e437bc14caaf6f91baeb` (`main`, confirmed with `gh api repos/kurtvalcorza/clipseg-segmentation-pipeline/commits/main`)  
**Notebook Git blob:** `100adc4fbae4da4572a27d028886390f32e54719`. This is the blob executed in the recorded Kaggle Tesla T4 run of 2026-09-21 (`fetched_blob_verified: true`). The recorded generating revision `9150e7b` is not an ancestor of `main` (the feature branch was squash-merged as `f6c9a41`), but `git diff 9150e7b bf3ae25 -- src tools/build_notebook.py tools/notebook_template.py pyproject.toml` is empty. Generator `--check` and `tools/validate_release_assets.py` both exit 0 at the reviewed commit.  
**Finding prefix:** `CLS`  
**Out of scope:** `tutorials/DIMER_MultiModel_Image_Segmentation_Workshop.ipynb` (a separate notebook with its own open fix PR #9).

## Executive assessment

The default path is well engineered and it reproduces. The notebook carries the package's three modules verbatim, asserts the inline manifest against the module identity, stages and re-hashes the pinned `CIDAS/clipseg-rd64-refined` snapshot, reads eight FoodSeg103 row groups pinned by SHA-256, splits by image with a disjointness check, demonstrates four dataset refusals and a duplicate-phrase refusal, scores two honest baselines and the frozen model, fine-tunes only the decoder on cached tower activations with validation-mIoU selection, scores the untouched test split, exports a safetensors adapter with a manifest and reloads it with 8/8 identical masks. The score semantics (uncalibrated sigmoid, caller-owned threshold) and the limits of one seeded split are stated correctly, and the interpretation section's quantitative claims check out against the recorded per-record rows (records under 0.2 IoU 24 → 1, above 0.8 63 → 111, `garlic` the only remaining 0.0).

A direct CPU run of every code cell, with the install skipped and `EPOCHS` reduced to 1 to fit the review's time cap, reproduced the recorded frozen-model numbers exactly:

| Measure | This review (CPU, pinned venv, `EPOCHS = 1`) | Kaggle T4 record (blob `100adc4f`, `EPOCHS = 8`) |
|---|---|---|
| Split / dataset | 600 / 60 / 140, 4 refusals | same |
| Drawn scene, frozen mean IoU | 0.947 | 0.947 |
| Baselines (test mean IoU) empty / full | 0.000 / 0.283 | 0.000 / 0.283 |
| Frozen test mean IoU / Dice / precision / recall | 0.637 / 0.715 / 0.835 / 0.736 | 0.637 / 0.715 / 0.835 / 0.736 |
| Validation mean IoU epoch 0 → 1 | 0.603 → 0.822 | 0.603 → 0.822 |
| Adapted test mean IoU / Dice | 0.809 / 0.879 (1 epoch) | 0.842 / 0.904 (epoch 4 of 8) |
| Reload parity | 8/8 | 8/8 |
| Exports | 11 entries under `outputs/` | same 11 |

Five problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (CLS-M1).** The only recorded run of this blob stopped in cell 3 with the restart `RuntimeError` (`cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`) and passed on a second attempt. The release record calls it **PASSED** and the release procedure says a restart "is expected".
2. **Reruns do not start from the frozen model (CLS-M2).** `pipe` is created once in Section 3 and `adapt` continues from whatever decoder is loaded. Every documented rerun after the default path (BYOD "re-run from that cell", `EPOCHS`, `LEARNING_RATE`, `THRESHOLD` "for both the frozen and the adapted model", `SPLIT_SEED`) reports an already-adapted decoder as "frozen model". Verified: re-running Section 6 printed `frozen_model_test` mean IoU 0.809, the adapted number, with `adapted: True` inside the result.
3. **The stated BYOD minimum does not work (CLS-M3).** The notebook says "at least eight images". Cell 13 validates every split against the 8-record minimum, so any upload under **50** images is refused, and an 8-image upload is refused with `2 records; 8..5000 are required`. The release record has no BYOD run (REL12).
4. **A common BYOD layout silently trains on the masks (CLS-M4).** The loader keys zip members by basename. With the usual `images/0001.png` + `masks/0001.png` layout, every record's image is replaced by its own mask, and the upload is accepted without a warning.
5. **Guided layer largely absent (CLS-M5).** Declared `GUIDED`, but there is no audience statement, how-to-use, roadmap, glossary, prediction prompt, checkpoint or conclusion template, and the 1,370 carried lines of Section 2 are not labelled as infrastructure.

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** (2026-09-26), `ml-worker` `origin/main` |
| Intended audience | Not stated. Prerequisites: "basic Python, NumPy and PIL; what a per-pixel sigmoid is …; what intersection-over-union and Dice measure …" |
| Supported runtime | "Google Colab or Kaggle, Python 3.12; CPU or CUDA"; CUDA used when present; float32 on every device |
| Promised outcomes | One-pass `Run all` with no configuration edit; pinned install; carried modules; digest-verified snapshot; digest-pinned FoodSeg103 records split by image without leakage; inference contract on a drawn scene with an input manifest and a rejection probe; baselines and frozen model; bounded decoder fine-tuning with validation selection; held-out evaluation; six panels showing "the reference mask, the frozen mask and the adapted mask side by side"; the scene re-segmented; adapter export and reload parity; optional experiments; BYOD through "the same validation, image-disjoint split, baselines, fine-tuning, held-out evaluation, artifact export and reload-parity cells" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded revision `9150e7b` |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code, 3 of them carried modules: 88, 835 and 447 lines). Also read: the generator, the template, `tools/validate_release_assets.py`, the three package modules, `README.md`, `STATUS.md`, `MODEL_CARD.md`, `tutorials/README.md`, `docs/release-verification.md`, `docs/WEIGHTS.md`. The repository has no `docs/execution-evidence/` directory and no `AGENTS.md`.
- **Documented execution evidence.**
  - Sources: `docs/release-verification.md`, plus the archived executor output `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-clipseg-segmentation/v3/evidence/` (`run_summary.json`, `executed-pass1.ipynb`, `executed.ipynb`, `outputs/`, workspace).
  - The run: Kaggle Tesla T4, 2026-09-21, **blob `100adc4f`, the reviewed blob**, clean HF cache, image torch 2.10.0+cu128 / numpy 2.0.2 / transformers 5.0.0.
  - Attempt 1 failed in cell 3 with the restart `RuntimeError` after 237.2 s. Attempt 2 ran 11/11 code cells in 198.1 s (`restarted_after_install_cell: true`, total 435.4 s).
  - No Colab run, no BYOD run and no optional-experiment run of this blob is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py` on Windows, CPU only (`CUDA_VISIBLE_DEVICES=-1`, `HF_HUB_OFFLINE=1`), in the build venv `venvs/dimer-next16` the release record names (Python 3.12.10, torch 2.14.0+cu130, transformers 4.57.6, numpy 2.5.3, pillow 11.3.0, pyarrow 25.0.1 — the notebook's pins). Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, the notebook's own executor hook.
  - **Not a clean runtime:** the eight pinned snapshot files and the eight row-group files were hard-linked into a scratch working directory; the notebook's own `verify_snapshot` and `fetch_corpus` re-hashed them and fetched nothing.
  - **Execution method:** every code cell ran **verbatim from the notebook JSON** in one namespace. The only substitution was `EPOCHS = 8` → `1` (a form-field literal, the way an EXE6 executor sets a field) to stay inside the review's 20-minute CPU cap. The BYOD branch was driven through a `google.colab.files.upload` shim returning one generated zip, because the notebook has no location field.
  - **Runs:** one probe run of 391 s (default path, Section 6 and Section 7 reruns, BYOD loader cases, BYOD through cells 13–23), plus one loader-only probe; both exited on their own.

## 2. Separate judgments

- **Technical correctness:** sound on the default path. Identity, digest, split-disjointness and reload guards all hold, and the metrics code matches its definitions. Defects: the in-kernel install (CLS-M1), a pipeline object whose state leaks across reruns (CLS-M2), a BYOD loader that collapses folders by basename (CLS-M4), and a BYOD run that still records the FoodSeg103 `corpus` block (CLS-m4).
- **Promise fulfilment:** every engineering promise of the default path is met except one-pass `Run all`. Not delivered as stated: the frozen-mask panel (CLS-m1), the BYOD path at the stated size (CLS-M3) and for a common layout (CLS-M4), and the optional experiments' frozen-versus-adapted readings (CLS-M2).
- **Scientific validity:** the default experiment is valid: image-disjoint split, validation-only selection, test used once, honest baselines, explicit no-dispersion caveat. Validity breaks only on reruns, where the "frozen" baseline and epoch 0 are already adapted (CLS-M2).
- **Learner experience:** accurate, careful prose with good reading guidance in Sections 6, 8 and the interpretation section. It is a dense reference notebook rather than a guided one (CLS-M5). Two numeric inconsistencies (CLS-m2, CLS-m3) undercut its own "what normal output looks like".
- **Spec conformance (2.2):** applicable `MUST`s not met: RUN1, RUN10, ENV6, REL2, REL11 (CLS-M1); DAT12, DAT13, DAT14 in practice (CLS-M2, CLS-M3, CLS-M4); DAT19 for the folder layout (CLS-M4); REL12 (no BYOD verification recorded; CLS-M3); SRC3 "knowingly stale instructions" for the panel and scene-number text (CLS-m1, CLS-m2); UX12 for the runtime estimate (CLS-m3). `SHOULD`s largely unmet without a recorded deviation: GDL1–GDL4, GDL6, GDL7, GDL9–GDL11, GDL14, EXE1 (`THRESHOLD`), EXE2 (BYOD location). Met: ST1–ST8, RUN2–RUN9, RUN11–RUN14, ENV1–ENV5, ENV7–ENV9, MOD (as checked by the validator), DAT1–DAT11, DAT17, DAT18, SPL1, SPL3, SPL5–SPL10, FT1–FT8, EVAL1–EVAL3, EVAL5–EVAL8, EVAL10, EVAL14, EVAL15, UNC1–UNC4, OUT1–OUT10 (OUT7/OUT9 in part on BYOD; CLS-m4), ART1–ART8, VER1–VER5, SRC1, SRC2 (default path), SRC4–SRC12.

## 3. Promise and objective tracing

| Claim (opening / section) | Implementation | Observable result | Learner interpretation |
|---|---|---|---|
| `Run all` in a fresh runtime completes with no intervention | cell 3 in-kernel `pip install` + stale-import guard | Kaggle attempt 1 `RuntimeError`; attempt 2 after a restart | **Not delivered** (CLS-M1) |
| Pinned, digest-verified model | cell 11 manifest assert, `stage_missing_files`, `verify_snapshot`, `from_pretrained` | `verified_files: 8`, revision `999e0328…`, `source: local-snapshot` | clear |
| Digest-pinned real masks, split by image without leakage | cell 13 `fetch_corpus`, `read_corpus`, `build_sample_dataset`, `check_split_disjoint` | 800 records, 600/60/140, three digests | clear |
| Four dataset refusals before the model runs | cell 13 probes | duplicate id, empty mask, wrong shape, too small all rejected with the failed rule | clear |
| Inference contract on a drawn scene | cell 15 | checks all `True`, mean IoU 0.947, absent phrases area 0.0 | prose quotes 0.86 for the same scene (CLS-m2) |
| Baselines and frozen model on the test split | cell 17 | 0 / 0.283 / 0.637 with definitions and four rows | clear; "frozen" is wrong on reruns (CLS-M2) |
| Bounded decoder fine-tuning, validation selection | cell 19 `pipe.adapt` | per-epoch validation rates, `best_epoch` | clear |
| Held-out evaluation, four-way comparison | cell 21 | comparison table, report JSON, two asserts | clear and well explained |
| Six panels: reference, **frozen** and adapted mask side by side | cell 23 | panels show reference and adapted only (648 px = 2 × 320 + 8) | **Not delivered** (CLS-m1) |
| Scene re-segmented after adaptation | cell 23 | 0.947 → 0.953 (1 epoch) / 0.958 (Kaggle) | clear |
| Adapter export and reload parity | cell 23 | 64 tensors, 4,514,484 bytes, 8/8 | clear |
| Optional experiments | cell 24 prose | reruns report an adapted decoder as frozen; `THRESHOLD` has no control | **Not working as stated** (CLS-M2) |
| BYOD through the same cells, "at least eight images" | cell 13 upload → same cells | under 50 images refused in cell 13; `images/` + `masks/` layout accepted with the masks as images | **Not delivered as stated** (CLS-M3, CLS-M4) |

| Objective | Learner activity | Evidence it was exercised |
|---|---|---|
| install the pinned runtime | run cell 3 | needs a restart (CLS-M1) |
| read what the carried package guarantees | none; 1,370 lines, no guidance on what to read | none (CLS-M5) |
| stage and digest-verify the snapshot | run cell 11 | printed dicts |
| fetch, validate and split without leakage | run cell 13 | digests, refusals |
| read the output contract correctly | prose only; no question or prediction | none beyond reading (CLS-M5) |
| measure the frozen model beside two baselines | run cell 17 | table; wrong on reruns (CLS-M2) |
| run the bounded fine-tuning | run cell 19 | history |
| evaluate on the test split | run cell 21 | comparison |
| look at adapted masks next to the frozen ones | cell 23 panels | frozen masks not drawn (CLS-m1) |
| export and reload with verified parity | run cell 23 | 8/8 |

## 4. Journeys

- **First-time learner (source inspection).** The opening is long but accurate and honest about scope, and Sections 6 and 8 tell the learner how to read the metrics. After the Section 1 install the learner meets 1,370 lines of carried package code that are described ("Nothing in these cells runs a model yet") but not labelled as infrastructure they can skip. No section asks for a prediction or checks understanding, and there is no audience statement, roadmap, glossary or conclusion template (CLS-M5). Section 5 tells the learner to expect 0.86 on the drawn scene; they will see 0.947 (CLS-m2). Section 9 promises a frozen-mask panel they will not find (CLS-m1).
- **Clean default.**
  - Documented: Kaggle Tesla T4, 2026-09-21, blob `100adc4f`. Attempt 1 failed in cell 3 with the restart `RuntimeError` after 237.2 s; attempt 2 ran 11/11 cells in 198.1 s (CLS-M1). Metrics as tabulated above.
  - Direct: local CPU, install skipped, pinned venv, `EPOCHS = 1`: 11/11 cells, frozen numbers identical to the record; the tower cache took 120 s on CPU.
  - No Colab run.
- **Active learning (direct).** After the default run I followed the optional-experiment prose literally: re-ran Section 6 (cell 17) and Section 7 (cell 19). Section 6 printed `frozen_model_test` mean IoU **0.809**, the adapted value from the default run, with `adapted: True` in the returned dict. Section 7's epoch 0, labelled `frozen model`, reported validation mean IoU **0.822** instead of the frozen 0.603 (CLS-M2). `THRESHOLD` is not a form field (CLS-M2). `LEARNING_RATE` and `SPLIT_SEED` were not run; they are subject to the same state leak by source inspection.
- **Reuse and recovery (direct, upload shim only).**
  - Through cells 13–23 after the default run, a 60-image generated set (red circles, blue squares, yellow triangles) reached export and reload parity 8/8. Its Section 5 scene "frozen" mean IoU was 0.957 (the true frozen value is 0.947), its Section 6 "frozen" test mean IoU 0.946 against **0.939** from a freshly loaded pipeline, and its epoch 0 "frozen model" 0.952 against **0.953** fresh (CLS-M2). `result.json` recorded `data_source: BYOD (my_shapes.zip)` but still carried the FoodSeg103 `corpus` block (CLS-m4).
  - Size: 8, 20, 37, 38 and 49 images refused in cell 13 with messages about the split sizes (`2 records; 8..5000 are required` for 8 images); 50 and 60 accepted (CLS-M3).
  - Refused with a clear message: missing `masks.csv`, a `label` column instead of `prompt`, a mask of the wrong size, an 80-character prompt, a non-zip upload.
  - Refused with a message that does not say what to do: a macOS zip (`2 file(s) have no masks.csv row, e.g. ._img000.png`), a `README.txt` in the zip (CLS-m5).
  - **Accepted silently and wrong:** `images/0000.png` + `masks/0000.png` (every record's image equals its mask, 2 colours), and a stray `other/img000.png` that replaced `img000.png` with a black image (CLS-M4).
  - JPEG masks are accepted; compression ringing raised the mean mask area from 0.111 to 0.122 (CLS-m5).
  - The Colab upload widget itself was not verified.

## 5. Findings

### Major

#### CLS-M1 — `Run all` needs a manual restart after the install cell, and the release record reports it as a pass

- **Cell/section:** Section 1, cell 3 (generated by `_INSTALL_GUARD` in `tools/build_notebook.py` lines 47–70, emitted at lines 470–471). Release records: `docs/release-verification.md` step 4 ("an interpreter restart after the install is expected", line 87), the evidence tables (lines 143, 155, 156) and *Current status* (line 163); `README.md`, `STATUS.md`, `tutorials/README.md` ("Release-grade").
- **Observed issue:** cell 3 pip-installs nine pins (`torch==2.14.0`, `numpy==2.5.3`, `transformers==4.57.6`, …) into the running kernel. In a hosted image that has already imported other versions it raises `RuntimeError: Core dependencies changed while older modules were loaded … Restart the runtime, then rerun from the top.` The Section 1 prose and the troubleshooting block present this stop as designed behaviour.
- **Consequence:** the opening promises that `Run all` completes with no intervention; in the supported runtime class it does not. The record reports "**PASSED** — 11/11 code cells ok (1 restart after install cell)" and promotes the blob to Release-grade, which RUN10 and the §5 closing sentence exclude.
- **Evidence (documented):** `run_summary.json` for blob `100adc4f`: attempt 1 `ok: false` after 237.2 s with `cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3`; attempt 2 `ok: true` after 198.1 s; `restarted_after_install_cell: true`.
- **Recommended correction:** adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass. The setup cell:
  - bootstraps uv;
  - creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`);
  - installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`);
  - runs the pinned stages in that environment.

  The kernel's preloaded NumPy/torch are then never replaced, so no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb`. This repository's own `tutorials/DIMER_MultiModel_Image_Segmentation_Workshop.ipynb` already runs this way and recorded a restart-free Colab T4 `Run all` on 2026-09-27.

  Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement the pattern in the repository's notebook generator, regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a hosted Colab or Kaggle `Run all` of the regenerated blob completes all code cells in **one pass** (`restarted_after_install_cell: false`) from a fresh runtime with an empty cache. The release records name that blob, step 4 no longer says a restart is expected, and no restart-dependent run is reported as a clean pass.
- **Spec:** RUN1, RUN10, ENV6, REL2, REL11.

#### CLS-M2 — Reruns after the default path report an adapted decoder as the "frozen model"

- **Cell/section:** Section 3 (cell 11, the only `from_pretrained`), Sections 5–7 (cells 15, 17, 19), the BYOD instruction in the opening (template line 73) and *Optional experiments* in the interpretation section (template line 498). Root cause in `ClipSegSegmentationPipeline.adapt` (`src/clipseg_segmentation_pipeline/pipeline.py` lines 621–752), which starts from the model's current decoder weights and never restores the base decoder, while epoch 0 is labelled `"frozen model"`. `THRESHOLD` is assigned in cell 15 (template line 242), not a form field.
- **Observed issue:** `pipe` is loaded once. After Section 7 its decoder is adapted. Every documented rerun starts below Section 3 — BYOD says "re-run from that cell" (Section 4); the experiments say "raise `EPOCHS` and watch", "change `LEARNING_RATE`", "set `THRESHOLD` to 0.3 or 0.7 before Section 6 … for both the frozen and the adapted model", "change `SPLIT_SEED`". So Section 5's "frozen" scene, Section 6's "frozen model" and Section 7's epoch 0 are all the previously adapted decoder, and Section 7 trains on top of it. `pipe.evaluate` sets `adapted: True`, but no cell prints it. There is also no control for `THRESHOLD`, and "before Section 6" names no cell to edit.
- **Consequence:** the comparison the notebook teaches — frozen versus adapted on the same records — is silently invalid on every rerun path, including the BYOD path learners are told to read "baselines first" on. A learner raising `EPOCHS` sees a curve that starts at the old optimum and concludes extra epochs do nothing; a BYOD learner's "frozen" row is a FoodSeg103-adapted model.
- **Evidence (direct, CPU, `EPOCHS = 1`):**
  - Re-running cell 17 after the default path: `frozen_model_test` mean IoU **0.809** = the default run's adapted 0.809; `frozen_test['adapted'] = True`. True frozen value: 0.637.
  - Re-running cell 19: epoch 0 `note: 'frozen model'`, validation mean IoU **0.822**; first run 0.603.
  - BYOD (60 generated images) through cells 13–23 after the default run: scene "frozen" 0.957 (true 0.947); test "frozen" 0.946 vs **0.939** from a freshly loaded pipeline; epoch 0 0.952 vs 0.953 fresh. The gap is small on this easy synthetic set; it grows with how far the earlier adaptation moved the decoder.
  - Source: `adapt` has no `load_state_dict` or base-decoder restore before training (probe `adapt_resets_to_base_decoder: false`).
- **Recommended correction:** make every rerun entry point start from the verified base: either reconstruct the pipeline at the top of Section 4 (or Section 6) from the already-verified snapshot (`ClipSegSegmentationPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)`, no download), or add a package-level `reset_decoder()` that restores the base tensors and call it there. Label any printed "frozen" row from `evaluate(...)['adapted']` and refuse to label an adapted model as frozen. Make `THRESHOLD` a form field in Section 5 and give each optional experiment its exact rerun cell. Change it in `tools/notebook_template.py` (and the package if adding `reset_decoder`), then regenerate.
- **Acceptance check:** after a complete default run, set `USE_BYOD = True` (or change `EPOCHS`, `LEARNING_RATE`, `THRESHOLD`, `SPLIT_SEED`) and rerun exactly as the notebook instructs: Section 6's frozen metrics equal those of a freshly loaded pipeline on the same records to 1e-6, the returned dict has `adapted: False`, Section 7's epoch 0 equals the fresh frozen validation rate, and Section 5's frozen scene mean IoU is 0.947 on the drawn scene.
- **Spec:** DAT13, DAT14, SRC2, GDL10, UX5, UX7, EXE1.

#### CLS-M3 — BYOD refuses anything under 50 images while the notebook says "at least eight"

- **Cell/section:** opening BYOD paragraph (template line 73: "one row per image, at least eight images"); Prerequisites data contract ("a dataset needs 8..5,000 records"); cell 13 `dataset_manifests = {name: validate_dataset(part) …}` with the default `min_records=MIN_RECORDS` (8) applied to each split; `split_dataset` fractions 0.15 / 0.2 (`samples.py` lines 369–389). Release record: `docs/release-verification.md` has no BYOD row.
- **Observed issue:** `split_dataset` makes test = round(0.2 n) and validation = round(0.15 n), and cell 13 then requires every split, including validation, to hold at least 8 records. The smallest accepted upload is 50 images. The refusal names the split's size, not the upload's, and does not say how many images are needed.
- **Consequence:** a learner who follows the stated minimum — or brings 10–40 labelled photos, a realistic first BYOD — is stopped in cell 13 with a message that contradicts their own count ("2 records" for an 8-image upload). The promised "use your own masks" path is not usable at the advertised scale.
- **Evidence (direct):** 8 images → split 2/1/5, refused `2 records; 8..5000 are required`; 20 → `4 records …`; 37 → `7 records …`; 38 → `6 records …`; 49 → `7 records …`; 50 and 60 accepted and, for 60, carried through to export and reload parity 8/8.
- **Recommended correction:** decide the real BYOD floor and state it. Either validate the uploaded set against `MIN_RECORDS` once and validate the splits with `min_records=1`, as `pipe.adapt` and `pipe.evaluate` already do internally, or keep per-split minimums and state "at least 50 images" in the opening, the Prerequisites and `tutorials/README.md`. In either case refuse an undersized upload **before** splitting with a message that names the upload's image count and the minimum. Then record a BYOD run in `docs/release-verification.md` covering one representative accepted set and one rejected input.
- **Acceptance check:** an upload of exactly the stated minimum reaches Section 9's reload parity; an upload one image below it is refused in cell 13 with a message naming the uploaded count and the minimum; `docs/release-verification.md` records a BYOD run of the blob under review (REL12).
- **Spec:** DAT12, DAT19, REL12, UX10.

#### CLS-M4 — The BYOD loader collapses folders by basename, so `images/` + `masks/` uploads train on the masks

- **Cell/section:** `load_byod_dataset` (`src/clipseg_segmentation_pipeline/samples.py` lines 392–434: `members[Path(info.filename).name] = …` for zips, `members[file.name] = …` for directories, and `Path(row['file']).name` for the CSV); BYOD prose in the opening and Section 4.
- **Observed issue:** every zip member and directory file is stored under its basename, later ones overwriting earlier ones, and the CSV's `file`/`mask` columns are reduced to basenames too. The very common layout `images/0001.png` + `masks/0001.png` therefore maps both columns to the same bytes (the mask), and any duplicate basename elsewhere in the archive silently replaces an image. Neither case is refused.
- **Consequence:** the notebook fine-tunes and evaluates on images that are their own masks (or on the wrong photo) and reports normal-looking metrics. The learner gets no signal that their data never reached the model. This is the silent-wrong-data failure DAT19 exists to prevent.
- **Evidence (direct):** a 60-record zip with `images/0000.png … ` + `masks/0000.png …` and `file = images/0000.png, mask = masks/0000.png`: accepted; record 0's image has 2 distinct colours and equals its mask as RGB. A zip with a stray `other/img000.png` (black): accepted; `img000`'s image mean pixel 0.0 instead of 209.0.
- **Recommended correction:** key members by their full normalised relative path (reject absolute paths and `..`), resolve CSV entries by relative path, and refuse with a named message when two members share a key or when an image and its mask resolve to the same member. Document the accepted layouts (flat, or relative paths in `masks.csv`). Add regression tests for both cases in `tests/`.
- **Acceptance check:** the two-folder zip above loads with each record's image distinct from its mask; a zip whose CSV maps `file` and `mask` to the same member, or that holds two members with the same relative key, is refused in cell 13 with a message naming the colliding paths.
- **Spec:** DAT13, DAT19.

#### CLS-M5 — Declared `GUIDED`, but the guided layer is largely absent

- **Cell/section:** opening (cell 0) and Prerequisites (cell 1); Section 2 carried-module cells 5, 7, 9; the interpretation section (cell 24). Template `tools/notebook_template.py` (opening lines ~60–110, Section 2 note, interpretation lines ~480–510).
- **Observed issue:** no statement of who the notebook is for, no **How to use this notebook**, no roadmap, no Input → Model → Output contract in one place, no glossary for IoU / micro IoU / Dice / FiLM / sigmoid / decoder, no prediction prompt before the baselines or the fine-tuning, no interpretation checkpoint with a worked answer, no conclusion template. Section 2's 1,370 carried lines are not labelled as infrastructure the learner may skip, and cannot be collapsed by metadata. Troubleshooting covers install, snapshot and row-group failures but not memory (the cache holds ~1.3 GB on the host), a missing GPU, or BYOD.
- **Consequence:** the prose explains well, but the learner's only activity is running cells. The objectives "read the output contract correctly" and "look at the adapted masks next to the frozen ones" are never exercised, and the one activity offered (the optional experiments) does not work (CLS-M2).
- **Evidence (source inspection):** markers for audience, how-to-use, roadmap, glossary, "What to notice"/"Expected result", infrastructure label and conclusion template are absent; three "Look for"/"Watch" notes exist (Sections 1, 4, 7).
- **Recommended correction:** add the GDL layer in the template: audience and how-to-use, a roadmap, an input→output contract, a short glossary, a prediction prompt before Section 6 ("which baseline will the frozen model beat, and by how much?") and before Section 7, two collapsible checkpoints (reading precision against recall; why the drawn scene barely moves), an **Infrastructure — you may run without reading** label on Section 2 with collapsed metadata, a conclusion template, and BYOD/memory/GPU troubleshooting. Turn one optional experiment (the threshold sweep) into a Predict → Change → Run → Observe → Explain activity once CLS-M2 is fixed.
- **Acceptance check:** the regenerated notebook contains each GDL1–GDL14 element listed above, Section 2 carries an infrastructure label and `collapsed`/`jupyter.source_hidden` metadata, and at least one activity with a prediction prompt and a collapsible sample answer runs after the default path without changing it.
- **Spec:** GDL1–GDL4, GDL6, GDL7, GDL9–GDL11, GDL13, GDL14, UX8.

### Minor

#### CLS-m1 — Section 9 promises a frozen-mask panel that is not drawn

- **Cell/section:** Section 9 prose (cell 22, template line 395) and the "look at the adapted masks next to the frozen ones" objective; panel code in cell 23 (template line ~410–432); `tutorials/README.md` *Outputs* (which correctly says reference and adapted).
- **Observed issue / evidence:** the prose says each panel holds "the reference mask, the frozen mask and the adapted mask side by side"; the code builds two overlays and prints `'panels': ['reference overlay', 'adapted overlay']`; the written panel is 648 px wide (2 × 320 + 8). Direct and documented.
- **Recommended correction:** keep the frozen masks from Section 6 (or re-segment the six records with the base before adaptation) and add the third overlay, or correct the prose and the objective.
- **Acceptance check:** each panel under `outputs/clipseg_segmentation_examples/` shows exactly the overlays the prose names.
- **Spec:** SRC3, UX1.

#### CLS-m2 — Section 5 quotes 0.86 for the drawn scene; the scene scores 0.947

- **Cell/section:** Section 5 prose (cell 14, template line 218: "The inference-only card recorded a mean IoU of 0.86 on the four drawn shapes"). Elsewhere the notebook says 0.95 (cells 22, 24) and 0.947 (cell 24), and `docs/release-verification.md` says the inference-only card recorded 0.947.
- **Evidence:** direct CPU run 0.947; Kaggle record 0.947.
- **Recommended correction:** quote one value (0.947) everywhere, with its source.
- **Acceptance check:** every learner-facing statement of the frozen drawn-scene mean IoU equals the recorded value.
- **Spec:** SRC3.

#### CLS-m3 — "About 2 minutes of cell time … including the pinned install" does not match the record

- **Cell/section:** opening (template line 67), Prerequisites (template line 122), `tutorials/README.md` Run-all column.
- **Observed issue / evidence:** the recorded T4 run took 435.4 s (237.2 s for the failing first attempt, most of it the install, plus 198.1 s); even the second attempt alone was 198 s. Even the cells after the install (11–23, 02:01:31 → 02:04:27) took 176 s, about 3 minutes, of which the snapshot download was 66 s.
- **Recommended correction:** state measured times per stage with the environment, and label the total as the recorded wall time (or an estimate) including install and downloads.
- **Acceptance check:** every runtime figure names its environment and matches a recorded run.
- **Spec:** UX12.

#### CLS-m4 — BYOD exports keep the FoodSeg103 `corpus` block and no BYOD digest of the upload

- **Cell/section:** cell 23 `result_payload['corpus']` (unconditional), cell 21 report.
- **Evidence (direct):** after the BYOD run `result.json` has `data_source: BYOD (my_shapes.zip)` and `corpus.name: FoodSeg103 (validation split), first eight parquet row groups`. The split digests are recorded; the uploaded archive's SHA-256 is not.
- **Recommended correction:** write `corpus` only for the sample path; for BYOD record the upload's file name, byte count and SHA-256.
- **Acceptance check:** a BYOD `result.json` contains no FoodSeg103 fields and does contain the upload's SHA-256.
- **Spec:** OUT7, OUT9.

#### CLS-m5 — BYOD friction: upload-only, macOS zips and stray files refused without guidance, JPEG masks inflate

- **Cell/section:** cell 13 BYOD branch (template line ~157); `load_byod_dataset` unlisted-file rule; `coerce_mask` (`samples.py` line 255, `convert('L') > 0`).
- **Evidence (direct):** no location field — the branch always imports `google.colab` and opens an upload dialog (EXE2); a zip made by macOS Finder is refused with `2 file(s) have no masks.csv row, e.g. ._img000.png`, and a `README.txt` is refused the same way, without saying the files can be deleted or ignored; JPEG masks are accepted and their compression ringing raised the mean mask area from 0.111 to 0.122.
- **Recommended correction:** add a `BYOD_PATH` form field read before any upload; ignore `__MACOSX/`, `._*` and `.DS_Store` (or say to remove them); state that masks must be lossless (PNG) or binarise with a stated rule and warn on non-binary values.
- **Acceptance check:** BYOD runs from a path without importing `google.colab`; a Finder-made zip loads; a JPEG mask either is refused with a named rule or yields a warning.
- **Spec:** EXE2, DAT12, DAT19, UX10.

### Suggestions

- **CLS-S1:** update the declared notebook specification from 2.0 to 2.2 (generator `NOTEBOOK_SPEC`, validator expectations, documents).
- **CLS-S2:** add a small threshold sweep table (frozen and adapted precision/recall at 0.3/0.5/0.7 on validation) as the worked version of the threshold experiment.
- **CLS-S3:** add a per-record IoU histogram (frozen vs adapted) — the prose already cites the 24 → 1 and 63 → 111 counts; showing them makes them checkable.
- **CLS-S4:** record per-stage wall times and peak memory in `result.json`.

## 6. Readiness

**Needs revision.**

- Open Majors: CLS-M1 to CLS-M5.
- Unmet applicable `MUST`s: RUN1, RUN10, ENV6, REL2, REL11 (CLS-M1); DAT12, DAT13, DAT14, DAT19 (CLS-M2, CLS-M3, CLS-M4); REL12 (CLS-M3); SRC3 (CLS-m1, CLS-m2); UX12 (CLS-m3).
- Remaining gates after the fixes: a one-pass hosted `Run all` of the regenerated blob from a fresh runtime, a recorded BYOD run (one accepted set, one refusal), and the rerun acceptance check of CLS-M2.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, pinned venv, install skipped, `EPOCHS = 1`, hard-linked inputs):** the default path's frozen numbers equal the record; the Section 6 and Section 7 rerun results (CLS-M2); the BYOD size thresholds (CLS-M3); the basename collisions (CLS-M4); the two-panel output (CLS-m1); the scene value 0.947 (CLS-m2); the BYOD `corpus` block (CLS-m4); the macOS/JPEG behaviour (CLS-m5).
- **Verified from documented evidence:** the restart in the only hosted run of this blob (CLS-M1) and the per-record counts the interpretation cites.
- **Inferred, not run:** that `LEARNING_RATE` and `SPLIT_SEED` reruns show the same state leak (same code path); the size of the CLS-M2 distortion on a realistic BYOD set; Colab behaviour, including the upload widget.
- **Only Kurt can confirm:** whether the BYOD floor should be 8 images (changing the per-split rule) or 50 (changing the prose).
- **Finding most likely to be wrong:** CLS-M5's severity. The notebook's prose is unusually careful about interpretation, and a maintainer could reasonably treat the missing GDL elements as `SHOULD` deviations to document (Minor) rather than a Major learner-experience defect.
