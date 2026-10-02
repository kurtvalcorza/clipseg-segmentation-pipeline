# DIMER Image Segmentation Notebook — Review

**Verdict: Needs revision**
**Review date:** 2 October 2026
**Repository:** `kurtvalcorza/clipseg-segmentation-pipeline`
**Notebook:** `tutorials/DIMER_MultiModel_Image_Segmentation_Workshop.ipynb`
**Reviewed commit:** `bf3ae25850c9f8e94050e437bc14caaf6f91baeb` (`main`, confirmed against the GitHub API at review time)
**Notebook Git blob:** `6e455fb153b7c1aa974c33c2a947de7f779fff15` (carried `workshop.py` sha256 `51129c5383dc…`)
**Finding prefix:** `SEG`
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (`ml-worker` `origin/main` `b1cfe13`).

## Executive assessment

The default workflow is well engineered: an isolated hash-locked environment, carried and digest-checked
reference source, a frozen 120/40/40/12 FoodSeg103 selection, epoch selection on validation only, a fresh-process
adapter reload with probability and metric parity, and two comparisons that are kept apart (frozen vs adapted
CLIPSeg; SAM vs SAM 2 under identical oracle prompts). A fresh Colab T4 `Run all` of this exact blob passed on
2026-09-27. No Blocker was found.

Two Majors remain. The box-precision activity, the notebook's controlled comparison, displays per-image IoU deltas
but not the boxes actually used: on the real activity images 10 of 12 boxes are clipped by the image edge at +10%
and 11 of 12 at +25% (3 become the whole image), so "a less precise box" is mostly "a box pushed into the image
edge", and the learner is asked to notice a condition the displayed output does not contain (SEG-M1). The BYOD
branch runs the full adaptation route but shows the learner only a download link and a truncated log tail, never
their own metrics (SEG-M2). Nine Minors cover misattributed BYOD error messages (DAT19), tables that omit the
filled-box baseline and the precision/recall the notebook says it reports, a terminal summary without reload or
paired results, duplicated and very long output, an undisplayed `transformers` version (ENV3), guided-layer gaps
(infrastructure labels, how-to-use, glossary), the Colab download link, and a stale runtime statement.

## 1. Review contract and evidence

| Item | Recorded value |
|---|---|
| Revision | `bf3ae25` (`main` = GitHub API `commits/main` at review time); notebook blob `6e455fb1` |
| Spec version | NOTEBOOK_SPEC 2.2 (declared in metadata and opening cell) |
| Profile / mode | `E2E` / `WORKSHOP` |
| Audience | "Basic Python and Colab familiarity are sufficient; no prior segmentation knowledge is required" (md-00) |
| Prerequisites | Colab, T4 GPU runtime selected before Run all |
| Supported runtime | Google Colab, Linux x86_64, T4 GPU (code-02 refuses anything else); isolated `uv` Python 3.12.12 env |
| Promised outcomes | interpret masks and overlap metrics; identify unequal prompting assistance; run a controlled comparison; explain what an exported adapter and fresh reload establish; adapt the CLIPSeg decoder (4 epochs); optional own labelled data; optional unlabelled single image |
| Status | `Candidate` (metadata, opening, `tutorials/README.md`) |

Scope: all 24 cells (11 markdown, 12 code, the carried-source cell), the carried runner `workshop.py`, the carried
`clipseg_reference` modules (equal to `src/` by the validator), exported files, failure messages.

### Evidence actually obtained

- **Documented execution evidence:** `docs/release-verification.md`, workshop table row 1 — fresh Colab T4 `Run
  all` of blob `6e455fb153b7` at `54ae910`, 393 s, PASS, optional branches off. This blob equals the reviewed
  blob, so the row covers the default path of this revision. The executed notebook itself is not committed
  (`docs/execution-evidence/` does not exist); the row's figures were not re-derived here. Row 2 is a CPU builder
  pre-flight, not supported-runtime evidence.
- **Direct execution (this review, Windows, CPU, Python 3.13 anaconda, no GPU, no model weights):** the carried
  runner's `prepare` stage on the real pinned FoodSeg103 row groups (52 s; 120/40/40/12 roles; oracle filled-box
  test mean IoU 0.5467, equal to the hosted record); the SAM 2 stage's bookkeeping with a stand-in model (P01);
  the BYOD `prepare` path with synthetic valid and invalid directories (P03). Repository checks: `pytest` 63 passed
  / 2 skipped, `ruff`, `validate_release_assets.py`, `build_notebook.py --check` all pass.
- **Source inspection:** every cell and the carried runner.
- **Not verified:** SAM / SAM 2 / CLIPSeg behaviour on any input (no weights, no GPU); the BYOD and unlabelled
  branches on Colab (never executed on a supported runtime); Colab rendering of `FileLink`.

### Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Strong conceptual scaffolding (predictions before each result, collapsible worked interpretations, conclusion template, troubleshooting). Gaps: SEG-M1 (activity lacks the evidence its note asks for), m2, m4, m6, m7, m9 |
| Clean default | Documented execution evidence (Colab T4, this blob) + direct execution of `prepare` | Completes; summary renders. Summary omits reload/paired/activity results (m3); output volume (m4) |
| Active learning | Direct execution of the activity's bookkeeping with a stand-in SAM 2 on the real boxes | The comparison is pre-run, not learner-changed (S1); its display hides edge clipping (M1) |
| Reuse and recovery | Direct execution of BYOD `prepare` (valid + 4 invalid inputs); BYOD/unlabelled on Colab not verified | Valid BYOD accepted; invalid inputs refused before model execution, but messages misname the record (m1); results never shown (M2) |

## 2. Separate judgments

- **Technical correctness:** strong. Hash-locked env without a restart, carried-file integrity, decoded-image
  de-duplication and group/role checks, stage receipts that invalidate descendants, optimizer step count and
  decoder-update checks, frozen-parameter guard on the optimizer, fresh-process reload with fixed tolerance,
  explicit export allowlist with reload-verified checksums. No defect that corrupts a result was found.
- **Promise fulfilment:** every promised default stage runs. The controlled comparison runs but its displayed
  evidence does not support the reading the notebook asks for (M1). BYOD runs but its result is not delivered to
  the learner (M2). md-09 says precision/recall/area are reported; they are only in exported JSON (m2).
- **Scientific validity:** comparisons are held to equal prompts and frozen selections; test data never selects.
  The activity's manipulation is not uniform across images because of edge clipping, and nothing displayed lets
  the learner see that (M1). No dispersion for the SAM vs SAM 2 difference on 40 images (S2).
- **Learner experience:** good prediction/interpretation rhythm. Friction: duplicated and long outputs (m4),
  unlabelled infrastructure and no how-to-use or glossary (m6, m7), a download link that may not work on Colab
  (m8), no time estimate although one was measured (m9).
- **Spec conformance:** unresolved `MUST`s: **DAT19** (m1), **ENV3** (m5), **REL12** (BYOD positive and negative
  validation on a supported runtime: not recorded). RUN8 partially (m3). GDL2/6/11 `SHOULD`s (m6, m7).

## 3. Findings

### SEG-M1 — Major: the box-precision activity hides the boxes it actually used

- **Cell/section:** §7 "Controlled activity: make the box less precise" (md-15, code-16, md-17); carried
  `workshop.py` `sam_stage` (`box_experiment.json`).
- **Observed issue:** code-16 prints `box_experiment.json` — per image only `delta_iou`/`delta_dice` and three
  counts. The clipped coordinates and realised expansion go to `controlled_experiment.json`, which is never
  displayed. md-17 asks the learner to notice "boxes whose expansion is limited by the image boundary". FoodSeg103
  targets are large: 8 of 12 activity boxes already touch an edge at 0%.
- **Consequence:** the learner reads "1 improved / 11 worsened at +25%" (hosted record) as the effect of a less
  precise box. For most images the box grew on only some sides, and three boxes became the whole image, where the
  CPU pre-flight saw SAM 2 select the inverse region. The one controlled comparison the notebook promises
  ("run a controlled comparison") is interpreted without the variable that dominates it.
- **Evidence:** direct execution (P01): with the real pinned activity images, `expand_box` clips 10/12 boxes at
  +10% and 11/12 at +25%, 3 of them to `[0, 0, W, H]`; the `box_experiment.json` entries written by `sam_stage`
  (stand-in SAM 2) carry no clipping field, and code-16 displays nothing else. Documented execution evidence:
  release-verification row 2 attributes the worsened cases to "boxes expanded to the image edges".
- **Recommended correction:** record, per image and expansion, the requested and realised expansion, the clipped
  sides, whether the box became the whole image and the box's area fraction; add per-expansion counts of clipped
  and whole-image boxes and improved/worsened counts split by clipped vs not clipped; display a compact per-image
  table (IoU at 0 / 10 / 25%, clipped sides) instead of the raw JSON; update md-15/md-17 to predict and explain
  edge-limited and whole-image boxes without asserting an outcome.
- **Acceptance check:** after `sam_stage('sam2')` (stand-in model allowed), every `box_experiment.json` record
  has `clipped_sides`, `whole_image_box` and `realised_expansion_pixels`, and each expansion entry has
  `clipped_records` and `whole_image_records`; for a box touching the left edge, `clipped_sides` contains `left`;
  code-16 displays the clipped sides per image; md-17 names edge-limited and whole-image boxes.

### SEG-M2 — Major: the BYOD branch never shows the learner's results

- **Cell/section:** §9 BYOD, code-20; `run_stage` in code-04; `workshop.py` `main` (`byod`).
- **Observed issue:** after the six child stages, code-20 shows only `FileLink(.../segmentation_results.zip)`.
  The child report's summary is a single JSON line longer than 2,000 characters, so `run_stage` does not print it
  live, and the 7,000-character log tail that follows starts mid-line. No `summary.md`, score table, reload
  result or epoch is rendered for the BYOD run (the default path renders `summary.md` in code-18).
- **Consequence:** a learner who brings their own data — the transfer step (DAT13, completion/transfer) —
  finishes with no readable result in the notebook and must download and open JSON to learn whether adaptation
  helped on their data.
- **Evidence:** source inspection of code-20 and code-04 (P02); report line length inferred from the number of
  summary groups (27 model/protocol/role groups). BYOD has never run on Colab (not verified).
- **Recommended correction:** let the display helpers take a run root; after the BYOD run render the child's
  `summary.md`, the test score tables and the reload result, and print where the bundle is.
- **Acceptance check:** code-20 renders `<byod_root>/outputs/summary.md` and calls `show_score_table` with the
  BYOD root; with a stand-in BYOD root holding the default output files, the cell's helpers display the BYOD
  run's numbers, not the default run's.

### SEG-m1 — Minor (DAT19 MUST, UX10): BYOD validation errors misname or omit the offending record

- **Cell/section:** `workshop.py` `prepare` (BYOD loop) and `validate_roles`.
- **Observed issue:** `validate_roles` validates one record at a time with `samples.validate_dataset([entry])`,
  so every message says `records[0]`. Header checks (size, sides, EXIF, binary mask, paths) name no record. A
  missing `image`/`mask` key raises a bare `KeyError`; unknown role and duplicate messages name no record.
- **Consequence:** with 13–256 records the learner cannot tell which file to fix; the message points at the
  wrong record.
- **Evidence:** direct execution (P03): mask size mismatch at index 5 → `records[0].mask: shape (20, 20) does not
  match …`; missing prompt at index 7 → `records[0] is missing 'prompt'`; missing `mask` key at index 4 →
  `KeyError: 'mask'`; 8-px image at index 3 → `BYOD image/mask sides must be 16–4096 pixels`. A valid 13-record
  directory is accepted.
- **Recommended correction:** prefix every BYOD/role error with the record index and id; check required keys and
  the manifest shape up front with a named message.
- **Acceptance check:** each of the four P03 cases raises `ValueError` whose message names `records[i]` with the
  correct index and the record id, and never `records[0]` for another record; the valid directory still passes.

### SEG-m2 — Minor (EVAL15): score tables omit the filled-box baseline and the secondary rates

- **Cell/section:** `show_score_table` (code-04); §4 code-10, §6 code-14; md-09, md-15.
- **Observed issue:** md-15 tells the learner to use the filled-box baseline to see "what the model adds beyond
  localization", but the §6 table reads only `sam_metrics.json`/`sam2_metrics.json`. md-09 says "We also report
  Dice, foreground precision/recall, area fractions"; tables show IoU and Dice only.
- **Consequence:** the learner cannot make the comparison the note asks for until §8, and never sees CLIPSeg's
  under-segmentation (precision vs recall) in the notebook.
- **Evidence:** source inspection (P04).
- **Recommended correction:** add mean precision (over images with predicted foreground), recall and predicted
  area to the table; include the empty baseline in §4 and the oracle filled box in §6.
- **Acceptance check:** the §6 table has an `oracle_filled_box` row; both tables have precision, recall and
  predicted-area columns computed from the per-image rows.

### SEG-m3 — Minor (RUN8): the terminal summary omits reload, paired and activity results

- **Cell/section:** `workshop.py` `report` (`summary.md`), rendered by code-18.
- **Observed issue:** `summary.md` holds two test tables and the selected epoch. Reload parity, the paired
  frozen→adapted counts (computed into `paired_changes.json` after the summary is written) and the activity
  counts are absent. The design spec §6 asks for paired improved/unchanged/worsened counts.
- **Consequence:** the completion point does not state whether the artifact reloaded or how many test images the
  adaptation helped; BYOD (M2) inherits the gap.
- **Evidence:** source inspection (P05).
- **Recommended correction:** compute the paired changes before writing `summary.md` and add fresh-reload,
  paired-change and box-activity lines.
- **Acceptance check:** `report()` on a complete synthetic run writes a `summary.md` containing "Fresh reload",
  the paired improved/unchanged/worsened counts that equal `paired_changes.json`, and the per-expansion activity
  counts.

### SEG-m4 — Minor: duplicated and very long output

- **Cell/section:** `run_stage` (code-04); code-10, code-12.
- **Observed issue:** `run_stage` streams each stage's lines, then reprints the last 7,000 characters of the same
  log on success. code-12 prints `reload_parity.json` in full (92 records, about 470 lines); code-10 prints the
  raw adaptation record including every trainable tensor name.
- **Consequence:** the learner scrolls past repeated and machine-oriented output to find the result the "What to
  notice" refers to (framework dimension 7).
- **Evidence:** source inspection (P06).
- **Recommended correction:** print the log tail only on failure; show a compact training table (epoch, loss,
  validation mIoU, decoder change, selected epoch, optimizer steps) and a one-line reload summary; keep the files.
- **Acceptance check:** `run_stage` prints the log tail only when the stage fails; code-10 and code-12 no longer
  call `show_json` on those files and their helpers print at most one line per epoch / one summary line.

### SEG-m5 — Minor (ENV3 MUST): the `transformers` version is not displayed

- **Cell/section:** code-04 environment check.
- **Observed issue:** only `sys.version` and `torch.__version__` are printed; `transformers` (which implements
  all three models) is recorded only in the exported `provenance.json`.
- **Evidence:** source inspection (P07).
- **Recommended correction / acceptance check:** the check prints the `transformers` version and the CUDA
  version torch was built with.

### SEG-m6 — Minor (GDL2, GDL11): infrastructure cells unlabelled; no "How to use this notebook"

- **Cell/section:** md-00, md-01, code-02, code-04.
- **Observed issue:** code-02 and code-04 show raw code with no `# @title Infrastructure` and code-02 is not
  collapsed; the opening has no how-to-use note on form fields, infrastructure cells or where outputs go.
- **Evidence:** source inspection (P08).
- **Acceptance check:** code-02 and code-04 start with `# @title Infrastructure: …` and are form-collapsed; md-00
  has a "How to use this notebook" list covering Run all, infrastructure cells, form fields and output location.

### SEG-m7 — Minor (GDL6): no glossary

- **Observed issue:** recurring terms (semantic class mask vs instance, decoder, adapter, oracle prompt, candidate
  mask, predicted quality, threshold, logits) are spread across sections with no collected glossary.
- **Evidence:** source inspection (P09).
- **Acceptance check:** a collapsible glossary in the opening defines at least those eight terms.

### SEG-m8 — Minor: the results link may not download in Colab

- **Cell/section:** code-18 (and code-20).
- **Observed issue:** the only download route offered is `FileLink` to an absolute runtime path. In Colab such a
  link generally does not serve the file; the learner is not told to use the Files sidebar.
- **Evidence:** source inspection (P10); Colab behaviour not verified in this review (inferred).
- **Acceptance check:** the cell prints the bundle path relative to the working directory and the text explains
  downloading it from Colab's Files sidebar.

### SEG-m9 — Minor (UX12): stale runtime statement

- **Cell/section:** md-00.
- **Observed issue:** "cold runtime, memory and disk requirements still need fresh-Colab measurement", although a
  393 s fresh T4 run is recorded for this blob (memory and disk were not captured).
- **Evidence:** source inspection vs `docs/release-verification.md` (P11).
- **Acceptance check:** md-00 states the measured time with its environment and date and says peak memory and
  disk are not yet measured.

### Suggestions (optional, not release requirements)

- **SEG-S1 (GDL10):** let the learner choose the expansion (a form field and a small re-run cell) so the activity
  is learner-changed rather than pre-run.
- **SEG-S2:** paired SAM vs SAM 2 win/loss counts and a bootstrap interval for the 40-image difference.
- **SEG-S3:** accept a zip archive for BYOD in addition to a directory.
- **SEG-S4:** display per-call latency (recorded in every row) separately from load and install time, as the
  design spec §6 describes.

## 4. Positive findings and non-findings

- Selection never touches test labels; epoch zero is an allowed, labelled result.
- Reload runs in a new process (PID checked) with tolerance fixed before execution and metric parity.
- The oracle nature of point/box prompts and the semantic-union target are stated before the results.
- Not a finding: the point prompt covers one component of a multi-component target (12 of 40 test targets) —
  documented in §2 and §6 as part of the protocol.
- Not a finding: fp16 cached encoder activations during adaptation — disclosed in `clipseg.json` and does not
  touch the evaluation path.

## 5. Promise-to-evidence matrix

| Claim / objective | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Validate images and freeze roles | `prepare` / `validate_roles` | roles printed, reference previews | md-05/md-07 | Delivered (direct + hosted) |
| Interpret IoU / Dice | code-08 worked example | dict of TP/FP/FN/IoU/Dice | md-09 answer | Delivered |
| Adapt CLIPSeg, lock selection | `clipseg` | history JSON, test table | md-11 | Delivered; output volume m4, rates m2 |
| Export + fresh reload | `reload` | parity JSON | md-13 | Delivered; volume m4; not in summary m3 |
| Compare SAM vs SAM 2 under equal prompts | `sam_stage` | test table, previews | md-15 | Delivered; filled-box baseline missing m2 |
| Run a controlled comparison | `sam_stage` box expansion | deltas JSON | md-17 | **Partially** — M1 |
| Conclude with evidence | `report`, md-19 template | `summary.md` | conclusion template | Delivered; summary thin m3 |
| Own labelled data | `byod` | FileLink only | — | **Not delivered to learner** — M2; messages m1 |
| Unlabelled single image | `infer` | results JSON + masks | md-21 | Source-inspected only |

## 6. Readiness

**Needs revision.** Open Majors SEG-M1, SEG-M2; unresolved `MUST`s DAT19 (m1) and ENV3 (m5). After fixes,
readiness becomes **Verification pending** until a hosted Colab T4 run of the fixed notebook covers: default
`Run all`; BYOD with a valid directory plus one rejected input (REL12); the unlabelled branch; the report
download. Status stays Candidate; promotion is a human decision.

## 7. Verified versus inferred

- Verified by direct execution: M1 clipping counts and missing fields (P01), m1 messages (P03), the real
  `prepare` stage and filled-box baseline value.
- Verified by source inspection: M2, m2–m7, m9.
- Inferred: the BYOD log-tail truncation length (M2); Colab `FileLink` behaviour (m8).
- Only Kurt can confirm: whether learners read the activity as intended once clipping is shown.
- **Finding most likely to be wrong:** SEG-m8 — Colab may render `FileLink` to an absolute `/content/...` path as
  a working link in some configurations; the fix (printing the path and naming the Files sidebar) is harmless
  either way.

Probe archive: `DIMER_MultiModel_Image_Segmentation_Workshop_Review_Probes.zip` (`run_probes.py`,
`results.json`, `source_manifest.json`).
