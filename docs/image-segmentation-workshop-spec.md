# DIMER Multi-Model Image Segmentation — Notebook Specification

Status: **Proposed implementation specification. No new notebook, benchmark result, or Colab qualification is claimed.**

Prepared: 2026-09-27. Normative basis: DIMER NOTEBOOK_SPEC 2.2.

## 1. Recommended unit

**DIMER Notebook: Image Segmentation with Text, Points, and Boxes**

Driving question: **Which pixels belong to the target, and how does the information we give the model change its answer?**

Place this in **Level 2 — Applied tasks**, after Open-Vocabulary Object Detection and before Multi-Model Image Matching. Explain classification → label, detection → box, segmentation → pixel mask. Basic Python and Colab familiarity are sufficient; no prior segmentation knowledge is assumed.

| Field | Proposed decision |
|---|---|
| Host repository | `kurtvalcorza/clipseg-segmentation-pipeline`, because its decoder adaptation and artifact lifecycle anchor the complete workflow |
| Artifact | `tutorials/DIMER_MultiModel_Image_Segmentation_Workshop.ipynb` |
| Profile / mode | `E2E` / `WORKSHOP`, usable independently |
| Models | CLIPSeg rd64-refined, SAM ViT-B, SAM 2.1 Hiera-Small |
| Default adaptation | CLIPSeg decoder only; both SAM models remain frozen |
| Hardware | Fresh Colab T4-class GPU; sequential model residency |
| Default dataset | Pinned FoodSeg103 subset with ingredient masks |
| Core results | Reference/prediction overlays, IoU and Dice, controlled prompt comparison, frozen/adapted comparison, adapter export and fresh reload |
| Learner record | Optional personal completion notes; no required submission |

Catalogue description:

> Explore how text, points, and boxes guide image segmentation. Inspect pixel masks, compare SAM and SAM 2 under the same prompts, adapt CLIPSeg to a small labelled dataset, and evaluate whether the change helps on held-out images. Export and reload the adapted model, then explain its errors and limitations.

These are recommended design choices for review before implementation. This specification does not modify the curriculum count or publish a Colab link for an artifact that does not yet exist.

## 2. Verified starting point

GitHub `main` heads checked on 2026-09-27 match the inspected local sources:

| Repository | Commit | Model revision |
|---|---|---|
| CLIPSeg | `2b004834303160e19ad2aef37b9ac7eeb996476c` | `CIDAS/clipseg-rd64-refined` at `999e0328d9e10b484360c477313983f9afdd7050` |
| SAM ViT | `ed74a93f0c743dd184efe4f88b23f71b772e4d38` | `facebook/sam-vit-base` at `70c1a07f894ebb5b307fd9eaaee97b9dfc16068f` |
| SAM 2 | `df023e1af8c0f49e74af3f308e3e50b236e15850` | `facebook/sam2.1-hiera-small` at `ee5bba1d82bb8749febdf90f45e84b687142ba03` |

The local fleet inventory dated 2026-09-26 lists these three pipelines as Live. That inventory is not a fresh DIMER service availability probe.

The CLIPSeg implementation supports phrase-conditioned masks, bounded decoder adaptation with frozen image/text towers, validation-selected epochs including epoch zero, and safetensors adapter export/reload. The inspected SAM wrappers support image prompts; this unit makes no video-tracking claim. Upstream SAM 2 documents point and box prompting and multiple candidate masks ([model card](https://huggingface.co/facebook/sam2.1-hiera-small)).

Reuse source behavior through the notebook generator. The released notebook must carry its reference implementation, model manifests, and configuration. Runtime execution must not clone repositories, fetch Python source, import an installed DIMER pipeline package, or call a DIMER worker/service.

**Source issue to resolve before release:** CLIPSeg's README identifies its weight license as Apache-2.0 while the fleet inventory contains an earlier MIT licensing decision. Verify the pinned upstream weight terms and reconcile the provenance records; do not equate the original code license with the weight license.

## 3. Learning outcomes and boundaries

Learners should be able to:

1. Identify the image, prompt, reference mask, predicted mask, and probability/logit map.
2. Distinguish semantic class masks from instance masks and promptable object selection.
3. Calculate IoU and Dice on a small worked pixel grid.
4. Explain why a precise box supplies more location information than a class phrase.
5. Make and test a prediction by changing one prompt variable.
6. Distinguish decoder adaptation from full-model training and assess held-out changes.
7. Reload an exported adapter and verify that predictions match.
8. Conclude with measured evidence, failure examples, and limits on generalization.

Out of scope: video tracking, automatic segmentation of every object, panoptic segmentation, medical or geospatial validation, training SAM, detector-to-segmenter composition, and model-wide benchmark rankings. Detector-generated boxes are a later composed-systems exercise.

## 4. Data and target contract

Use the existing verified FoodSeg103 acquisition path. Source: [FoodSeg103 dataset card](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103). Existing repository constants pin dataset revision `176acc3edd2432ee126bda6fb01469eadeb018df` and eight row groups with digest verification. Preserve those integrity checks; freeze the new selection in a notebook manifest.

Proposed bounded selection: **212 distinct images**, comprising 120 training, 40 validation, 40 final test, and 12 activity/new-input images. These are initial resource-budget choices, not measured feasibility claims. If changed during preflight, update the spec and manifest before the qualifying run.

- Use a deterministic seed (42), stable candidate ordering, and a fixed selection manifest containing source IDs, row-group identities, image/mask digests, class IDs, phrases, and roles.
- Follow the existing target rule: choose the largest eligible ingredient class by labelled pixel area, excluding background/other; ties follow the existing deterministic class-ID rule. The binary target is **all pixels of that class**, including disconnected regions. It is not a single-instance annotation.
- Do not silently replace the target with its largest connected component. Show a disconnected example and explain why one SAM prompt may capture only one region.
- Deduplicate by decoded image identity before allocating roles. Related derivatives remain in one role. Report exclusions, invalid records, duplicate groups, and the final per-role counts.
- These are tutorial train/validation/test roles carved from the pinned source pool, whose current source IDs identify a validation slice. Do not call the result an official FoodSeg103 test benchmark. Unknown upstream pretraining overlap prevents a claim of completely unseen data.
- Masks must align with decoded images, use documented binary values, and have at least one foreground pixel for the labelled core task. Validate dimensions, class mapping, image orientation, finite data, and uniqueness before model execution.
- Preserve original-resolution references. Resize labels with nearest-neighbor interpolation only; resize model logits/maps with each verified wrapper's documented postprocessing before thresholding. Never use a resized display overlay as the scoring reference.
- Freeze all roles before adaptation. Test references are reserved for final scoring and the explicitly labelled geometric-prompt simulator; they never select thresholds, epochs, examples, or model settings.

Food photographs make an accessible task, but this narrow domain does not establish performance on people, roads, documents, or remote sensing. Record dataset provenance and applicable terms; do not assume a dataset-host tag settles every underlying image's reuse rights.

## 5. Comparisons and prompt fairness

There are **two separate comparisons**, not one overall leaderboard:

| Comparison | Information held constant | Question |
|---|---|---|
| Frozen vs adapted CLIPSeg | Same images, ingredient phrases, reference masks, threshold, and preprocessing | Did bounded domain adaptation help? |
| SAM ViT-B vs SAM 2.1 Hiera-Small | Same images, exact geometric prompts, references, and scoring rules | How do these two models respond to the same assistance? |

Cross-family displays are illustrative and retain their prompt labels. Never infer that a box-prompted SAM score demonstrates better autonomous recognition than a text-prompted CLIPSeg score.

Define and export the prompt protocols before scoring:

- **Text:** exact canonical ingredient phrase from the manifest, with no per-test rewriting.
- **Point:** one positive pixel in the largest connected target component, chosen by maximum distance from its boundary; resolve ties in row-major order. The reference-derived point is a simulated informed user click.
- **Box:** tight rectangle around the full target-class union, clipped to the image, expressed in each wrapper's verified coordinate convention. Export a canonical pixel-coordinate representation and its conversion. The reference-derived box is oracle-assisted localization.
- Score the two SAMs separately under point and box protocols. Both receive byte-identical canonical prompts. Do not combine point and box unless introducing a separately labelled protocol.
- Select multiple candidate masks using highest model-predicted quality with lowest candidate index on ties. Never select the candidate with highest reference IoU. A predicted quality score is not measured IoU or calibrated certainty.
- The point and box protocols deliberately expose how target ambiguity and disconnected class regions affect results. Record that the input annotation is semantic while SAM prompting may select an object or part.

A fixed empty-mask baseline demonstrates why background accuracy is misleading. A filled-box baseline accompanies the oracle-box protocol only: it uses the same localization assistance, so it cannot be presented as an unassisted baseline.

## 6. Metrics and interpretation

For each image, display `IoU = TP / (TP + FP + FN)` and `Dice = 2TP / (2TP + FP + FN)`, with a small hand-worked example. Score foreground pixels on the original image grid.

- Primary: **mean per-image target IoU**. Label it explicitly; it is not mean IoU over all 103 semantic classes.
- Secondary: mean per-image Dice, pooled foreground IoU, pixel precision and recall, predicted area fraction, and failure count.
- Empty prediction against a positive reference has IoU/Dice zero. Precision with no predicted positives is undefined: store null and display the reason. Recall is defined for the positive references in this unit.
- For optional empty-reference fixtures, both-empty IoU/Dice are undefined and excluded from overlap means with counts shown; false-positive empty-reference cases score zero overlap. Separately report empty-reference false-positive frequency.
- Always report metric denominators and exclusions. Do not silently drop failed model calls or mismatched masks to produce a better score.
- Show paired per-image changes, their distribution, and improved/unchanged/worsened counts. Small tutorial samples support descriptive conclusions, not broad superiority claims.
- Select displayed examples by fixed IDs plus deterministic best/worst/error rules. Do not show only successful masks.
- Display inference latency separately from installation, downloads, model loading, and adaptation. Use GPU synchronization and declared warm-up rules; record hardware and precision.

CLIPSeg's default binary threshold is fixed at 0.5 for both frozen and adapted headline results. An optional validation-only threshold experiment may examine 0.3/0.5/0.7; any chosen threshold must be locked before final test scoring and reported as a separate comparison.

## 7. Guided sequence

Every major section follows **orient → Input / Model or System / Output → predict → run → notice → interpret**. Experiments add **change one thing → compare → conclude with evidence and limitations**. Predictions/reflections are optional text entries with worked guidance and must not block Run all.

| Stage | What runs | What the learner does |
|---|---|---|
| 1. Orient | Task diagram and example image/mask | Identify what segmentation adds to a detection box |
| 2. Prepare | Runtime checks, isolated environment, pinned model/data acquisition | Read resource and provenance information |
| 3. Validate | Input checks, role manifest, target overlays | Notice class union versus individual object |
| 4. Establish baselines | Empty masks and frozen CLIPSeg on validation | Predict whether pixel accuracy could be misleading |
| 5. Compare interaction modes | SAM and SAM 2 on fixed validation/activity examples | Compare identical points and identical boxes; explain assistance |
| 6. Adapt | Bounded CLIPSeg decoder training and validation selection | Inspect loss and validation curves; distinguish training from validation |
| 7. Lock and evaluate | Final frozen/adapted CLIPSeg and both SAM prompt protocols on test | Report paired changes and failure cases without retuning |
| 8. Controlled experiment | SAM 2 box-expansion study on activity images | Predict the effect of less precise localization |
| 9. New input | All three interfaces on reserved activity images | Inspect unfamiliar inputs; do not imply unsupported generalization |
| 10. Export and reload | Adapter, outputs, manifest, fresh-process verification | Explain what was saved and what parity establishes |
| 11. Conclude | Worked evidence-based conclusion and optional personal record | State one finding, one counterexample, and one limitation |
| 12. Optional BYOD | Labelled adaptation route and unlabelled inference route | Apply the same contracts to compatible data |

Controlled experiment: keep SAM 2, images, selection rules, and all settings fixed; expand each tight box by **0%, 10%, and 25% of its width/height on each side**, clipping at image boundaries. Use the 12 activity images, not the test set. Export resulting coordinates and actual expansion after clipping. Show paired IoU/Dice and overlays. Wider boxes may help, harm, or leave results unchanged; no improvement assertion is permitted.

## 8. Bounded adaptation and artifacts

The default Run all path must actually adapt CLIPSeg (RUN7/FT2). Initial proposed budget: 4 epochs, batch size 4, learning rate `3e-4`, seed 42, existing decoder BCE loss and gradient clipping. Freeze image/text encoders, cache their activations, and record trainable/total parameters. Confirm feasibility during hosted preflight; do not silently change batch size, skip training, or swap models when resources run short.

Use validation mean target IoU at threshold 0.5 for epoch selection, including the frozen epoch-zero reference; keep the earliest epoch on ties. Test metrics cannot influence selection. A selected epoch zero is a valid negative result and must not be concealed. Record actual optimizer steps, nonzero training updates, and selected checkpoint identity separately.

Export the selected decoder state with the existing safetensors adapter contract, including the base revision, base-file hashes, threshold, tensor schema, training configuration, selection reason, and history. If epoch zero wins, label the artifact as a selected frozen decoder; do not claim it contains an improvement or selected learned changes.

Reconstruct the pipeline in a new process from verified base files and exported adapter only. Score fixed parity probes and the selected test evaluation through the reloaded artifact. Compare probabilities with a stated numerical tolerance chosen before qualification, require identical binary masks for the qualification probes, and report metric parity. A mismatch fails verification; do not widen tolerance after seeing a failure without diagnosis and a new documented run.

Export a downloadable bundle containing `run_summary.json`, sample/role manifest, prompt manifest, per-image metrics, comparison tables, training history, representative PNG masks/overlays, adapter safetensors and manifest, and checksums. Exclude credentials and optional private BYOD images unless the learner explicitly requests their export.

## 9. Runtime and user-data paths

Use one isolated Python 3.12 worker environment and an explicit compatible dependency lock. Keep the Colab host kernel's preloaded NumPy/framework stack intact. Existing repository pins are inputs to compatibility testing, not proof that the combined notebook runs. Carry model/source namespaces separately to prevent helper-name collisions.

- Fresh default Run all must require no manual restart, file upload, payment, DIMER account, or secret. Public model/data acquisition uses anonymous access by default. Optional `HF_TOKEN` comes from Colab Secrets without printing or persisting it.
- Acquire only pinned model/data files, verify digests before loading, and show bounded retries and progress. Refuse integrity failures; no unpinned fallback.
- Load one model at a time, persist CPU results, and free GPU allocations before the next. Show actual peak memory and elapsed time in release evidence. Learner time and download estimates remain pending preflight measurement.
- Expose meaningful options as Colab form fields. Give file-path fields for scripted/BYOD use; nonempty paths must not trigger upload widgets.
- Keep default controls coherent: changing the data source must enter the appropriate validation/split/adaptation route rather than replacing only the final inference image.
- Add actionable troubleshooting for GPU absence, incompatible packages, download/integrity failures, memory exhaustion, empty predictions, misaligned masks, invalid prompts, and malformed BYOD.

**Labelled BYOD:** accept a manifest plus RGB images and single-channel binary masks, with unique `id`, `image`, `mask`, `prompt`, and optional source/group identity. Explain size/phrase limits before selection; validate against the reused wrapper contract. Support disjoint train/validation/test roles and reserved new-input examples, group related images, reject overlap, and run validate → split/preprocess → baseline → decoder adaptation → evaluation → inference/export → fresh reload. Minimum role counts must be explicit and tested; they are structural requirements, not sufficient evidence of generalization.

**Unlabelled BYOD:** offer clearly labelled inference-only image + phrase/point/box controls. Without reference masks, show outputs and area statistics without fabricated accuracy metrics. It supplements rather than replaces the required labelled adaptation route (DAT14).

Both paths execute locally inside the hosted runtime. State that Colab is a third-party hosted environment; inputs are not transmitted to DIMER or a model inference API. No notebook telemetry or automatic upload of participant data.

## 10. Release acceptance

The new notebook remains Candidate until all applicable NOTEBOOK_SPEC 2.2 requirements and these checks pass:

1. **Standalone generation:** generator output is reproducible; embedded code parses, uses isolated namespaces, and needs no runtime source checkout or service.
2. **Provenance:** all three model revisions and files, dataset bytes, selection manifest, terms, and the CLIPSeg license discrepancy are resolved and recorded.
3. **Fresh execution:** the exact candidate notebook completes Colab default Run all from a fresh T4 runtime with no secrets, upload, restart, or manual rerun; record source blob and executed output hash.
4. **Data separation:** actual counts match the frozen manifest; decoded duplicates/related records cannot cross roles; test data cannot select settings.
5. **Prompt protocol:** geometric assistance is disclosed; prompts match between SAMs; reference masks never select model candidates. Coordinate conversion and disconnected-target examples are verified.
6. **Metric correctness:** independent tiny fixtures cover known overlaps, disjoint masks, empty predictions/references, pooled versus mean scores, and original-grid alignment.
7. **Real adaptation:** default execution performs optimizer updates; frozen towers remain frozen; validation selection includes epoch zero and records ties and failures honestly.
8. **Controlled experiment:** exported activity results change only box expansion; all fixed/clipped coordinates and denominators are recorded.
9. **Artifact verification:** exact tensor schema/digests are checked before loading; a corrupt artifact fails clearly; fresh-process prediction/mask/metric parity passes.
10. **BYOD evidence:** representative labelled data reaches adaptation, evaluation, export, and reload; unlabelled inference works; malformed masks, out-of-bounds prompts, and role overlap are rejected clearly (REL12).
11. **Guided content:** every substantive output has an explanation of what to notice and a worked interpretation; exercises, limitations, troubleshooting, optional completion note, and AI disclosure are present.
12. **Honest release record:** report failures, resource measurements, selected epoch, and scope. Neither prior repository runs nor tests with mocked models qualify this new notebook.

No minimum accuracy improvement is required for release. Correct execution, valid comparisons, and honest evidence are required.

## 11. Disclosure and sources

Include the following in the notebook:

> **AI Assistance Disclosure**  
> Generative AI assisted with this notebook's code and instructional content under maintainer direction. The maintainer remains responsible for review, validation, and release decisions. AI-generated material may contain errors; validation claims are limited to documented runs and configurations. AI use does not imply independent verification, provider endorsement, or release approval.

Implementation references:

- [DIMER NOTEBOOK_SPEC](https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md), inspected version 2.2; freeze the exact normative blob at implementation time.
- [CLIPSeg repository at inspected revision](https://github.com/kurtvalcorza/clipseg-segmentation-pipeline/tree/2b004834303160e19ad2aef37b9ac7eeb996476c), especially README, model adapter/export code, samples.py, and notebook generator.
- [SAM ViT repository at inspected revision](https://github.com/kurtvalcorza/sam-vit-segmentation-pipeline/tree/ed74a93f0c743dd184efe4f88b23f71b772e4d38).
- [SAM 2 repository at inspected revision](https://github.com/kurtvalcorza/sam2-segmentation-pipeline/tree/df023e1af8c0f49e74af3f308e3e50b236e15850).
- [SAM 2 upstream model card](https://huggingface.co/facebook/sam2.1-hiera-small) and [FoodSeg103 dataset card](https://huggingface.co/datasets/EduardoPacheco/FoodSeg103).

Before implementation, confirm the proposed host and default adaptation scope. During preflight, freeze sample IDs, the combined runtime lock, resource budget, BYOD minimums, and reload tolerances. The main unverified assumption is that the proposed three-model workflow and four-epoch adaptation fit a practical fresh-Colab session; qualification must measure this rather than inherit an older notebook's timing.
