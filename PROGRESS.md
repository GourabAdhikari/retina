# PROGRESS.md

## 2026-06-19

### Completed
- Read `AGENTS.md`, `PLAN.md`, `TREE.md`, and inspected the empty `main.ipynb`.
- Converted `main.ipynb` into the runnable project entry point for **Phase 1 — Dataset Analysis** only.
- Added notebook cells for:
  - configuration via `RASTA_DATA_ROOT` and clinical spreadsheet path,
  - clinical table loading,
  - RASTA patient-eye manifest scanning,
  - class distribution, bilateral eye availability, layer availability, and leakage checks,
  - clinical missingness heatmap and demographic summaries,
  - image resolution / lightweight quality audit,
  - biomarker summaries and cohort boxplots,
  - preliminary leakage-safe patient-level fold assignment for review.
- Added lightweight supporting package files under `src/rasta_cad/` to keep notebook code clean and reusable.

### Notes
- The notebook was **not run**, per project instruction and because the dataset is not present on this machine.
- Current implementation scope is Phase 1 only; later phases should be added incrementally after reviewing Phase 1 outputs on the dataset/GPU machine.

### Adjustment
- Made `main.ipynb` fully self-contained and removed all dependency on helper Python files under `src/`.
- Future work will keep implementation inside notebook cells only, phase-by-phase.
- Updated the notebook configuration cell to make Windows dataset/clinical paths easy to edit via `DATA_ROOT_STR` and `CLINICAL_PATH_STR`.

### Phase 2 progress
- Added a new **Phase 2 — Preprocessing Pipeline and Dataset** section to `main.ipynb`.
- Implemented notebook-local OCTA preprocessing:
  - image corruption / zero-variance / low dynamic-range / SNR QC with exclusion logging,
  - optional CNN-quality hook for a future lightweight scan-quality classifier,
  - percentile 2nd–98th normalization,
  - CLAHE with OpenCV fallback to scikit-image/no-op,
  - bicubic resize to 224×224,
  - training-fold-only OCTA layer mean/std computation.
- Implemented clinical preprocessing:
  - fold-fitted continuous z-scoring,
  - binary and ordinal feature handling,
  - missingness masks instead of mean/median imputation,
  - trainable `ClinicalMaskTokenImputer` for learned per-feature missing tokens and ordinal embeddings.
- Implemented Phase 2 artifacts:
  - QC-passed manifest and full QC exclusion logs,
  - patient-level stratified folds with zero patient-overlap validation,
  - fold-wise clinical preprocessor JSON files,
  - fold-wise clinical mutual-information reports,
  - fold-wise OCTA layer normalization statistics.
- Implemented `RastaPhase2Dataset` returning the planned sample dictionary with synchronized Sup/Deep/CC augmentation and zero placeholder masks when segmentation masks are unavailable.

### Phase 3 progress
- Added a new **Phase 3 — OCTA-SimCLR Pretraining** section to `main.ipynb`.
- Implemented layer-specific SimCLR pretraining components entirely inside notebook cells:
  - OD/OS bilateral natural-positive pair dataset with unilateral same-image augmented fallback,
  - conservative OCTA augmentations: horizontal flip, mild rotation, random resized crop, Gaussian blur, and mild intensity jitter,
  - 1-channel ConvNeXt-Tiny backbone initialized from ImageNet when available,
  - 2-layer projection head (768 → 256 → 128),
  - NT-Xent loss with temperature 0.07,
  - independent Sup/Deep/CC training loop with SGD, momentum, weight decay, cosine LR, AMP, gradient accumulation, checkpointing, and backbone-only export.
- Added Phase 3 pair-summary/audit outputs and a backbone registry for later Phase 4 loading.
- Set expensive CUDA pretraining gate to `RUN_PHASE3_PRETRAINING = True` for the GPU run.
- Added `PHASE3_REQUIRE_CUDA = True` so Phase 3 fails fast if accidentally launched in a non-CUDA runtime.
- Added Phase 3 bounded print logging in `main.ipynb`:
  - disabled tqdm/widget progress by default to avoid frozen notebook progress bars,
  - set default physical batch size to 128 and Windows-safe `PHASE3_NUM_WORKERS = 0`,
  - prints first 3 epochs, every 5 epochs, final epoch, and each completed layer-run with elapsed/ETA,
  - writes exact per-epoch history CSV after every epoch to avoid spamming notebook output.
- Added Phase 3 flat-loss safety diagnostics after observed loss stayed at `ln(255) ≈ 5.541`:
  - preflight cell checks one throwaway optimizer step, grad flow, parameter delta, embedding/feature variance, positive-vs-negative cosine stats, and saves `outputs/phase3_octa_simclr/phase3_preflight_diagnostics.csv`,
  - per-epoch history now records NT-Xent random-baseline loss and delta,
  - layer-run aborts at epoch 10 if loss remains within `0.02` of random baseline, saving a checkpoint first.
- After preflight showed `inf` gradients and zero parameter update, disabled Phase 3 AMP, lowered LR from `0.1` to `0.03`, added finite-gradient checks, and added gradient clipping at norm `1.0`.

## 2026-06-22

### Phase 3 resumability
- Made Phase 3 OCTA-SimCLR training resumable in `main.ipynb`.
- Changed Phase 3 config cell to enable automatic resume from `*_last.pt` and save the last checkpoint every epoch.
- Changed Phase 3 checkpoint/training-loop cell to:
  - centralize last/best/backbone checkpoint paths,
  - load existing `*_last.pt` automatically,
  - restore model, optimizer, scheduler, history, and RNG state,
  - continue at the next epoch,
  - treat already-complete checkpoints as complete and export the backbone without retraining.
- Changed Phase 3 launch cell to print resume settings before layer runs.

## 2026-06-23

### Phase 3 logging update
- Changed `main.ipynb` Phase 3 console logging to restore live visibility without tqdm.
- Phase 3 now prints:
  - run start/resume summary with remaining epochs and ETA if prior epoch timings exist,
  - `START epoch ...` before every epoch,
  - `END epoch ...` after every epoch with loss, LR, epoch time, average epoch time, elapsed time for the current resumed run, estimated remaining time, and estimated total layer-run time.
- Set `PHASE3_LOG_EVERY_N_EPOCHS = 1` so end-of-epoch logs print every epoch by default.

### Phase 1/2 fast reload
- Added a standalone fast-reload cell to `main.ipynb` before Phase 3.
- The cell restores imports, constants, preprocessing helpers, `manifest_clin`, `phase2_manifest`, `phase2_folds`, `clinical_preprocessors`, and `layer_stats_by_fold` from saved Phase 1/2 outputs.
- It checks required artifact files first and raises a clear message to rerun Phase 1/2 if any are missing.

### Planning update
- Updated `PLAN.md` to replace optional human-provided masks with self-generated OCTA pseudo-masks.
- Added classical vessel/FAZ/CC flow-deficit pseudo-mask generation, optional frozen segmentation-student refinement, mask QC, biomarker extraction, and classifier mask/biomarker fusion.
- Adjusted losses, ablations, contributions, and deliverables to reflect weak segmentation without manual ground-truth masks.
- Note: existing Phase 2 notebook placeholder masks now need replacement by generated masks in a later implementation pass.
- Updated HTML docs under `docs/` to reflect self-generated OCTA pseudo-masks, optional segmentation student, generated biomarkers, tabular fusion, and current notebook status through Phase 3.
- Added `docs/segmentation-pseudo-masks-explainer.html` as the dedicated explainer for classical masks, optional Attention U-Net/U-Net++ student refinement, mask QC, and dataset fields.

## 2026-07-02

### Phase 3 completion audit
- Confirmed Phase 3 output histories exist for all 15 SimCLR runs: 5 folds × Sup/Deep/CC.
- Each history has 100 epochs, finite losses, and loss decreased well below the NT-Xent random baseline (~5.541).
- Best losses: Sup ~1.54–1.80, Deep ~1.81–2.07, CC ~2.66–2.93.
- Confirmed positive-pair summaries exist for all folds/layers, with bilateral OD/OS positives plus unilateral self-augmented fallback.
- Confirmed `.pt` checkpoint/backbone files are intentionally absent from this copy because they are too large to move.
- Noted current local `outputs/phase2_preprocessing/` copy is incomplete for fold 3/4 preprocessor/stat artifacts; rerunning fast reload here would fail unless those artifacts are restored or regenerated.

### Phase 4 progress
- Reimplemented **Phase 4 — Image Encoders with Projection Heads** in `main.ipynb` after intentional removal of old Phase 4/5 cells.
- Added independent Sup/Deep/CC ConvNeXt-Tiny encoders with no shared backbone parameters.
- Added Phase 4 projection head: `Linear(768→512) → GELU → LayerNorm → Linear(512→256) → LayerNorm`.
- Forward output now includes ordered `layer_tokens [B,3,256]`, per-layer tokens, pooled 768-d features, and retained final spatial maps for Grad-CAM.
- Added fold/layer-specific Phase 3 backbone loading helpers with metadata validation and graceful fallback when large `.pt` files are absent locally.
- Added backbone availability audit at `outputs/phase4_image_encoders/phase4_backbone_availability.csv`.
- Added a clearly marked **SUMMARIZATION CELL — Status through Phase 3 only** before Phase 4, so Phase 1–3 artifact health can be checked without rerunning analysis/preprocessing/pretraining.

### Phase 5 progress
- Added **Phase 5 — Tabular Metadata Encoder** cells to `main.ipynb`.
- Implemented fold-aware tabular encoder support:
  - clinical learned missing-token imputation with one trainable scalar token per feature,
  - learned scalar embeddings for ordinal clinical fields (e.g. smoking),
  - optional generated OCTA biomarker stream with fold-fitted z-score stats,
  - learned biomarker missing tokens for missing/QC-failed biomarker values,
  - training-time random masking of 10% observed clinical features.
- Implemented Phase 5 architecture exactly as planned:
  - `LayerNorm(D_tab)` → `Linear(D_tab→128)` → `GELU` → `Dropout(0.3)` → `Linear(128→256)` → `GELU` → `LayerNorm` → `Dropout(0.2)`.
- Added utilities to infer Phase 8 biomarker columns when available, save fold-wise biomarker stats, save tabular schema JSON files, and write `outputs/phase5_tabular_encoder/phase5_tabular_encoder_summary.csv`.
- Added a no-image-read smoke test for output shape `[B, 256]` when Phase 2 artifacts are loaded.
- Revised Phase 5 after Phase 4 output review: biomarker detection now includes current `FAZ/Faz/DENS_/PERF_` table columns, and eye-specific biomarker columns ending `_OD`/`_OS` are aligned to each sample's `eye` instead of feeding both eyes as separate tabular features.

### Phase 6 progress
- Added **Phase 6 — Cross-Layer Attention Module** cells to `main.ipynb`.
- Implemented layer-identity embeddings for ordered Sup/Deep/CC tokens before attention.
- Implemented pre-norm transformer encoder block with 8 heads, 32-d head size, 256→1024→256 GELU feed-forward path, dropout 0.1, residuals, and mean pooling to `F_octa ∈ R^256`.
- Added attention extraction and layer-dominance utility for later explainability: per-sample Sup/Deep/CC dominance from attention received across heads/query tokens.
- Added `Phase6LayerAwareOCTAEncoder` wrapper to compose Phase 4 image encoders with Phase 6 attention.
- Added smoke test and saved Phase 6 summary/audit artifacts under `outputs/phase6_cross_layer_attention/` when run.

### Phase 7 progress
- Added **Phase 7 — Multimodal Cross-Attention Fusion** cell to `main.ipynb`.
- Implemented `Phase7CrossAttentionFusion`: OCTA query attends to tabular/biomarker context with 4-head cross-attention, residual add, LayerNorm, mask concat, and 512-d unified projection.
- Implemented `Phase7MaskEncoder`: 5 generated pseudo-mask channels (`sup_vessel`, `deep_vessel`, `sup_faz`, `deep_faz`, `cc_flow`) through a lightweight CNN to `F_mask ∈ R^128`.
- Added `phase7_stack_generated_masks` to stack Phase 2/8 mask tensors in canonical order.
- Added `Phase7MultimodalFusionModel` wrapper that can compose Phase 6 OCTA encoder, Phase 5 tabular encoder, and Phase 7 mask/fusion modules.
- Added no-image-read smoke test with placeholder generated masks and saved Phase 7 summary artifacts under `outputs/phase7_multimodal_fusion/` when run.

### Phase 8 progress
- Added **Phase 8 — Self-Generated Segmentation Masks and Biomarkers** cell to `main.ipynb`.
- Implemented deterministic classical pseudo-mask generation:
  - Sup/Deep vessel masks via percentile normalization, CLAHE-preprocessed input, Frangi/Sato vesselness, adaptive thresholding, morphology cleanup, and skeletonization.
  - Sup/Deep FAZ masks via central low-flow/avascular connected-component selection with area and centroid plausibility checks.
  - CC flow-deficit masks via local background correction, Sauvola/adaptive dark-region detection, and morphology cleanup.
- Added generated-mask QC for vessel density, FAZ area/centering, and CC flow-deficit fraction.
- Added Phase 8 biomarker extraction: vessel density, skeleton density, FAZ area/perimeter/circularity/centroid distance, CC flow-deficit fraction/count/mean area/median area.
- Added artifact generation utilities writing masks, `phase8_generated_masks_biomarkers.csv`, `phase8_mask_qc.csv`, summary JSON, and visual montage sheets.
- Added `phase8_merge_artifacts`, `RastaPhase8Dataset`, and `make_phase8_datasets_for_fold` so Phase 8 masks/biomarkers can feed Phase 5/7 without changing classifier code.
- Left `PHASE8_RUN_GENERATION = False` by default because local machine has no dataset; set it to `True` on the data machine after Phase 2/fast reload.
- Patched Phase 8 morphology cleanup for scikit-image 0.26+ deprecation warnings: replaced deprecated binary opening/closing calls and added compatibility wrappers for `remove_small_objects`/`remove_small_holes` threshold parameters.
- Reviewed newly added Phase 8 outputs: 476 samples, 321/476 QC pass (67.4%). Found montage generation bug (`plt` missing) causing 155 `phase8_error` entries and zero montage files; patched `main.ipynb` to import `matplotlib.pyplot as plt` and prevent montage failures from overriding mask QC.
- Reviewed rerun Phase 8 outputs: 476 samples, 355/476 QC pass (74.6%), 155 montage PNGs generated, no `phase8_error` column. Remaining failures are mostly vessel-density QC (Sup 49, Deep 86); FAZ failures rare (1 Sup, 3 Deep), CC flow passes all. Visual montage review suggests vessel masks are very sparse and mainly capture large/edge vessels, so classical vessel thresholding needs tuning before using student refinement.
- Updated `PLAN.md`: added explicit classical-mask improvement gate before U-Net/Attention U-Net/U-Net++; student refinement now allowed only after improved classical pseudo-labels pass QC/visual review. Added U-Net++ as optional stronger student, not current default.
- Rewrote Phase 8 vessel-mask extraction in `main.ipynb` for an improvement pass: multiscale vesselness + local contrast + top-hat scoring, adaptive/percentile candidate thresholds, lighter morphology, bottom-right watermark suppression, tighter vessel QC density range, and `mask_source='classical_improved'`. Added montage definition to `PLAN.md`.
- Made `PLAN.md` explicit that U-Net++ must not be silently skipped: after improved classical masks, agent must document Decision A/B/C. Default expectation is to evaluate U-Net++ unless classical-only is explicitly justified by QC/montages.
- Reviewed `classical_improved` Phase 8 outputs: 476 samples, 473/476 QC pass (99.4%). Vessel densities improved substantially (Sup mean 0.127, Deep mean 0.096). Remaining failures are 3 FAZ failures only. Found montage directory contains stale PNGs from prior run because old montages were not cleared; patched `main.ipynb` to delete old montage PNGs before generation. Current mask files themselves look improved via a fresh contact sheet; rerun Phase 8 once more to refresh montage sheets.
- Checked PLAN Step C and added explicit U-Net++ decision gate to `main.ipynb` as **Phase 8C/D** instead of moving to Phase 9.
- Set default Step C decision to `B_train_unetpp_student` with training disabled by default (`PHASE8_RUN_STUDENT_TRAINING = False`) until refreshed Phase 8 artifacts are available on data/GPU machine.
- Implemented compact U-Net++ student code in-notebook: QC-passed pseudo-mask dataset, Sup/Deep 2-head vessel+FAZ task, CC flow-deficit task, Tversky+Focal+Dice loss, Dice/IoU metrics, fold training loop, checkpoint/history artifact outputs, and decision summary CSV/JSON.
- Patched Phase 8C/D U-Net++ fold launch bug: `task` now remains `supdeep`/`cc`; fold-specific names are passed separately via `run_name` (e.g. `fold0_supdeep`).
- Patched Phase 8C/D missing import: added `import torch.nn.functional as F` for U-Net++ upsampling/loss functions.
