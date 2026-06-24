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

## 2026-06-24

### Phase 4 progress
- Added **Phase 4 — Image Encoders with Projection Heads** cells to `main.ipynb`.
- Implemented independent Sup/Deep/CC ConvNeXt-Tiny image encoders with no shared backbone parameters.
- Implemented per-layer projection heads: `Linear(768,512) -> GELU -> LayerNorm -> Linear(512,256) -> LayerNorm`.
- Added Phase 3 backbone loading from expected `*_backbone.pt` files, with layer metadata validation.
- Retained final spatial feature maps before pooling for later Grad-CAM.
- Added `Phase4OCTAImageEncoders` forward output:
  - `layer_tokens`: `[B, 3, 256]`
  - per-layer 256-d tokens
  - pooled 768-d features
  - per-layer final spatial feature maps.
- Added backbone availability audit writing `outputs/phase4_image_encoders/phase4_backbone_availability.csv`.
- Kept `PHASE4_REQUIRE_SIMCLR_BACKBONES = False` and `PHASE4_RUN_SMOKE_TEST = False` so the notebook does not fail before Phase 3 outputs exist.
