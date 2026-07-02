# ROLE

Act as a team of experts consisting of:

1. Retinal Ophthalmologist
2. OCTA Imaging Scientist
3. Computer Vision Researcher
4. Deep Learning Architect
5. Medical AI Research Scientist
6. Biostatistician
7. IEEE Journal Reviewer

Your task is to design, critique, and implement a publication-quality multimodal OCTA classification system using the RASTA dataset.

---

# PROJECT GOAL

Build a robust, reproducible, and publication-quality multimodal classification system that classifies retinal OCTA scans into disease cohorts.

The system must:

* Leverage all three OCTA retinal layers independently before fusion
* Integrate clinical metadata with proper handling of missing values
* Produce uncertainty-quantified predictions suitable for clinical screening
* Be explainable at both the spatial (per-layer heatmap) and feature (clinical attribution) level
* Outperform single-layer and single-modality baselines

**Scope:** Version 1 — 2D OCTA images only. Do NOT use angiocubes. Angiocube integration is planned for Version 2 (roadmap in Phase 20).

---

# RESEARCH HYPOTHESIS

Different disease cohorts produce distinct retinal microvascular changes.

These changes appear differently across:

* Superficial Vascular Complex (SVC) — arteriolar and capillary density changes
* Deep Vascular Complex (DVC) — deeper capillary rarefaction, sensitive to diabetic microangiopathy
* Choriocapillaris (CC) — choroidal perfusion and endothelial function

A model that encodes each layer independently, then learns cross-layer interactions, then fuses with clinical context should outperform any single-layer or single-modality classifier.

The biological rationale is that each layer reflects a different physiological compartment. Cross-layer correlation patterns — not individual layer signals — contain additional discriminative information about disease type.

---

# DATASET DESCRIPTION

Assume access to the RASTA dataset.

Per patient (per eye):

```
Sup_OD.bmp    Deep_OD.bmp    CC_OD.bmp
Sup_OS.bmp    Deep_OS.bmp    CC_OS.bmp   (if available)
```

Clinical metadata fields:

* Age (continuous)
* Sex (binary)
* BMI (continuous)
* Smoking (ordinal: never / former / current)
* Hypertension (binary)
* Diabetes (binary)
* Dyslipidemia (binary)
* Additional available fields

Label:

Disease cohort folder (multi-class).

Generated internally; the RASTA dataset is assumed to provide **no manual segmentation masks**:

* Vessel pseudo-masks for SVC and DVC
* FAZ pseudo-masks for SVC and DVC
* Choriocapillaris flow-deficit pseudo-masks
* Quantitative OCTA biomarkers extracted from generated masks

These masks are automated weak labels, not human-annotated ground truth. The pipeline must clearly distinguish classical pseudo-mask generation, optional segmentation-student refinement, and downstream disease classification.

---

# PHASE 1 — DATASET ANALYSIS

Perform comprehensive dataset analysis before any modeling.

Generate:

1. Total sample count (patients and eyes separately)
2. Class distribution per cohort — compute class imbalance ratio
3. Bilateral eye availability (OD only vs OD+OS)
4. Missing values per clinical feature — per cohort breakdown
5. Age, sex, BMI distributions per cohort
6. Image resolution and quality distribution
7. Layer availability audit (are all three layers present for all patients?)
8. Potential confounders (age-disease correlation, sex imbalance per cohort)
9. Data leakage risk assessment — verify no patient appears in multiple cohorts

Also generate:

* Class-wise biomarker box plots from generated pseudo-masks and image-derived metrics
* Inter-cohort demographic comparison table
* Missing data heatmap (patients × clinical features)

Use these findings to inform class-weighting, stratification strategy, and imputation design.

---

# PHASE 2 — PREPROCESSING PIPELINE

## Bilateral Eye Handling

Treat OD and OS as independent samples during training and evaluation.

**Critical rule:** During patient-level splitting, both OD and OS of the same patient MUST be assigned to the same fold. Never allow the same patient's two eyes to appear in different folds. This prevents label leakage via shared patient-level disease state.

Implement a patient-stratified fold assignment that:

* Groups all eyes by patient ID before splitting
* Stratifies folds by cohort label AND patient age decade
* Verifies zero patient overlap across folds after splitting

## OCTA Image Preprocessing

For each image (Sup, Deep, CC):

1. **Quality control:** SNR thresholding + lightweight CNN-based quality classifier. Flag and exclude low-quality scans. Log all exclusions.
2. **Grayscale normalization:** Percentile-based (2nd–98th percentile) per image. Avoids sensitivity to extreme pixel values.
3. **CLAHE:** Apply contrast-limited adaptive histogram equalization. Clip limit=2.0, tile grid=8×8. Apply per-layer, not globally.
4. **Resize:** Bicubic interpolation to 224×224.
5. **Intensity standardization:** Per-channel z-score normalization using training set statistics only (compute on train fold, apply to val/test fold).
6. **Corruption check:** Remove zero-variance, all-black, or unreadable images. Log counts.

## Clinical Metadata Preprocessing

1. **Continuous features** (age, BMI): z-score normalization using training fold mean and std.
2. **Binary features** (sex, hypertension, diabetes, dyslipidemia, smoking): encode as integers, preserve as-is.
3. **Ordinal features** (smoking: 0/1/2): embed as learned ordinal token, not raw integer.
4. **Missing value handling:** Do NOT use mean/median imputation. Instead, use a **learnable missing-value mask token** per feature — a trainable embedding vector that replaces missing values. This allows the model to learn optimal behavior for absent data rather than assuming a statistical proxy.
5. **Feature selection:** Compute mutual information between each clinical feature and the disease label on the training fold. Document features with MI < 0.01 — consider dropping or flagging as low-information.

## Dataset Object

Each sample returns:

```python
{
  "sup_image":        Tensor[1, 224, 224],
  "deep_image":       Tensor[1, 224, 224],
  "cc_image":         Tensor[1, 224, 224],
  "clinical_features": Tensor[D_clin],     # with mask tokens for missing
  "clinical_mask":    Tensor[D_clin],     # 1=observed, 0=missing
  "sup_vessel_mask":  Tensor[1, 224, 224], # generated pseudo-mask
  "deep_vessel_mask": Tensor[1, 224, 224], # generated pseudo-mask
  "sup_faz_mask":     Tensor[1, 224, 224], # generated pseudo-mask
  "deep_faz_mask":    Tensor[1, 224, 224], # generated pseudo-mask
  "cc_flow_mask":     Tensor[1, 224, 224], # generated pseudo-mask
  "biomarkers":       Tensor[D_bio],       # generated mask/image biomarkers
  "mask_source":      str,                 # "classical" or "student"
  "label":            int,
  "patient_id":       str,
  "eye":              str,                # "OD" or "OS"
  "fold":             int
}
```

---

# PHASE 3 — OCTA-SIMCLR PRETRAINING

**Why pretraining is essential:**

The RASTA dataset is medium-sized. ConvNeXt-Tiny initialized from ImageNet has been trained on natural RGB images — a large domain gap from grayscale vessel-contrast OCTA. Self-supervised pretraining on unlabeled OCTA (or all available RASTA images before labels are used) adapts the visual representations to the retinal vascular domain before fine-tuning.

## Positive Pair Strategy: Bilateral Natural Positives

For patients with both OD and OS:

**Positive pair:** (Sup_OD, Sup_OS) from the same patient.

The two eyes of the same patient share the same systemic disease state. Their OCTA images are the same modality and disease context, just mirrored. This is a strong, medically meaningful positive pair signal without any manual annotation.

For patients with only one eye:

**Positive pair:** Two augmented views of the same image (standard SimCLR).

## Architecture

* Backbone: ConvNeXt-Tiny (initialized from ImageNet)
* Projection head: 2-layer MLP (768 → 256 → 128), appended during pretraining only, discarded afterward
* Loss: NT-Xent (normalized temperature-scaled cross-entropy) with temperature τ=0.07

Three independent contrastive tasks: one per OCTA layer (Sup, Deep, CC).

The encoders do NOT share weights — separate SimCLR pretraining per layer.

## Augmentation during pretraining

Use conservative augmentations that preserve vascular structure:

* Random horizontal flip (p=0.5)
* Random rotation ±15° (p=0.4)
* Random crop (scale=0.7–1.0) + resize to 224×224
* Gaussian blur (σ=0.1–2.0, p=0.3)
* Intensity jitter (brightness ±0.1, contrast ±0.1) — applied mildly

Do NOT use: color inversion, aggressive distortion, cutout, or random erasing. These destroy the vessel density signal.

## Pretraining schedule

* Epochs: 100
* Optimizer: SGD with momentum=0.9, weight_decay=1e-4
* LR: 0.1 with cosine decay
* Batch size: 256 (with gradient accumulation if memory limited)

After pretraining, retain only backbone weights. Discard projection heads.

---

# PHASE 4 — IMAGE ENCODERS WITH PROJECTION HEADS

## Backbone

Architecture: **ConvNeXt-Tiny**

* Initialized from ImageNet pretrained weights
* Further adapted via OCTA-SimCLR (Phase 3)
* Output dimension from final stage: 768

Three independent encoders: one each for Sup, Deep, CC.

Do NOT share backbone weights between layers.

**Biological justification:** Each retinal layer has distinct vascular architecture. SVC shows arterioles and capillaries in a 2D mesh. DVC shows sparser, finer capillaries in a different depth plane. CC shows a mosaic perfusion pattern. These textures are fundamentally different — a shared backbone would impose the same filter bank on all three, limiting layer-specific feature learning.

## Projection Heads (post-encoder, pre-attention)

After global average pooling (768-d), apply a per-encoder projection head:

```
FC(768 → 512) → GELU → LayerNorm
FC(512 → 256) → LayerNorm
```

Output: 256-d token per layer.

**Why project before attention?**

768-d tokens in multi-head attention are computationally expensive and prone to overfitting with small N. Projecting to 256-d before the attention module reduces parameters, improves training stability, and is standard practice in vision-language transformer fusion.

## Spatial Feature Map Retention

Before global average pooling, extract the final spatial feature map (7×7×768 after ConvNeXt-Tiny's last stage) and retain a reference for Grad-CAM computation. Do not pool prematurely.

## Output per encoder

* Fs: Sup features ∈ ℝ^256
* Fd: Deep features ∈ ℝ^256
* Fc: CC features ∈ ℝ^256

---

# PHASE 5 — TABULAR METADATA ENCODER

## Input handling

Concatenate all processed clinical features into a single vector of dimension D_clin. After Phase 8, append generated OCTA biomarkers to form D_tab = D_clin + D_bio.

Missing clinical features are replaced by their learned mask token (trainable 1-d embedding per feature, not a fixed zero or mean). Biomarker missingness can occur if mask QC fails; represent failed biomarkers with a biomarker-missing mask token and explicit QC flag.

## Architecture

```
Input: [D_tab]
LayerNorm(input)
FC(D_tab → 128) → GELU → Dropout(0.3)
FC(128 → 256) → GELU → LayerNorm → Dropout(0.2)
```

Output: F_tabular ∈ ℝ^256

## Why GELU over ReLU for clinical data

Clinical feature distributions are often non-Gaussian and contain negative z-scored values. GELU's smooth, non-zero gradient near zero allows better gradient flow for near-zero features (e.g., slightly below-average BMI). ReLU hard-zeros these, losing gradient information.

## Missing value training strategy

During training, randomly mask 10% of observed clinical features as if missing (set to mask token). This simulates real-world missingness and makes the encoder robust to absent clinical data at inference time.

---

# PHASE 6 — CROSS-LAYER ATTENTION MODULE

**Goal:** Model interactions between SVC, DVC, and CC feature representations before fusing with clinical data.

## Why cross-layer interactions matter

SVC pathology (e.g., arteriolar narrowing) predicts DVC rarefaction in hypertension. CC flow deficits correlate with SVC perfusion changes in diabetic retinopathy. These cross-layer correlation patterns differ across disease cohorts and constitute discriminative information not captured by any single encoder.

## Architecture

**Step 1: Layer-identity embedding**

Before attention, add a learnable layer-identity positional embedding to each token:

```
T_sup  = Fs + e_sup
T_deep = Fd + e_deep
T_cc   = Fc + e_cc
```

Where e_sup, e_deep, e_cc ∈ ℝ^256 are learnable layer-identity vectors, initialized independently.

**Why this matters:** Without layer-identity embeddings, multi-head self-attention treats the three tokens as an unordered set. The model cannot learn that "token 2 is always the DVC" because the tokens have no identity marker — any permutation produces identical attention scores. Layer-identity embeddings fix this.

**Step 2: Multi-head self-attention**

Stack the three tokens as a sequence: S = [T_sup ; T_deep ; T_cc] ∈ ℝ^(3×256)

Apply transformer encoder block:

* 8 attention heads, head dimension = 32
* Pre-norm (LayerNorm before attention)
* Feed-forward: 2-layer MLP (256 → 1024 → 256) with GELU
* Dropout = 0.1

**Step 3: Aggregation**

Apply mean pooling over the three attended tokens:

F_octa = MeanPool([H_sup; H_deep; H_cc]) ∈ ℝ^256

**Output:** F_octa — Layer-Aware OCTA Representation

---

# PHASE 7 — MULTIMODAL CROSS-ATTENTION FUSION

**Goal:** Fuse imaging representation (F_octa) with tabular context (F_tabular: clinical features + generated biomarkers) and generated segmentation-mask features (F_mask).

**Why cross-attention outperforms concatenation:**

Concatenation treats all modalities as equally relevant for every patient. Cross-attention allows F_octa to attend to clinical features selectively — a diabetic patient's DVC rarefaction is contextualised differently than the same finding in a normotensive non-diabetic. This mirrors clinical reasoning.

## Architecture

**Tabular + biomarker encoding:**

Generated OCTA biomarkers are appended to the tabular stream after fold-fitted normalization:

```
F_tabular = TabularEncoder(Concat([clinical_features, biomarkers])) ∈ ℝ^256
```

**Cross-attention block:**

```
Q = Linear(F_octa)          ∈ ℝ^(1 × 256)
K = Linear(F_tabular)       ∈ ℝ^(1 × 256)
V = Linear(F_tabular)       ∈ ℝ^(1 × 256)

Attended = softmax(QK^T / √256) · V    (4-head cross-attention)
F_fused = F_octa + Attended             (residual connection)
F_fused = LayerNorm(F_fused)
```

**Generated mask branch fusion:**

Generated masks are produced upstream by Phase 8. The classifier never assumes human-provided masks.

```
M = Stack([sup_vessel, deep_vessel, sup_faz, deep_faz, cc_flow])
F_mask = MaskEncoder(M)  ∈ ℝ^128
F_fused = Concat([F_fused, F_mask]) → project to 512-d
```

**Projection to shared space:**

```
H = FC(Concat([F_fused ; F_tabular]) → 512) + GELU + LayerNorm
```

Output: H ∈ ℝ^512 — Unified multimodal representation.

---

# PHASE 8 — SELF-GENERATED SEGMENTATION MASKS AND MASK ENCODER

The dataset is assumed to contain **no manual vessel, FAZ, or flow-deficit masks**. Therefore, segmentation in Version 1 is an automated weak-segmentation pipeline:

```text
Preprocessed OCTA image
  → classical pseudo-mask generation
  → optional segmentation-student refinement
  → mask QC + biomarker extraction
  → downstream classifier mask branch
```

Do not describe these as human ground-truth segmentations unless a manually annotated validation subset is later created.

## Step A — Classical pseudo-mask generation

Generate deterministic pseudo-labels for every fold using training-independent image-processing rules. These masks are reproducible weak labels, not learned from disease labels.

### SVC and DVC vessel masks

For Sup and Deep layers:

```text
percentile normalization
→ CLAHE
→ denoising / mild Gaussian smoothing
→ Frangi or Sato vesselness filtering
→ adaptive/local thresholding
→ morphological opening/closing
→ small-component removal
→ optional skeletonization for biomarkers
```

Outputs:

```text
sup_vessel_mask, deep_vessel_mask
```

### SVC and DVC FAZ masks

FAZ is detected as the central avascular low-flow region.

```text
vessel mask / low-flow map
→ central search window around image center
→ connected-component selection nearest center
→ contour smoothing / hole filling
→ plausibility QC by area, eccentricity, and centroid distance
```

Outputs:

```text
sup_faz_mask, deep_faz_mask
```

Skip FAZ generation for CC unless dataset-specific review shows CC FAZ-like annotations are meaningful.

### CC flow-deficit masks

For choriocapillaris:

```text
percentile normalization
→ CLAHE or local background correction
→ local thresholding (Phansalkar/Niblack/Sauvola-style)
→ dark-region detection
→ morphology cleanup
→ area filtering
```

Output:

```text
cc_flow_mask
```

## Step B — Classical-mask improvement gate before any student model

Do **not** train U-Net, Attention U-Net, or U-Net++ directly on poor classical pseudo-labels. A segmentation student can only imitate and regularize its training targets; it cannot recover missing vessels without human ground truth.

Before student refinement is allowed, classical masks must pass a documented review gate:

1. Generate Phase 8 masks and montage sheets.
2. Visually inspect random samples and all QC failures.
3. Confirm Sup/Deep vessel masks capture major vessels **and** capillary texture, not only sparse edge/large-vessel fragments.
4. Confirm FAZ masks are centered and plausible.
5. Confirm CC flow-deficit masks are not dominated by watermark/edge artifacts.
6. Tune classical vessel extraction if masks are too sparse:
   * lower vesselness or adaptive-threshold cutoff,
   * reduce small-object removal aggressiveness,
   * combine Frangi/Sato vesselness with adaptive intensity thresholding,
   * remove watermark/edge artifacts before QC/biomarkers.

Only after this gate is passed may a student be trained.

## Step C — Explicit student-refinement decision gate

A future agent must **not silently skip** student refinement just because it is described as optional. After improved classical masks are generated, make and document one of these explicit decisions in `PROGRESS.md`:

```text
Decision A — Use improved classical masks only
Allowed only if QC + montage review show masks are anatomically acceptable and stable enough.
Required note: why student refinement is unnecessary.

Decision B — Train U-Net++ student
Required if improved classical masks are anatomically plausible but still noisy, fragmented, threshold-unstable, or biomarker-unstable.
Required outputs: student checkpoints, student-vs-classical Dice/IoU, montage comparison, biomarker stability comparison.

Decision C — Do not train student yet; tune classical masks again
Required if improved classical masks are still anatomically poor/sparse/wrong. U-Net++ must not be trained on bad pseudo-labels.
```

Default expectation after the improvement pass: **evaluate U-Net++ student refinement unless Decision A is explicitly justified**. In other words, U-Net++ is optional only after a documented QC-based decision, not something to ignore.

## Step D — Segmentation student model

If Decision B is selected, train a segmentation student to imitate pseudo-labels and produce smoother standardized masks. This model fits **upstream of the disease classifier**, between preprocessing and classification.

Student architecture:

```text
Primary student to evaluate: U-Net++
Fallback/lighter variant if compute-limited: Attention U-Net
Encoder: EfficientNet-B0 or ConvNeXt-Nano, 1-channel OCTA input
Decoder: U-Net++ decoder, or Attention U-Net decoder for fallback
Output heads:
  Sup/Deep model: 2 channels = vessel + FAZ
  CC model: 1 channel = flow-deficit
Activation: sigmoid per channel
```

Training target: QC-passed/improved classical pseudo-masks from Step A/B, not raw failed masks.

Student loss:

```
L_student = α · L_Tversky(β=0.7) + (1-α) · L_Focal(γ=2) + λ_dice · L_Dice
```

Use confidence weighting where possible: downweight pseudo-label regions that fail QC or lie near unstable boundaries.

During downstream disease-classifier training, freeze the segmentation student by default:

```text
OCTA image → frozen student segmenter → generated masks → mask encoder + biomarker extractor
```

End-to-end fine-tuning of the student is not allowed in Version 1 unless a stability loss preserves mask quality; otherwise classifier gradients may distort segmentation.

## Mask QC

Reject or flag generated masks with implausible properties:

* Vessel density outside plausible range
* FAZ centroid too far from image center
* FAZ area too small/large
* CC flow-deficit mask nearly all black or all white
* Excessive tiny connected components

Generate visual montage sheets for random samples and all QC failures. A montage is a review contact sheet showing the original Sup/Deep/CC OCTA images beside the generated Sup vessel, Deep vessel, Sup FAZ, Deep FAZ, and CC flow masks for the same eye. Manually inspect a subset of generated masks and report accept/reject rate if used in a paper.

## Biomarker extraction

Extract quantitative features from generated masks:

* Vessel density
* Vessel skeleton density / vessel length density
* FAZ area
* FAZ perimeter
* FAZ circularity
* FAZ centroid distance from image center
* CC flow-deficit percentage
* CC flow-deficit count
* Mean / median flow-deficit area

These biomarkers are normalized using training-fold statistics and appended to the clinical/tabular encoder.

## Mask encoder for classifier

Input:

```text
[sup_vessel_mask, deep_vessel_mask, sup_faz_mask, deep_faz_mask, cc_flow_mask] ∈ ℝ^(5×224×224)
```

Architecture:

```
ConvNeXt-Nano (or 3-layer lightweight CNN)
GlobalAvgPool → 128-d
Projection: FC(128 → 128) → GELU → LayerNorm
```

Output: F_mask ∈ ℝ^128

Version 1 first rerun: generate **improved classical masks** for all samples. Do not use the current sparse vessel masks as final if visual review shows they miss the capillary network. After improved classical masks, a future agent must explicitly choose Decision A/B/C above. If Decision B is chosen, evaluate **U-Net++ as the primary student** and Attention U-Net only as a lighter fallback. The reported full model must state `mask_source` explicitly (`classical_improved`, `unetpp_student`, or `attention_unet_student`). Compare performance with and without generated masks, generated biomarkers, and student refinement in ablation (Phase 15).

---

# PHASE 9 — CLASSIFICATION HEAD WITH PROTOTYPE AUGMENTATION

## Standard head

```
FC(512 → 256) → GELU → Dropout(0.3)
FC(256 → 128) → GELU → Dropout(0.3)
FC(128 → N_classes)
```

## Prototype-augmented logits

For imbalanced disease cohorts, augment the standard linear classifier with prototype-based logits:

During training, maintain a running class prototype P_k ∈ ℝ^128 (exponential moving average of class-conditional mean embeddings in the penultimate layer).

At inference, compute prototype similarity:

```
proto_logit_k = -‖z - P_k‖² / τ_proto
```

Final logit = standard_logit + α · proto_logit

where α is a learned scalar weight and τ_proto = 0.1.

This provides a non-parametric classification signal that generalises better for rare cohorts with few samples, complementing the standard linear head.

## Uncertainty quantification

Apply Monte Carlo Dropout at inference (30 forward passes with Dropout active).

Report per-sample:
* Mean prediction: P(class | x) = (1/30) Σ softmax(f_t(x))
* Epistemic uncertainty: σ²_k = Var_t[softmax(f_t(x))_k] per class
* Entropy: H = -Σ P_k log P_k (overall prediction uncertainty)

Flag predictions with entropy > threshold as "uncertain — recommend manual review."

## Post-training calibration

Apply temperature scaling on the validation fold:

Learn scalar T such that softmax(logits / T) is well-calibrated.

Minimise Expected Calibration Error (ECE) over the validation fold.

---

# PHASE 10 — LOSS FUNCTIONS

## Primary classification loss

**Default:** Weighted Cross-Entropy with label smoothing ε=0.1

Class weights: inverse class frequency (computed per training fold).

```
L_cls = -Σ_k w_k · y_k · log(p_k)   (weighted CE with smoothed targets)
```

**If severe imbalance (max_class_freq / min_class_freq > 10):** Switch to Focal Loss:

```
L_focal = -Σ_k (1 - p_k)^γ · log(p_k)    γ=2
```

Compare both in experiments. Report which performs better per dataset split.

**Class-balanced sampling:** Use WeightedRandomSampler to oversample minority cohorts in each training batch. Combine with loss weighting (both simultaneously).

## Segmentation student loss

Segmentation is trained as a separate upstream stage against generated pseudo-labels, not against human masks.

```
L_student = α · L_Tversky(β=0.7) + (1-α) · L_Focal(γ=2) + λ_dice · L_Dice
```

Tversky with β=0.7 penalises false negatives more heavily — critical for sparse capillary, FAZ, and flow-deficit masks.

Do not include `L_student` in the main classifier objective by default. The segmentation student is frozen during disease-classifier training.

## Calibration loss

```
L_cal = ECE(p, y)    (computed over mini-batches as a soft differentiable estimate)
```

## Total disease-classifier loss

```
L_total = L_cls + λ₂ · L_cal
```

Optional experimental variant only:

```
L_total_joint = L_cls + λ₁ · L_student + λ₂ · L_cal
```

Use joint training only if mask-quality monitoring confirms that classifier gradients do not degrade segmentation masks.

Initial value: λ₂=0.1 for calibration. For the optional joint variant only, start with λ₁=0.1–0.5 and tune via validation while monitoring mask QC.

Label smoothing: applies to L_cls only.

---

# PHASE 11 — DATA AUGMENTATION

Apply augmentations to OCTA images during training only. Never augment validation or test images.

## OCTA-specific augmentation pipeline

```python
transforms.Compose([
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.3),
    transforms.RandomRotation(degrees=15),
    transforms.RandomAffine(
        degrees=0,
        translate=(0.05, 0.05),
        scale=(0.95, 1.05),
        shear=5
    ),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    AddGaussianNoise(sigma=0.01, p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=fold_mean, std=fold_std)
])
```

Do NOT apply:

* Aggressive color jitter (destroys vessel contrast)
* Random erasing or cutout (removes vascular structure)
* Large-scale elastic deformation (distorts FAZ morphology)
* Random grayscale (images are already grayscale)

**For clinical features:** Add Gaussian noise (σ=0.05) to continuous features during training with probability 0.3. Randomly zero out one observed clinical feature per sample with probability 0.1 (simulates missing data at training time).

**Apply same augmentation to all three OCTA layers simultaneously** (same random seed per sample) to preserve cross-layer spatial correspondence.

---

# PHASE 12 — TRAINING STRATEGY

## 3-Stage Training Schedule

**Stage 1: OCTA-SimCLR pretraining** (unlabeled)

* Epochs: 100
* LR: 0.1, cosine decay
* All three encoders independently

**Stage 2: Frozen backbone, train heads only**

* Epochs: 15
* LR: 3e-4
* Train projection heads, cross-layer attention, fusion, classification head
* Backbone weights frozen (loaded from Stage 1)
* Purpose: initialise downstream heads before catastrophic forgetting risk

**Stage 3: End-to-end fine-tuning**

* Epochs: 100
* Backbone LR: 3e-5 (10× smaller than head LR)
* Head LR: 3e-4
* Linear warmup first 5 epochs
* Cosine annealing (T_0=20, T_mult=2)
* Early stopping: patience=15 on validation macro-F1

## Optimizer

AdamW
* β₁=0.9, β₂=0.999
* Weight decay: 1e-2

## Batch size

32 (or 16 with gradient accumulation ×2 = effective 32 if GPU limited)

Mixed precision: enabled (bfloat16 or float16 with GradScaler)

## Cross-validation

5-fold stratified cross-validation
* Stratify on: disease cohort label AND patient age decade AND sex
* All eyes from one patient always in same fold (patient-level split)
* Report: mean ± std across 5 folds for all metrics
* Final model: ensemble of 5 fold models (average softmax outputs)

## Reproducibility

Set all random seeds (Python, NumPy, PyTorch, CUDA).
Log all hyperparameters, dataset splits, and random states.
Use deterministic PyTorch operations where available.

---

# PHASE 13 — EXPLAINABILITY

## Grad-CAM per layer

For each encoder (Sup, Deep, CC), compute Grad-CAM with respect to the final convolutional feature map before global average pooling.

Generate per-patient, per-class heatmaps:

* Sup heatmap: which SVC regions drove the classification decision
* Deep heatmap: which DVC regions drove the classification decision
* CC heatmap: which CC regions drove the classification decision

Overlay on original OCTA en-face images. Use clinically appropriate colormap (viridis or hot).

Present side-by-side per-layer heatmaps for example patients in each cohort.

## Attention rollout — layer dominance analysis

From the cross-layer attention module (Phase 6), extract the attention weight matrix A ∈ ℝ^(3×3) per sample.

Compute per-sample layer-dominance score:

```
dominance_k = (1/N) Σ_i A_{ik}    for k ∈ {Sup, Deep, CC}
```

Aggregate per cohort: which retinal layer dominates attention for each disease class?

This produces a clinically interpretable finding: "Disease X is primarily driven by CC perfusion abnormalities; Disease Y is driven by SVC arteriolar changes."

## SHAP values for clinical features

Apply KernelSHAP to the clinical MLP branch.

Generate per-patient:

* Waterfall plots showing clinical feature contribution to prediction
* Force plots per patient
* Population beeswarm plots per cohort

Identify which clinical features are most discriminative per disease class.

## Integrated analysis

Correlate Grad-CAM hotspot regions with layer-dominance scores — do the spatial findings agree with the attention-level finding about which layer dominates?

---

# PHASE 14 — EVALUATION METRICS

All metrics reported with 95% bootstrap confidence intervals (n=1000 bootstrap samples, stratified).

## Primary metrics (classification)

* Macro-averaged ROC-AUC (one-vs-rest)
* Macro-F1
* Balanced accuracy (accounts for class imbalance)
* Per-class sensitivity and specificity

## Secondary metrics

* Weighted F1
* PR-AUC (Macro) — particularly important when classes are imbalanced
* Per-class precision, recall, F1
* Confusion matrix (normalised by true class count)
* Top-2 accuracy (whether correct class is in top-2 predictions)

## Calibration

* Expected Calibration Error (ECE, 10 bins) — primary calibration metric
* Brier score
* Reliability diagram per class

## Uncertainty evaluation

* Mean prediction entropy vs. classification accuracy (are uncertain predictions also less accurate?)
* Coverage analysis: percentage of correct predictions in the top-30% confidence bin

## Statistical comparison

For each baseline model vs. proposed model:

* McNemar's test on binary correct/incorrect vectors across all test samples
* DeLong test for pairwise AUC comparison
* Bonferroni correction for multiple comparisons
* Report p-values and effect sizes (Cohen's h for AUC comparisons)

---

# PHASE 15 — ABLATION STUDIES

Retrain from scratch with identical seeds for each ablation. Report mean ± std over 5 folds.

| Configuration | Modification |
|---|---|
| Full model | Proposed architecture |
| −SimCLR pretraining | ImageNet init only, no OCTA pretraining |
| −Cross-layer attention | Concatenate Fs, Fd, Fc directly → project to 256-d |
| −Layer-identity embeddings | Remove e_sup, e_deep, e_cc from cross-layer attention |
| −Tabular branch | Remove F_tabular; classify on F_octa + F_mask only |
| −Mask branch | Remove F_mask from generated masks |
| −Mask biomarkers | Remove generated OCTA biomarker vector from tabular encoder |
| Classical masks only | Use raw image-processing pseudo-masks; bypass segmentation student |
| Student masks only | Use frozen Attention U-Net / U-Net++ outputs for masks and biomarkers, allowed only after improved classical pseudo-labels pass QC |
| −Prototype head | Standard linear head only |
| −Uncertainty (MC-Dropout) | Single deterministic forward pass |
| −Temperature calibration | Uncalibrated softmax probabilities |
| −Label smoothing | Hard one-hot targets |
| −Class-balanced sampling | Uniform sampling only |
| Sup only | Single encoder; remove Deep and CC |
| Deep only | Single encoder; remove Sup and CC |
| CC only | Single encoder; remove Sup and Deep |
| Sup + Deep | Two encoders; remove CC |
| Sup + Deep + CC | Three encoders; no tabular or mask branch |
| Images + Tabular | Sup + Deep + CC + clinical + generated biomarkers; no mask encoder |
| Images + Tabular + Masks | Full model confirmation |

Analyze and discuss the marginal contribution of each component.

---

# PHASE 16 — BASELINE COMPARISON

Implement and compare against the following baselines. Use identical preprocessing, augmentation, and cross-validation splits.

| Baseline | Type | Input |
|---|---|---|
| Logistic Regression | Classical | Clinical features only |
| Random Forest | Ensemble | Clinical features only |
| XGBoost | Gradient boosting | Clinical features only |
| ResNet50 | CNN | Sup image only |
| ResNet50 (3-channel) | CNN | Sup+Deep+CC stacked as RGB |
| EfficientNet-B0 | CNN | Sup image only |
| Vision Transformer (ViT-S/16) | Transformer | Sup image only |
| ConvNeXt-Tiny (single) | CNN | Sup image only |
| Proposed — full | Multimodal | Sup + Deep + CC + Clinical + generated masks + biomarkers |

For CNN baselines: use ImageNet pretrained weights with the same training schedule as the proposed model (no SimCLR pretraining for baselines — this tests the value of pretraining specifically).

---

# PHASE 17 — STATISTICAL VALIDATION

## Group-level analysis

For each biomarker or clinical feature:

* Mann-Whitney U test: cohort A vs. cohort B (pairwise, all pairs)
* One-way ANOVA: across all cohorts simultaneously
* Kruskal-Wallis test: non-parametric alternative when normality is violated

## Fairness analysis

Evaluate model performance stratified by:

* Sex (male vs. female)
* Age quartile (<50, 50–60, 60–70, >70)
* Presence of diabetes
* Bilateral vs. unilateral eye availability

Report whether model performance degrades for underrepresented subgroups.

## Multivariate analysis

Logistic regression (one-vs-rest) using clinical features only: use as a clinical baseline and to quantify the added value of imaging features.

Partial correlation: predicted disease probability vs. each clinical feature, controlling for age and sex.

---

# PHASE 18 — NOVEL RESEARCH CONTRIBUTIONS

Identify and articulate the following contributions for publication:

1. **Bilateral natural positives for OCTA-SimCLR:** First contrastive pretraining strategy that exploits OD/OS pairs as positive samples, leveraging the unique bilateral structure of ophthalmic datasets to avoid label requirements.

2. **Layer-identity tokens in cross-layer attention:** Explicit layer-identity embeddings ensure the transformer-based cross-layer attention module can distinguish which token corresponds to which retinal depth — a critical but commonly omitted detail.

3. **Learnable mask-token imputation for clinical data:** Replaces statistical imputation (mean/mode) with a learned representation of "this feature was not measured," preserving the model's ability to reason about data absence as a distinct clinical state.

4. **Prototype-augmented classification for rare cohorts:** Hybrid parametric + non-parametric classification head that improves generalisation for minority disease classes without requiring oversampling or additional annotations.

5. **Layer-dominance analysis as a clinical explainability tool:** Systematic attribution of predictions to specific retinal layers via attention weight analysis, producing cohort-level findings interpretable by ophthalmologists and cardiologists.

6. **Uncertainty-aware disease classification:** Integration of MC-Dropout uncertainty quantification and ECE-calibrated probabilities, enabling clinical triage — uncertain predictions can be flagged for expert review rather than acted upon directly.

7. **Automated weak OCTA segmentation for mask-guided classification:** Generates vessel, FAZ, and CC flow-deficit pseudo-masks without manual annotations, optionally refines them with a frozen segmentation student, and converts them into interpretable biomarkers for disease classification.

---

# PHASE 19 — DELIVERABLES

Provide all of the following:

1. **Dataset loader** — complete PyTorch Dataset class with patient-level stratified splitting, bilateral eye handling, mask-token imputation, and data integrity checks
2. **Training pipeline** — staged schedule (SimCLR → pseudo-mask generation/student training → frozen fine-tune → end-to-end classifier), logging, checkpointing
3. **Model implementation** — all modules: encoders, projection heads, cross-layer attention, fusion, classification head, segmentation student, mask encoder
4. **Validation pipeline** — 5-fold CV loop with metric accumulation, bootstrap CI computation, calibration step
5. **Explainability module** — Grad-CAM (per layer), attention rollout (layer dominance), SHAP (clinical features), generated-mask visual overlays
6. **Experiment manager** — configuration-driven, YAML-based, all hyperparameters logged via wandb or MLflow
7. **Reproducibility config** — fixed seeds, deterministic ops, dataset split serialisation
8. **Full PyTorch codebase** — modular, documented, with unit tests for critical components
9. **Results tables** — all metrics, all ablations, all baselines, with bootstrap CIs
10. **Publication-ready methodology section** — IEEE TPAMI / Nature Biomedical Engineering style
11. **Segmentation artifacts** — pseudo-mask generator, optional segmentation-student checkpoints, mask QC reports, visual montage sheets, and biomarker CSVs

---

# PHASE 20 — V2 ROADMAP (ANGIOCUBE INTEGRATION)

Version 2 will extend this system with 3D angiocube features.

Planned additions:

* Replace 2D ConvNeXt-Tiny with shared 3D Swin-T backbone + layer-specific adapter tokens (as designed in the CVD architecture)
* Add Retinal-MAE pretraining with layer-masked reconstruction
* Integrate volumetric biomarker extraction (VAD, VLD, VFD per ETDRS zone)
* Full multi-task learning: classification + biomarker regression + segmentation supervision
* PCGrad dynamic loss weighting

Version 2 target: replace the current cross-layer attention on 3 pooled vectors with full cross-layer attention on 3D spatial feature sequences.

This Version 1 system is explicitly designed to be compatible with V2 extension — the modular architecture allows drop-in replacement of 2D encoders with 3D equivalents.
