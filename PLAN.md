# ROLE

Act as a team of experts consisting of:

1. Retinal Ophthalmologist
2. Cardiologist
3. OCTA Imaging Scientist
4. Computer Vision Researcher
5. Deep Learning Architect
6. Medical AI Research Scientist
7. Biostatistician
8. IEEE Journal Reviewer

Your task is to design, critique, and implement a publication-quality research project for cardiovascular risk prediction using retinal OCTA data from the RASTA dataset.

---

# PROJECT GOAL

Develop a state-of-the-art multimodal deep learning framework that predicts cardiovascular disease risk from retinal OCTA data while maintaining clinical interpretability through retinal vascular biomarkers.

The model must leverage:

* OCTA Angiocubes (3D)
* En-face OCTA Images
* Layer-specific vascular information
* Clinical metadata
* Retinal vascular biomarkers

The final system must outperform traditional biomarker-only methods, provide uncertainty-quantified predictions, and deliver clinically explainable outputs suitable for ophthalmic screening workflows.

---

# DATASET

Assume access to the RASTA dataset containing:

* 3D OCTA Angiocubes
* En-face OCTA images
* Superficial Vascular Complex (SVC)
* Deep Vascular Complex (DVC)
* Choriocapillaris (CC)
* Clinical cardiovascular information
* Cardiovascular risk labels
* FAZ metrics
* Vessel density metrics
* Perfusion density metrics

---

# RESEARCH HYPOTHESIS

Cardiovascular disease produces systemic microvascular dysfunction.

Retinal microvasculature reflects systemic vascular health.

Changes in:

* Vessel density
* Vessel length
* Fractal complexity
* Flow deficits
* FAZ morphology

can be detected in OCTA and used to predict cardiovascular risk.

Furthermore, SVC, DVC, and CC reflect different depths and types of microvascular pathology. Their interactions — not just their individual signals — contain predictive information not captured by any single layer alone.

---

# PHASE 1 — DATA ANALYSIS

Perform comprehensive dataset analysis.

Generate:

1. Dataset statistics
2. Patient demographics
3. Class distribution
4. Missing values
5. Label distribution
6. Image dimensions
7. Layer availability
8. Biomarker availability
9. Potential biases
10. Data leakage risks

Provide tables and visualizations.

Also compute:

* Inter-class biomarker distributions (CVD vs non-CVD)
* Spearman correlation matrix across all biomarkers
* Mutual information between each biomarker and the CVD label

These will inform biomarker selection in Phase 4.

---

# PHASE 2 — OCTA PREPROCESSING

Design the complete OCTA preprocessing pipeline.

Include:

## Quality Control

Detect and remove:

* Motion artifacts
* Blink artifacts
* Projection artifacts
* Poor quality scans
* Low signal regions

Use signal-to-noise ratio thresholding and a lightweight CNN-based quality scorer trained on labeled artifact examples.

## Normalization

Apply:

* Intensity normalization (percentile-based, not global min-max)
* Contrast-limited adaptive histogram equalization (CLAHE) per layer
* Layer-wise zero-mean unit-variance normalization

## Registration

Align:

* OCTA layers (SVC, DVC, CC) to a common reference frame using rigid registration
* En-face images to the angiocube projection
* Biomarker maps to registered layer space

Explain mathematically and algorithmically.

## Layer Separation

Separate:

* SVC — inner plexiform to inner nuclear layer
* DVC — outer plexiform to outer nuclear layer
* CC — sub-RPE slab

Maintain independent layer volumes for separate encoding streams.

Explain why each layer contributes differently to cardiovascular prediction:

* SVC reflects retinal arteriole and capillary density — sensitive to hypertension and early arteriosclerosis
* DVC reflects deeper capillary beds — more sensitive to diabetic microvascular changes
* CC reflects choroidal perfusion — reflects systemic hemodynamic status and endothelial dysfunction

---

# PHASE 3 — VESSEL SEGMENTATION

Build a vessel segmentation subsystem.

Architecture:

**Attention U-Net++**

Input per branch:
* SVC volume
* DVC volume
* CC volume

Outputs:
* Vessel masks
* Capillary masks
* FAZ masks
* Flow deficit maps

Design:

* Dense skip connections (U-Net++ style)
* Squeeze-and-excitation attention blocks at each skip
* Deep supervision at every decoder scale

Loss function:

Combined Tversky + Focal loss:

L_seg = α · L_Tversky(β=0.7, γ=0.3) + (1−α) · L_Focal(γ=2)

Set α=0.5. Tversky with β>0.5 penalizes false negatives more heavily — critical for sparse capillary detection.

The segmentation head is also used as an auxiliary supervision signal in multi-task training (Phase 11).

---

# PHASE 4 — BIOMARKER EXTRACTION

Extract interpretable retinal biomarkers.

## Vessel Biomarkers

* VAD — Vessel Area Density
* VLD — Vessel Length Density
* VFD — Vessel Flux Density
* VT — Vessel Tortuosity
* VBN — Vessel Branch Number

## FAZ Biomarkers

* FA — FAZ Area
* FR — FAZ Regularity Index
* Circularity = 4π·Area / Perimeter²
* FAZ Perimeter

## Perfusion Biomarkers

* Perfusion Density (per ETDRS region)

## Choriocapillaris Biomarkers

* FD% — Flow Deficit Percentage

## Regional Decomposition

Compute each biomarker at:

* Global level
* ETDRS 9-zone grid (1mm, 3mm, 6mm rings)
* 4-quadrant decomposition (superior, inferior, nasal, temporal)
* 8×8 patch-level maps (for spatial biomarker maps fed to the biomarker branch)

Provide complete mathematical definitions for every metric.
Explain biomedical significance for each.

## Biomarker Selection Step

Before encoding:

1. Compute Spearman correlation matrix across all extracted biomarkers
2. Compute mutual information between each biomarker and CVD label
3. Remove biomarkers with pairwise Spearman |ρ| > 0.85 (keep the one with higher MI)
4. Retain the top 7–9 most independently informative biomarkers

This prevents redundant gradients in the biomarker MLP head and improves convergence on a medium-sized dataset.

---

# PHASE 5 — RETINAL-MAE PRETRAINING (Layer-Masked)

Because the dataset is limited, design a self-supervised pretraining stage before fine-tuning.

## Architecture: Retinal-MAE

Use a **Masked Autoencoder** operating on 3D OCTA Angiocubes with **layer-level masking** instead of random patch masking.

**Why layer-level masking is superior to random patch masking:**

Standard MAE masks random patches — the encoder can reconstruct these by interpolating spatially nearby patches within the same layer. This does not force learning of cross-layer dependencies, which is the biological signal of interest.

Layer-level masking forces the model to reconstruct an entire missing retinal layer from the remaining layers:

* Mask SVC → reconstruct from DVC + CC
* Mask DVC → reconstruct from SVC + CC
* Mask CC → reconstruct from SVC + DVC

This compels the encoder to learn that SVC, DVC, and CC contain complementary vascular information — exactly the representation needed for downstream cardiovascular prediction.

## Pretraining Stage

Input: Unlabeled 3D OCTA volumes (can include external OCTA datasets if available)

Encoder: Shared 3D Swin-T (to be reused in fine-tuning)

Decoder: Lightweight 3D convolutional decoder (discarded after pretraining)

Objective: Mean squared error on masked layer voxels in normalized intensity space

## Fine-Tuning Stage

Freeze the pretrained encoder weights for the first 5 epochs.
Unfreeze all weights and fine-tune end-to-end with a reduced learning rate (lr × 0.1) for the imaging streams.

---

# PHASE 6 — SHARED 3D ENCODER WITH LAYER-SPECIFIC ADAPTERS

**Design rationale:**

The original approach of three fully independent 3D Swin Transformer encoders is computationally prohibitive and causes severe overfitting on a medium OCTA dataset.

The refined approach: one **shared 3D Swin-T backbone** initialized from Retinal-MAE pretraining, augmented with **lightweight layer-specific adapter token modules** attached at each transformer block.

## Shared Backbone

Architecture: 3D Swin Transformer (Swin-T configuration)

* Patch size: 4×4×4 voxels
* Window size: 7×7×7
* Embedding dimension: 96, doubled at each stage
* 4 stages with [2, 2, 6, 2] transformer blocks
* Shifted window self-attention with 3D relative position bias

## Layer-Specific Adapter Tokens

For each retinal layer (SVC, DVC, CC), prepend a small set of learnable adapter tokens (dimension 96, count 8) to the patch token sequence at each transformer stage input.

These adapter tokens:

* Are unique per layer (not shared)
* Interact with the patch tokens through self-attention
* Carry layer-identity information without duplicating the full encoder

Parameter cost per adapter: 8 × 96 × 4 stages = negligible vs. duplicating the full encoder.

## Outputs

* Fs — SVC feature map ∈ ℝ^(N×d)
* Fd — DVC feature map ∈ ℝ^(N×d)
* Fc — CC feature map ∈ ℝ^(N×d)

Where N = number of patch tokens, d = feature dimension after final stage.

**Do NOT merge layers at this stage.** Maintain independent feature sequences for cross-layer attention.

---

# PHASE 7 — CROSS-LAYER ATTENTION

Design a transformer-based cross-layer fusion module.

Goal: Model pairwise interactions between SVC, DVC, and CC feature representations before multimodal fusion.

## Architecture

For each pair of layers, compute bidirectional cross-attention:

* Q = Fs, K = Fd, V = Fd → attended SVC from DVC perspective
* Q = Fd, K = Fs, V = Fs → attended DVC from SVC perspective
* Repeat for (SVC, CC) and (DVC, CC) pairs

Use 8-head multi-head attention with dropout=0.1.

Concatenate all attended representations and apply a 2-layer MLP with LayerNorm to produce:

**F_octa — Layer-Aware OCTA Representation** ∈ ℝ^(N×d)

## Biological Justification

SVC pathology (e.g., arteriolar narrowing from hypertension) predicts DVC rarefaction. CC flow deficits correlate with SVC perfusion changes in cardiovascular disease. These cross-layer correlations are the structural signature of systemic microvascular dysfunction and must be learned explicitly rather than assumed to emerge from independent encoders.

---

# PHASE 8 — BIOMARKER BRANCH

Create a biomarker encoder.

Input: Selected biomarker vector (7–9 features post-selection, from Phase 4)

Architecture:

* FC(K → 128) + GELU + LayerNorm + Dropout(0.2)
* FC(128 → 256) + GELU + LayerNorm
* Single-head self-attention over biomarker feature tokens (treat each biomarker as a token of dimension 32, project and stack)
* Output: biomarker representation Fb ∈ ℝ^(1×256)

The self-attention step allows the model to learn which biomarkers are most jointly informative rather than treating them as an unordered flat vector.

---

# PHASE 9 — CLINICAL FEATURE BRANCH

Create a clinical metadata encoder.

Input examples:

* Age (continuous, z-score normalized)
* Sex (binary embedding)
* BMI (continuous, z-score normalized)
* Diabetes (binary)
* Hypertension (binary)
* Smoking (ordinal: never/former/current → embed)
* Cholesterol (continuous)
* eGFR if available

Architecture:

* Embed categorical features (lookup table, dim=16)
* Concatenate with normalized continuous features
* FC(D_in → 128) + GELU + LayerNorm + Dropout(0.3)
* FC(128 → 128) + GELU + LayerNorm + Dropout(0.3)
* FC(128 → 256) + GELU
* Output: clinical representation Fc_meta ∈ ℝ^(1×256)

Explain why GELU is preferred over ReLU for tabular clinical data (smoother gradient flow, better performance on clinical feature distributions).

---

# PHASE 10 — UNIFIED HIERARCHICAL CROSS-MODAL TRANSFORMER (UHCMT)

**Design rationale:**

Simple concatenation discards inter-modal relationships. Two separate attention stages (cross-layer then multimodal) produce representation fragmentation — the second stage has no mechanism to route information back to the layer-specific representations.

The Unified Hierarchical Cross-Modal Transformer (UHCMT) performs both operations in a single coherent transformer:

## Architecture

**Stage 1 — Cross-Layer Integration** (from Phase 7):
F_octa is the joint layer-aware representation from cross-layer attention.

**Stage 2 — Cross-Modal Fusion:**

* F_octa attends to Fb (biomarker) via cross-attention: Q=F_octa, K=Fb, V=Fb
* F_octa attends to Fc_meta (clinical) via cross-attention: Q=F_octa, K=Fc_meta, V=Fc_meta
* Biomarker and clinical modalities also cross-attend to F_octa for bidirectional information flow

Extract CLS token from each modality stream after fusion.

Concatenate CLS tokens: [CLS_octa ; CLS_bio ; CLS_clin]

Apply LayerNorm → FC(3d → 512) → GELU

Output: **H — Shared Multimodal Representation** ∈ ℝ^512

## Why UHCMT outperforms concatenation:

Concatenation treats modalities as independent feature sets. UHCMT allows imaging features to be re-weighted based on which biomarkers are abnormal, and clinical risk factors to modulate which OCTA regions receive attention. This mirrors clinical reasoning: a diabetic patient's DVC rarefaction is weighted differently than the same finding in a normotensive non-diabetic.

---

# PHASE 11 — MULTI-TASK LEARNING

Create three simultaneous prediction tasks from the shared representation H.

## Task 1 — Cardiovascular Risk Prediction

Head: FC(512 → 128 → 1) + Sigmoid

Loss: Binary Cross-Entropy with label smoothing (ε=0.1)

**Uncertainty quantification:**
Apply Monte Carlo Dropout (p=0.3) at inference time.
Run 30–50 stochastic forward passes.
Report:
* Mean prediction: P(CVD)
* Epistemic uncertainty: σ² across passes
* Calibration loss (temperature scaling): L_cal = ECE

Final output: P(CVD) ± σ

## Task 2 — Retinal Biomarker Prediction

Head: FC(512 → 256 → K) where K = number of selected biomarkers

Loss: Smooth L1 (Huber) loss — robust to biomarker outliers

This auxiliary task forces H to retain biomarker-relevant structure, acting as a regularizer that prevents the imaging branch from ignoring interpretable vascular signals.

## Task 3 — Segmentation Supervision

Decoder: Reuse U-Net++ skip connections from Phase 3.

Loss: Combined Tversky + Focal (as defined in Phase 3).

This provides dense spatial supervision that anchors the encoder to anatomically meaningful vessel structures.

## Dynamic Multi-Task Loss (Gradient Surgery)

**Replace static λ weights with PCGrad (Gradient Surgery):**

For each pair of task gradients (g_i, g_j):
If cos(g_i, g_j) < 0 (conflicting gradients):
   Project g_i onto the normal plane of g_j
   Replace g_i with projected g_i

This prevents any single task from dominating training and eliminates the need for manual λ hyperparameter search.

Total loss form for monitoring:

L_total = w₁·L_risk + w₂·L_biomarker + w₃·L_seg

Where w₁, w₂, w₃ are dynamically updated per batch by PCGrad.

Explain rationale for each task and how they mutually regularize each other.

---

# PHASE 12 — TRAINING STRATEGY

Provide complete training configuration:

## Optimizer

AdamW
* Learning rate: 3e-4 (imaging backbone: 3e-5 after unfreeze)
* Weight decay: 1e-2
* Betas: (0.9, 0.999)

## Scheduler

Cosine Annealing with Warm Restarts (T_0=10, T_mult=2)
* Linear warmup for first 5 epochs
* This avoids early instability when transitioning from frozen to unfrozen backbone

## Batch Size

16 (gradient accumulation × 4 if GPU memory limited, effective batch = 64)

## Cross-Validation

Stratified 5-fold cross-validation
* Stratify on CVD label AND age decade to prevent demographic leakage
* Report mean ± std across folds for all metrics

## Early Stopping

Patience = 15 epochs on validation AUC

## Augmentation Pipeline

For 3D OCTA volumes:
* Random horizontal/vertical flip (p=0.5)
* Random rotation ±15° in axial plane (p=0.3)
* Elastic deformation (α=1, σ=8, p=0.3)
* Intensity jitter (brightness ±0.1, contrast ±0.1, p=0.4)
* Gaussian noise (σ=0.01, p=0.2)
* MixUp-OCTA: blend two OCTA volumes with their labels (α=0.4) — applied at the volume level, not pixel level, to preserve vascular structure

For clinical and biomarker inputs:
* Gaussian noise on continuous features (σ=0.05, p=0.3)
* Random feature dropout (mask one feature to zero, p=0.1) — simulates missing clinical data at test time

## Pretraining → Fine-Tuning Schedule

1. Pretrain Retinal-MAE on unlabeled OCTA (50 epochs)
2. Freeze backbone, train heads only (10 epochs, lr=3e-4)
3. Unfreeze all, end-to-end fine-tune (100 epochs, backbone lr=3e-5)

Explain every hyperparameter choice.

---

# PHASE 13 — EXPLAINABILITY

Implement four complementary explainability methods:

## Grad-CAM 3D

Apply gradient-weighted class activation mapping to the final convolutional feature maps of the 3D Swin encoder.

Generate per-layer (SVC, DVC, CC) volumetric heatmaps showing which voxel regions most influenced the CVD risk prediction.

Overlay on original OCTA en-face projections for clinical presentation.

## Attention Rollout

Propagate attention weights through all transformer layers of the UHCMT.

Identify which retinal layer (SVC, DVC, or CC) dominated the multimodal fusion decision for each patient.

Visualize cross-layer attention matrices to reveal learned inter-layer dependencies.

## SHAP Values

Apply KernelSHAP to the biomarker and clinical branches.

Generate:
* Per-patient biomarker contribution waterfall plots
* Population-level beeswarm plots
* Dependence plots for top biomarkers vs CVD risk

## Feature Importance Analysis

Global permutation importance across biomarker features.
Correlation of biomarker SHAP values with Grad-CAM hotspot locations (do spatial imaging findings agree with biomarker importances?).

All explainability outputs must be presented in clinician-readable format with anatomical annotations.

---

# PHASE 14 — EVALUATION

Evaluate using the following metrics, all with 95% bootstrap confidence intervals (n=1000 bootstrap samples):

## Primary Metrics

* ROC-AUC (primary endpoint)
* PR-AUC (important for class imbalance)
* Sensitivity at 90% specificity (clinically relevant operating point)
* Specificity at 90% sensitivity

## Secondary Metrics

* Accuracy, Precision, Recall, F1 at optimal Youden threshold
* Expected Calibration Error (ECE) — model reliability
* Brier Score — probabilistic accuracy
* Decision Curve Analysis (DCA) — net clinical benefit across threshold range

## Uncertainty Evaluation

* Coverage of 95% prediction intervals (should be ~95%)
* Mean uncertainty vs. prediction error correlation
* Uncertainty in high-risk vs. low-risk quartiles

## Statistical Comparisons

* DeLong test for AUC comparison between proposed model and each baseline
* Report p-values with Bonferroni correction for multiple comparisons

Provide healthcare-specific interpretation of all metrics.

---

# PHASE 15 — ABLATION STUDIES

Perform systematic ablation. For each configuration, retrain from scratch with identical seeds and report mean ± std over 5 folds.

| Configuration | Change |
|---|---|
| Full model | Proposed architecture |
| −Layer adapters | Replace with 3 independent encoders |
| −Cross-layer attention | Concatenate Fs, Fd, Fc directly |
| −UHCMT | Replace with simple concatenation + MLP |
| −Biomarker branch | Remove Fb, clinical only |
| −Clinical branch | Remove Fc_meta, biomarkers only |
| −Retinal-MAE pretraining | Train encoder from scratch |
| −Uncertainty (MC-Dropout) | Single deterministic forward pass |
| −Dynamic loss (PCGrad) | Fixed λ₁=1.0, λ₂=0.5, λ₃=0.3 |
| −Segmentation supervision | Remove Task 3 |
| Biomarker-only baseline | No imaging, clinical + biomarkers only |
| Imaging-only baseline | No clinical or biomarker branches |

Analyze each ablation to quantify the contribution of each architectural decision.

---

# PHASE 16 — STATISTICAL VALIDATION

Perform:

* Mann-Whitney U Test — biomarker distributions between CVD vs non-CVD groups
* Paired t-test — proposed model vs. each baseline on AUC (across folds)
* One-way ANOVA — performance across age/sex subgroups (fairness analysis)
* Pearson/Spearman correlation — predicted CVD risk vs. clinical risk scores (Framingham, SCORE2)
* Logistic regression baseline — using only selected biomarkers + clinical features
* Partial correlation — biomarker-risk relationship controlling for age and sex

Report effect sizes (Cohen's d, rank-biserial r) alongside p-values.

---

# PHASE 17 — BASELINES

Compare proposed model against:

| Baseline | Type |
|---|---|
| Logistic Regression | Classical — biomarkers + clinical only |
| Random Forest | Ensemble — biomarkers + clinical |
| XGBoost | Gradient boosting — biomarkers + clinical |
| LightGBM | Gradient boosting — biomarkers + clinical |
| ResNet50 2D | CNN — en-face images only |
| ConvNeXt-T | CNN — en-face images only |
| Vision Transformer (ViT-S) | Transformer — en-face images only |
| 3D ResNet-18 | CNN — angiocube only |
| Biomarker-only MLP | MLP — no imaging |
| Clinical-only MLP | MLP — no biomarkers or imaging |
| Proposed (full) | UHCMT multimodal |

For imaging baselines, use the same augmentation pipeline and training schedule.
For classical baselines, use the same biomarker selection strategy.

---

# PHASE 18 — NOVEL RESEARCH CONTRIBUTIONS

Identify and articulate the following novel contributions for publication:

1. **Layer-Masked Retinal-MAE**: First application of layer-level masked autoencoding for OCTA pretraining — forces cross-layer representation learning aligned with the biological hypothesis.

2. **Layer-Specific Adapter Architecture**: Parameter-efficient multi-layer OCTA encoding via shared backbone + adapter tokens — achieves layer specificity without triplicated encoder cost.

3. **UHCMT Fusion**: Unified hierarchical cross-modal transformer that performs cross-layer and cross-modality attention in a single coherent architecture — superior to staged fusion.

4. **Uncertainty-Aware CVD Screening**: First integration of MC-Dropout uncertainty quantification in retinal OCTA-based cardiovascular prediction — critical for clinical trust and triage.

5. **PCGrad Multi-Task Training**: Dynamic gradient surgery for conflicting segmentation, biomarker, and classification tasks — removes manual λ tuning and improves training stability.

6. **Biomarker-Guided Cross-Modal Attention**: Interpretable fusion where clinical biomarkers modulate imaging attention weights — preserves clinical reasoning structure within deep learning.

---

# PHASE 19 — FINAL DELIVERABLES

Provide all of the following:

1. **Full architecture diagram** — annotated with tensor dimensions at each stage
2. **Training pipeline flowchart** — pretraining → fine-tuning → evaluation
3. **Data flow diagram** — from raw OCTA to clinical output
4. **Mathematical formulation** — complete notation for all losses, attention mechanisms, and biomarker computations
5. **Pseudocode** — PyTorch-style, covering: Retinal-MAE, shared encoder with adapters, UHCMT, multi-task heads, PCGrad loss, MC-Dropout inference
6. **PyTorch implementation plan** — file structure, class hierarchy, training loop, evaluation loop
7. **Hyperparameter table** — all values with justification
8. **Publication-ready methodology section** — IEEE TMI / Nature Biomedical Engineering style
9. **Experimental results section** — with placeholder tables for all metrics, ablations, and baseline comparisons
10. **Research paper outline** — title, abstract structure, section breakdown, figure list, contribution summary

Design the strongest publishable model possible while maintaining clinical relevance, interpretability, uncertainty awareness, and feasibility on a medium-sized OCTA dataset.
