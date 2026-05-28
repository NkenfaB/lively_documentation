# Algorithm methods summary

This document summarises every algorithm trained or designed for the cough-screening
task so the supervisor's request for *all the different algorithms* is satisfied
in one place. It is cited from Chapter 3 (Materials and Methods) and
Chapter 4 (Results and Applications).

## Dataset used for all algorithms

| Split | COVID | HEALTHY_OR_NONTARGET | Total |
|---|---|---|---|
| Train | 991 | 3317 | 4308 |
| Val | 204 | 550 | 754 |
| Test | 204 | 550 | 754 |

Sources: **Coswara** (Zenodo 7188627) and **COUGHVID** (Zenodo 4498364), both CC-BY 4.0.
Feature: **log-mel spectrogram (64 × 256 × 1)**, min-max scaled to [0, 1].
`HEALTHY_OR_NONTARGET` is a mixed control class — documented as a limitation in the report.

---

## 1. Baseline CNN from scratch

- **File:** `src/models/baseline_cnn.py` + `src/models/training.py`
- **Type:** Supervised learning, trained from scratch.
- **Architecture:** 3 × (Conv2D → BatchNorm → MaxPool), GlobalAveragePooling2D,
  Dropout(0.3), Dense(64), Dense(2, softmax).
- **Input:** `(64, 256, 1)` log-mel spectrogram.
- **Decision threshold:** 0.50 (default argmax).
- **Purpose:** Establishes a reproducible reference and proves the end-to-end
  pipeline (audio → spectrogram → tensor → softmax) works.
- **TFLite status:** Not attempted at this stage.

---

## 2. Class-weighted CNN

- **File:** `train_with_class_weights.py`
- **Type:** Supervised learning, from scratch, sklearn `balanced` class weights
  (COVID weight ≈ 2.17, HEALTHY weight ≈ 0.65).
- **Outcome:** Underperformed — uncapped balanced weights pushed the network to
  over-predict COVID, harming accuracy and healthy recall.
- **Why kept:** Negative result; motivates capped or moderated weighting in later variants.

---

## 3. Improved / regularised CNN

- **File:** `train_improved.py`
- **Type:** Supervised learning, from scratch. Extra Dropout layers (0.3–0.5
  after each Conv block), lower LR (1e-4), class weights capped at 3.0, early
  stopping (patience 5).
- **Outcome:** More stable training but did not beat the threshold-tuned baseline
  on COVID recall.
- **Why kept:** Shows regularisation alone cannot overcome the small-dataset
  ceiling — motivates the transfer-learning upgrade.

---

## 4. Threshold-tuned baseline CNN — previous best

- **Artefact:** `thesis_final_artifacts/final_screening_model_threshold035.keras`
- **Type:** Decision-rule applied to Algorithm 1's softmax output.
- **Decision rule:** `covid_prob >= 0.35 → COVID, else HEALTHY_OR_NONTARGET`
- **Test-set results (screening mode, threshold 0.35):**

| Metric | Value |
|---|---|
| Accuracy | 0.6724 |
| COVID precision | 0.4201 |
| COVID recall | 0.5539 |
| COVID F1 | 0.4778 |
| Healthy recall | 0.7164 |
| Confusion matrix | [[113, 91], [156, 394]] |

- **Why this threshold:** A screening tool prioritises sensitivity. Lowering from 0.50
  to 0.35 trades precision for COVID recall — the appropriate trade-off for a rapid
  screening aid (not a diagnosis).
- **TFLite status:** Export failed — LLVM crash (`Failed to infer result type(s)`)
  in TF 2.16 when converting from the Keras model directly.

---

## 5. MobileNetV2 transfer learning — new best (trained on RunPod)

- **Files:** `src/models/train_mobilenetv2_transfer.py`, `scripts/run_transfer_learning.sh`
- **Checkpoint:** `models/checkpoints/mobilenetv2_screening.keras`
- **TFLite:** `models/exported/mobilenetv2_screening.tflite`
- **Type:** Supervised transfer learning.
- **Backbone:** `MobileNetV2(include_top=False, weights='imagenet')` — 154 layers,
  pretrained on ImageNet.
- **In-graph preprocessing (serializable, no Lambda layer):**
  - `Resizing(224, 224)` — bilinear upsample from 64×256.
  - `Concatenate([x, x, x])` — grayscale → 3-channel.
  - `Rescaling(255.0)` then `Rescaling(1/127.5, offset=-1)` — equivalent to
    `mobilenet_v2.preprocess_input` but fully Keras-serializable.
- **Replaced head:** GlobalAveragePooling2D → Dropout(0.3) → Dense(128, ReLU) →
  Dropout(0.3) → Dense(2, softmax).
- **Class weights:** sklearn `balanced` (COVID ≈ 2.17, HEALTHY ≈ 0.65).
- **Two-phase training (RunPod CPU, TF 2.16.2):**
  1. Phase 1 — backbone frozen, head only. Adam LR=1e-3, 10 epochs, early stop patience 4.
  2. Phase 2 — top 30 backbone layers unfrozen (BN kept frozen). Adam LR=1e-5, 5 epochs.
- **TFLite export fix:** Exported via `model.export(saved_model_path)` first, then
  `TFLiteConverter.from_saved_model()` with `SELECT_TF_OPS`. This bypasses the
  LLVM crash that affected the direct Keras→TFLite path.
- **Test-set results (threshold 0.35):**

| Metric | Value | vs. Algorithm 4 |
|---|---|---|
| Accuracy | 0.6711 | −0.0013 |
| COVID precision | 0.4197 | −0.0004 |
| **COVID recall** | **0.5637** | **+0.0098** |
| **COVID F1** | **0.4812** | **+0.0034** |
| Healthy recall | 0.7109 | −0.0055 |
| Confusion matrix | [[115, 89], [159, 391]] | — |

- **Interpretation:** Improves the two primary screening metrics (COVID recall and F1)
  over the best previous baseline, with negligible trade-off on other metrics.
  Also solves the TFLite export failure, enabling mobile deployment.

---

## 6. Future extension — PNEUMONIA then TB

### Next target: PNEUMONIA

- **Intended classes:** `COVID`, `PNEUMONIA`, `HEALTHY_OR_NONTARGET`
- **Candidate dataset:** Sound-Dr (arXiv:2201.04581, GitHub: ReML-AI/Sound-Dr)
- **Status: blocked** — download access, licence, per-sample PNEUMONIA label, and
  cough-file isolation from breathing recordings must all be confirmed before integration.
- COUGHVID LRTI label is **not** mapped to PNEUMONIA (LRTI groups pneumonia +
  bronchitis + obstructive disease with no sub-label).
- CFCS pediatric dataset (FigShare) has explicit PNEUMONIA labels but is ages 0–11,
  82 samples — too small and age-restricted.

### Later: Tuberculosis

- **Intended classes:** `COVID`, `PNEUMONIA`, `TB`, `HEALTHY_OR_NONTARGET`
- **Dataset:** CODA / UCSF (Synapse `syn31472953`) — controlled access, pending.
- Pipeline already pre-wired: `data/raw/tb/`, TB label rules in `label_schema.py`,
  TB audit slot in `metadata/raw/tb_audit.json`.

### Excluded: Asthma

Most freely-available asthma datasets are lung auscultation, not cough recordings —
modality mismatch. Excluded from all stages of the roadmap.
