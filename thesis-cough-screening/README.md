# Thesis Cough Screening

Production-style scaffold for the thesis project:

`A mobile-integrated deep learning system for rapid screening of respiratory diseases using cough sound analysis`

## Project Overview

This repository sets up a local end-to-end machine learning pipeline for cough-audio screening of:

- `TB`
- `COVID`
- `HEALTHY_OR_NONTARGET`

The first version is intentionally focused on cough-compatible public datasets and a lightweight baseline model that can later be exported to TensorFlow Lite for mobile integration.

## Thesis Objective

Build a reproducible cough-audio pipeline that:

1. acquires public respiratory audio datasets,
2. audits and cleans their metadata,
3. unifies labels into a controlled schema,
4. preprocesses cough audio,
5. creates train/validation/test splits,
6. trains a baseline TensorFlow CNN on mel spectrograms,
7. evaluates the model with standard classification metrics,
8. prepares for later mobile deployment.

## Currently Implemented Scope

The currently implemented and trained task is **binary**:

- `COVID`
- `HEALTHY_OR_NONTARGET`

`HEALTHY_OR_NONTARGET` is a **mixed control class**. It may contain explicitly
healthy participants, COVID-negative participants, and other non-target controls
that are compatible with the source dataset documentation. This is documented as
a scientific limitation in `src/config/label_schema.py` and in the report.

### Future scope

- **PNEUMONIA** is the next target class. Candidate dataset: Sound-Dr
  (arXiv:2201.04581, GitHub: ReML-AI/Sound-Dr). Integration is blocked pending
  confirmation of (1) public download access and licence, (2) per-sample PNEUMONIA
  label, and (3) ability to isolate cough files from breathing recordings.
  See `notebooks/07_multidisease_dataset_plan.ipynb` for the full investigation.
- **TB** will be added when CODA / UCSF controlled-access data is granted
  (Synapse `syn31472953`). The pipeline is already pre-wired for TB.
- **Final intended scope:** `COVID`, `PNEUMONIA`, `TB`, `HEALTHY_OR_NONTARGET`.
- **Asthma is intentionally not included** — most freely available asthma
  datasets are lung auscultation rather than cough recordings, causing a
  modality mismatch with the existing corpus.
- **COUGHVID LRTI labels are not mapped to PNEUMONIA** — LRTI groups pneumonia,
  bronchitis, and obstructive disease together with no pneumonia sub-label.

## Algorithms Compared

This project compares several algorithms (see
`reports/tables/algorithm_methods_summary.md` and
`notebooks/06_algorithm_comparison.ipynb`):

1. **Baseline CNN from scratch** — `src/models/baseline_cnn.py` + `src/models/training.py`.
2. **Class-weighted CNN** — `train_with_class_weights.py` (negative result, kept on record).
3. **Improved / regularised CNN** — `train_improved.py` (negative result, kept on record).
4. **Threshold-tuned baseline CNN (current best screening result)** —
   `final_screening_model_threshold035.keras`, decision rule
   `covid_prob >= 0.35 → COVID`. Test-set: accuracy `0.6724`, COVID precision
   `0.4201`, COVID recall `0.5539`, COVID F1 `0.4778`, healthy recall `0.7164`.
5. **MobileNetV2 transfer learning (new)** —
   `src/models/train_mobilenetv2_transfer.py`. Pretrained ImageNet backbone with
   `include_top=False`, replaced classifier head, optional fine-tuning of the
   top backbone layers. Run with:

   ```bash
   bash scripts/run_transfer_learning.sh --epochs 10 --fine-tune-epochs 5
   # or
   python -m src.models.train_mobilenetv2_transfer --epochs 10 --fine-tune-epochs 5
   ```

   See `notebooks/05_transfer_learning_mobilenetv2.ipynb` and
   `reports/tables/runpod_transfer_learning_instructions.md`.

The TFLite export currently fails for the baseline in this TensorFlow / Keras
environment. MobileNetV2 is a more TFLite-friendly architecture, but no claim
of successful export is made until it is demonstrated end-to-end.

## What NOT to commit

- `data/raw/`, `data/processed/`, `data/processed/features/`
- `*.npy`, `*.wav`, `*.mp3`, `*.flac`, `*.ogg`, `*.webm`
- `.venv/`
- Large `.keras` checkpoints under `models/checkpoints/`
- Large `.tflite` artefacts under `models/exported/`
- Raw audio files of any kind

## Folder Structure

```text
thesis-cough-screening/
  README.md
  requirements.txt
  .gitignore
  .env.example
  data/
  metadata/
  notebooks/
  src/
  scripts/
  models/
  reports/
  tests/
```

## Setup

Use Python `3.10` or `3.11`.

```bash
cd ~/projects/thesis-cough-screening
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name thesis-cough-screening
```

Or run:

```bash
bash scripts/setup_env.sh
```

## Dataset Acquisition

### 1. Coswara

- Primary record: https://zenodo.org/records/7188627
- Website: https://coswara.iisc.ac.in/about
- Paper: https://www.nature.com/articles/s41597-023-02266-0

The download script uses the public Zenodo record and extracts the archive to `data/raw/coswara/`.

### 2. COUGHVID

- Dataset page: https://www.epfl.ch/labs/esl/index-html/datasets/coughviddataset/
- Public archive: https://zenodo.org/records/4498364
- Repository: https://github.com/esl-epfl/COUGHVID

The download script uses the public Zenodo archive and extracts it to `data/raw/coughvid/`.

### 3. CODA / TB dataset

- UCSF page: https://tbdata.ucsf.edu/s/rdc-dataset/a0U5w00000fTCKiEAO/ds000731
- Paper: https://www.nature.com/articles/s41597-024-03972-z
- Synapse: https://www.synapse.org/Synapse:syn31472953

This dataset is treated as manual / controlled access. The pipeline will:

- create `data/raw/tb/`,
- create `README_MANUAL_DOWNLOAD.md` there,
- skip fake downloads,
- fail gracefully downstream until the files are available.

## Label Strategy

The target schema is defined in [src/config/label_schema.py](/Users/tm/projects/thesis-cough-screening/src/config/label_schema.py).

High-level rules:

- map only clearly documented source labels,
- exclude ambiguous rows from training,
- keep `label_raw`,
- derive `label_binary` and `label_multiclass`,
- document mixed-control limitations in metadata notes.

## Pipeline Execution Order

1. Create and activate a virtual environment.
2. Install dependencies.
3. Run dataset acquisition.
4. Audit dataset metadata.
5. Build cleaned per-dataset metadata.
6. Create unified metadata and train/validation/test splits.
7. Precompute sample features if desired.
8. Train the baseline model.
9. Evaluate on the test split.
10. Export a TensorFlow Lite model scaffold.

## Assumptions And Limitations

- TB data is not downloaded automatically because access may require registration or approval.
- Dataset-specific metadata schemas differ and are handled with heuristics plus explicit label rules.
- Subject-independent splitting is used where subject identifiers exist; otherwise the pipeline falls back to sample-level stratification and logs the limitation.
- The baseline training code is thesis-friendly and easy to inspect, but not yet optimized for large-scale production training.
- The initial control class mixes healthy and broader non-target controls when necessary.

## Next Steps Toward Mobile Deployment

- stabilize label curation after manual review,
- improve cough event segmentation,
- benchmark feature extraction latency,
- export a well-performing model to TFLite,
- add quantization and mobile benchmarking,
- integrate the inference package into a mobile client once ML quality is acceptable.

## Exact Next Commands

Run these in order:

```bash
cd ~/projects/thesis-cough-screening
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
bash scripts/run_downloads.sh
bash scripts/run_audit.sh
bash scripts/run_preprocessing.sh
bash scripts/run_training.sh
bash scripts/run_evaluation.sh
python -m src.export.export_tflite --checkpoint models/checkpoints/baseline_cnn.keras --output models/exported/baseline_cnn.tflite
```

If the TB dataset is still missing, complete the manual download steps in [README_MANUAL_DOWNLOAD.md](/Users/tm/projects/thesis-cough-screening/data/raw/tb/README_MANUAL_DOWNLOAD.md) before expecting a true 3-class training run.
