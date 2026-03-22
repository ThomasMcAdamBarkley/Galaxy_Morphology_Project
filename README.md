# Galaxy Morphology Classification

Automated classification of galaxy shapes using a **hierarchical CNN pipeline**, built on the [Galaxy Zoo](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge) dataset. The project goes beyond accuracy metrics to include **Explainable AI (Grad-CAM)** at every decision step, ensuring the model learns genuine galactic structure rather than image artefacts.

---

## The "Row 3" Detective Story

During evaluation of the initial model I inspected a high-confidence prediction using Grad-CAM and discovered the model was **ignoring the galaxy entirely** — it was keying off a bright noise artefact in the corner of the image.

| Baseline model | Augmented model |
|---|---|
| Attends to background noise | Attends to the galactic nucleus |

**The fix:** a physics-aware augmentation pipeline — full 360° rotations, horizontal/vertical flips — that forces rotational invariance. Galaxies have no "up", so any orientation must produce the same prediction.

This diagnostic workflow (train → interrogate → fix → verify) is the core methodology of the project.

---

## Hierarchical Decision Tree

Rather than a flat 37-class classifier, the pipeline mirrors the Galaxy Zoo volunteer decision tree:

```
All Galaxies  (61,578)
└─ Q1 clf_top: What is the overall shape?
   ├─ Smooth          → Elliptical / Lenticular
   ├─ Features-Disk   → Continue ↓
   │   └─ Q2 clf_edge: Could this disk be viewed edge-on?
   │       ├─ Edge-on     → Edge-on disk
   │       └─ Not-edge-on → Continue ↓
   │           └─ Q3 clf_bar: Is there a bar feature?
   │               ├─ Bar    → Barred spiral
   │               └─ No-bar → Unbarred spiral
   └─ Star/Artifact   → Excluded from science sample
```

Each node trains only on the galaxies where that question is scientifically meaningful, giving each classifier a cleaner signal.

---

## Quick Start

```bash
# 1. Clone and set up environment
git clone https://github.com/TomBarkley/Galaxy_Morphology_Project.git
cd Galaxy_Morphology_Project
python -m venv galaxy_env
source galaxy_env/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

# 2. Train
make train                   # 3-class augmented baseline
make train-hierarchical      # all 3 decision-tree nodes

# 3. Evaluate
make evaluate                # confusion matrix + classification report

# 4. Predict on a single image
make predict IMAGE=data/images_train/100008.jpg

# 5. Grad-CAM only
make gradcam IMAGE=data/images_train/100008.jpg
```

All hyperparameters (image size, epochs, batch size, branch threshold) live in `config.yaml`.

---

## Project Structure

```
Galaxy_Morphology_Project/
├── config.yaml                        # All hyperparameters — edit here
├── Makefile                           # Task runner
├── requirements.txt
│
├── data/
│   ├── images_train/                  # 61,578 galaxy images (.jpg)
│   └── training_solutions_rev1.csv    # Galaxy Zoo probability labels
│
├── src/
│   ├── train_augmented.py             # Phase 2 — 3-class baseline with callbacks
│   ├── train_hierarchical.py          # Phase 3 — 3-node decision tree trainer
│   ├── evaluate.py                    # Metrics: confusion matrix, F1, Grad-CAM errors
│   ├── predict.py                     # Single-image morphology report
│   ├── analyze_native_reconstruct.py  # Standalone Grad-CAM tool (Keras 3 compatible)
│   ├── models/                        # Saved .keras model files (git-ignored)
│   │   ├── clf_top.keras
│   │   ├── clf_edge.keras
│   │   └── clf_bar.keras
│   ├── galaxy_model_augmented.keras   # Phase 2 baseline model
│   ├── 01_inspect_data.ipynb          # EDA + hierarchical pipeline walkthrough
│   └── ReadMe.md                      # Narrative development notes
│
└── outputs/                           # Generated figures (git-ignored)
    ├── training_history.png
    ├── history_top.png / history_edge.png / history_bar.png
    ├── confusion_matrix.png
    ├── classification_report.txt
    └── worst_predictions_gradcam.png
```

---

## Workflow

```bash
# Phase 2 — proper 3-class classifier
python src/train_augmented.py          # → src/galaxy_model_augmented.keras
python src/evaluate.py                 # → outputs/confusion_matrix.png
                                       #   outputs/classification_report.txt
                                       #   outputs/worst_predictions_gradcam.png

# Phase 3 — hierarchical tree
python src/train_hierarchical.py              # all nodes
python src/train_hierarchical.py --node top  # single node

# Inference
python src/predict.py --image data/images_train/100008.jpg
# → outputs/predict_100008.png  (multi-panel Grad-CAM report)
```

---

## Configuration

All tunable parameters are in `config.yaml` — no need to touch the source scripts:

```yaml
training:
  img_height: 128
  img_width:  128
  batch_size: 32
  epochs:     50          # EarlyStopping governs actual duration
  branch_threshold: 0.5   # min P(class) to enter a sub-classifier
  validation_split: 0.2
  random_seed: 42
```

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | CNN training and inference |
| `pandas`, `numpy` | Data loading and manipulation |
| `matplotlib`, `seaborn` | Visualisation |
| `scikit-learn` | Classification metrics |
| `Pillow`, `scikit-image` | Image preprocessing |
| `pyyaml` | Config loading |
| `jupyter` | Notebook support |

---

**Author:** Tom Barkley
**License:** MIT
