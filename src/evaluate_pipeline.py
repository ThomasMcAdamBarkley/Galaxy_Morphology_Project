"""
evaluate_pipeline.py — End-to-end evaluation of the full Q1→Q2→Q3 decision tree.

Routes each validation image through:
  Q1 (clf_top)  : Smooth / Features-Disk / Star-Artifact
  Q2 (clf_edge) : Edge-on / Not-edge-on        (Features-Disk only)
  Q3 (clf_bar)  : Bar / No-bar                 (Not-edge-on only)

Outputs:
  outputs/pipeline_report.txt        — per-node accuracy + confusion matrices
  outputs/pipeline_confusion.png     — 3-panel confusion matrix figure
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.preprocessing.image import ImageDataGenerator

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())

IMG_H       = _cfg["training"]["img_height"]
IMG_W       = _cfg["training"]["img_width"]
BATCH       = _cfg["training"]["batch_size"]
VAL_SPLIT   = _cfg["training"]["validation_split"]
SEED        = _cfg["training"]["random_seed"]
THRESHOLD   = _cfg["training"]["branch_threshold"]

TRAIN_DIR   = str(PROJECT_ROOT / _cfg["paths"]["train_dir"])
CSV_PATH    = str(PROJECT_ROOT / _cfg["paths"]["csv_path"])
MODELS_DIR  = PROJECT_ROOT / _cfg["paths"]["models_dir"]
OUTPUTS_DIR = PROJECT_ROOT / _cfg["paths"]["outputs_dir"]

CLASSES_TOP  = ["Class1.1", "Class1.2", "Class1.3"]   # Smooth, Features, Artifact
CLASSES_EDGE = ["Class2.1", "Class2.2"]                # Edge-on, Not-edge-on
CLASSES_BAR  = ["Class4.1", "Class4.2"]                # Bar, No-bar


# ── helpers ───────────────────────────────────────────────────────────────────

def make_val_gen(df, y_col, classes):
    dg = ImageDataGenerator(rescale=1./255, validation_split=VAL_SPLIT)
    return dg.flow_from_dataframe(
        dataframe=df,
        directory=TRAIN_DIR,
        x_col="filename",
        y_col=y_col,
        target_size=(IMG_H, IMG_W),
        batch_size=BATCH,
        class_mode="categorical",
        subset="validation",
        shuffle=False,
        seed=SEED,
    )


def predict_all(model, gen):
    gen.reset()
    preds, labels = [], []
    for _ in range(len(gen)):
        imgs, lbl = next(gen)
        preds.append(model.predict(imgs, verbose=0))
        labels.append(lbl)
    return np.concatenate(preds), np.concatenate(labels)


def save_confusion(y_true, y_pred, names, ax, title):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=names)
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    lines = []

    def log(s=""):
        print(s)
        lines.append(s)

    log("Loading models...")
    clf_top  = tf.keras.models.load_model(MODELS_DIR / "clf_top.keras")
    clf_edge = tf.keras.models.load_model(MODELS_DIR / "clf_edge.keras")
    clf_bar  = tf.keras.models.load_model(MODELS_DIR / "clf_bar.keras")
    log(f"  clf_top  : {clf_top.output_shape}")
    log(f"  clf_edge : {clf_edge.output_shape}")
    log(f"  clf_bar  : {clf_bar.output_shape}")

    df = pd.read_csv(CSV_PATH)
    df["filename"] = df["GalaxyID"].astype(str) + ".jpg"

    # ── Q1: top-level 3-class ─────────────────────────────────────────────────
    log("\n=== Q1 — clf_top (Smooth / Features-Disk / Star-Artifact) ===")
    df["top_label"] = df[CLASSES_TOP].idxmax(axis=1)
    gen_top = make_val_gen(df, "top_label", CLASSES_TOP)
    preds_top, labels_top = predict_all(clf_top, gen_top)
    y_true_top = np.argmax(labels_top, axis=1)
    y_pred_top = np.argmax(preds_top, axis=1)
    class_names_top = list(gen_top.class_indices.keys())
    log(classification_report(y_true_top, y_pred_top, target_names=class_names_top))

    # Map generator index back to original df rows (validation subset, no shuffle)
    val_indices = gen_top.index_array  # indices into gen_top.filenames

    # ── Q2: edge-on branch (Features-Disk galaxies only) ──────────────────────
    log("=== Q2 — clf_edge (Edge-on / Not-edge-on), Features-Disk subset ===")
    # Features-Disk = class index for Class1.2 in top generator
    feat_idx = gen_top.class_indices.get("Class1.2", 1)
    features_mask = y_true_top == feat_idx

    df_edge = df.iloc[val_indices[features_mask]].copy().reset_index(drop=True)
    df_edge["edge_label"] = df_edge[CLASSES_EDGE].idxmax(axis=1)

    y_true_edge, y_pred_edge, class_names_edge = [], [], []
    if len(df_edge) > 0:
        gen_edge = make_val_gen(df_edge, "edge_label", CLASSES_EDGE)
        preds_edge, labels_edge = predict_all(clf_edge, gen_edge)
        y_true_edge = np.argmax(labels_edge, axis=1)
        # Apply branch threshold — low-confidence predictions stay as "uncertain"
        max_conf = np.max(preds_edge, axis=1)
        y_pred_edge = np.where(max_conf >= THRESHOLD,
                               np.argmax(preds_edge, axis=1), -1)
        y_pred_edge_rep = np.argmax(preds_edge, axis=1)  # for report (no -1)
        class_names_edge = list(gen_edge.class_indices.keys())
        log(classification_report(y_true_edge, y_pred_edge_rep,
                                   target_names=class_names_edge))
        log(f"  Below threshold ({THRESHOLD}): {(max_conf < THRESHOLD).sum()} / {len(max_conf)} samples")
    else:
        log("  No Features-Disk samples in validation split.")

    # ── Q3: bar branch (Not-edge-on disks only) ───────────────────────────────
    log("=== Q3 — clf_bar (Bar / No-bar), Not-edge-on subset ===")
    y_true_bar, y_pred_bar, class_names_bar = [], [], []

    if len(df_edge) > 0 and len(class_names_edge) > 0:
        not_edge_idx = gen_edge.class_indices.get("Class2.2", 1)
        not_edge_mask_local = y_true_edge == not_edge_idx
        df_bar = df_edge.iloc[np.where(not_edge_mask_local)[0]].copy().reset_index(drop=True)
        df_bar["bar_label"] = df_bar[CLASSES_BAR].idxmax(axis=1)

        if len(df_bar) > 0:
            gen_bar = make_val_gen(df_bar, "bar_label", CLASSES_BAR)
            preds_bar, labels_bar = predict_all(clf_bar, gen_bar)
            y_true_bar = np.argmax(labels_bar, axis=1)
            y_pred_bar = np.argmax(preds_bar, axis=1)
            class_names_bar = list(gen_bar.class_indices.keys())
            log(classification_report(y_true_bar, y_pred_bar,
                                       target_names=class_names_bar))
        else:
            log("  No Not-edge-on samples available for Q3.")
    else:
        log("  Skipped (no Q2 results).")

    # ── Confusion matrix figure ───────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Full Pipeline — Per-Node Confusion Matrices", fontsize=13)

    save_confusion(y_true_top, y_pred_top, class_names_top, axes[0],
                   "Q1: clf_top (all galaxies)")

    if len(y_true_edge) > 0:
        save_confusion(y_true_edge, np.argmax(preds_edge, axis=1),
                       class_names_edge, axes[1],
                       "Q2: clf_edge (Features-Disk only)")
    else:
        axes[1].set_visible(False)

    if len(y_true_bar) > 0:
        save_confusion(y_true_bar, y_pred_bar, class_names_bar, axes[2],
                       "Q3: clf_bar (Not-edge-on only)")
    else:
        axes[2].set_visible(False)

    plt.tight_layout()
    out_fig = OUTPUTS_DIR / "pipeline_confusion.png"
    fig.savefig(str(out_fig), dpi=150, bbox_inches="tight")
    plt.close(fig)
    log(f"\nConfusion matrix figure saved to {out_fig}")

    # ── Save report ───────────────────────────────────────────────────────────
    report_path = OUTPUTS_DIR / "pipeline_report.txt"
    report_path.write_text("\n".join(lines))
    log(f"Pipeline report saved to {report_path}")


if __name__ == "__main__":
    main()
