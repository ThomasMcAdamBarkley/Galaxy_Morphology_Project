"""
train_hierarchical.py — Train 3 binary classifiers mirroring the Galaxy Zoo decision tree.

Decision tree implemented:
  Q1 (clf_top)  : Smooth / Features-Disk / Star-Artifact   — all galaxies
  Q2 (clf_edge) : Edge-on / Not-edge-on                    — Features galaxies only
  Q3 (clf_bar)  : Bar / No bar                             — Not-edge-on disks only

Architecture: EfficientNetB0 (ImageNet pretrained), two-phase transfer learning.
  Phase 1 (~10 epochs): Head only, base frozen, lr=1e-3
  Phase 2 (~30 epochs): Fine-tune top 30 EfficientNet layers, lr=1e-5

Usage:
    python src/train_hierarchical.py            # train all three
    python src/train_hierarchical.py --node top # train only clf_top
    python src/train_hierarchical.py --node edge
    python src/train_hierarchical.py --node bar
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- CONFIGURATION (config.yaml) ---
_cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
IMG_HEIGHT       = _cfg["training"]["img_height"]
IMG_WIDTH        = _cfg["training"]["img_width"]
BATCH_SIZE       = _cfg["training"]["batch_size"]
EPOCHS           = _cfg["training"]["epochs"]
BRANCH_THRESHOLD = _cfg["training"]["branch_threshold"]
VAL_SPLIT        = _cfg["training"]["validation_split"]
SEED             = _cfg["training"]["random_seed"]

TRAIN_DIR    = str(PROJECT_ROOT / _cfg["paths"]["train_dir"])
CSV_PATH     = str(PROJECT_ROOT / _cfg["paths"]["csv_path"])
MODELS_DIR   = PROJECT_ROOT / _cfg["paths"]["models_dir"]
OUTPUTS_DIR  = PROJECT_ROOT / _cfg["paths"]["outputs_dir"]

# Phase hyperparameters
PHASE1_EPOCHS  = 10
PHASE1_LR      = 1e-3
PHASE2_EPOCHS  = min(EPOCHS - PHASE1_EPOCHS, 40)  # cap total at config epochs
PHASE2_LR      = 1e-5
FINE_TUNE_FROM = -30   # unfreeze last 30 EfficientNet layers in phase 2
MAX_WEIGHT     = 20.0  # cap class weights to prevent gradient explosion (Class1.3)


# ---------------------------------------------------------------------------
# Decision-tree node definitions
# ---------------------------------------------------------------------------
NODES = {
    "top": {
        "classes":     ["Class1.1", "Class1.2", "Class1.3"],
        "labels":      ["Smooth", "Features-Disk", "Star-Artifact"],
        "filter_col":  None,   # always trained on full dataset
        "model_file":  "clf_top.keras",
        "description": "Q1 — Is the galaxy smooth, featured, or an artifact?",
    },
    "edge": {
        "classes":     ["Class2.1", "Class2.2"],
        "labels":      ["Edge-on", "Not-edge-on"],
        "filter_col":  "Class1.2",   # only train on featured/disk galaxies
        "model_file":  "clf_edge.keras",
        "description": "Q2 — Could this disk be viewed edge-on?",
    },
    "bar": {
        "classes":     ["Class3.1", "Class3.2"],
        "labels":      ["Bar", "No-bar"],
        "filter_col":  "Class2.2",   # only train on not-edge-on disks
        "model_file":  "clf_bar.keras",
        "description": "Q3 — Is there a bar feature through the centre?",
    },
}


# ---------------------------------------------------------------------------
# EfficientNetB0 transfer learning model
# ---------------------------------------------------------------------------
def build_efficientnet(num_classes: int):
    """Two-phase EfficientNetB0: returns (model, base) so Phase 2 can unfreeze base."""
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    # ImageDataGenerator rescales to [0,1]; EfficientNetB0 expects [0,255]
    x = tf.keras.layers.Rescaling(scale=255.)(inputs)
    x = base(x, training=False)
    x = GlobalAveragePooling2D()(x)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.5)(x)
    outputs = Dense(num_classes, activation="softmax")(x)

    model = tf.keras.Model(inputs, outputs)
    return model, base


def make_callbacks(model_path: str, patience_es: int, patience_lr: int):
    return [
        EarlyStopping(monitor="val_loss", patience=patience_es,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=patience_lr, min_lr=1e-7, verbose=1),
    ]


def plot_history(h1, h2, node_name: str):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle(f"EfficientNetB0 — {node_name} node training history")

    p1_len = len(h1.history["accuracy"])
    epochs_p1 = range(1, p1_len + 1)
    epochs_p2 = range(p1_len + 1, p1_len + len(h2.history["accuracy"]) + 1)

    for ax, metric in zip((ax1, ax2), ("accuracy", "loss")):
        ax.plot(epochs_p1, h1.history[metric],          "C0-",  label="Phase 1 train")
        ax.plot(epochs_p1, h1.history[f"val_{metric}"], "C0--", label="Phase 1 val")
        ax.plot(epochs_p2, h2.history[metric],          "C1-",  label="Phase 2 train")
        ax.plot(epochs_p2, h2.history[f"val_{metric}"], "C1--", label="Phase 2 val")
        ax.axvline(p1_len + 0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)

    out = OUTPUTS_DIR / f"history_{node_name}.png"
    fig.savefig(str(out), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves -> {out}")


# ---------------------------------------------------------------------------
# Training routine for a single node
# ---------------------------------------------------------------------------
def train_node(node_name: str, df_full: pd.DataFrame):
    cfg = NODES[node_name]
    print(f"\n{'='*60}")
    print(f"  Training node: {node_name.upper()}")
    print(f"  {cfg['description']}")
    print(f"{'='*60}")

    # 1. Filter to relevant sub-population
    if cfg["filter_col"] is not None:
        df = df_full[df_full[cfg["filter_col"]] >= BRANCH_THRESHOLD].copy()
        print(f"  Filtered to {len(df):,} galaxies "
              f"({cfg['filter_col']} >= {BRANCH_THRESHOLD})")
    else:
        df = df_full.copy()
        print(f"  Using full dataset: {len(df):,} galaxies")

    # 2. Hard label via argmax
    df["class_label"] = df[cfg["classes"]].idxmax(axis=1)
    print(f"  Class distribution:\n{df['class_label'].value_counts().to_string()}\n")

    # 3. Data generators
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=360,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        fill_mode="nearest",
        validation_split=VAL_SPLIT,
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=VAL_SPLIT)

    common = dict(
        directory=TRAIN_DIR,
        x_col="filename",
        y_col="class_label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        seed=SEED,
    )
    train_gen = train_datagen.flow_from_dataframe(df, subset="training",   shuffle=True,  **common)
    val_gen   = val_datagen.flow_from_dataframe(  df, subset="validation", shuffle=False, **common)

    print(f"  Class indices: {train_gen.class_indices}")

    # 4. Class weights (capped — Class1.3 has ~59 samples, raw ~348x causes explosion)
    class_names_sorted = sorted(train_gen.class_indices.keys())
    raw_weights = compute_class_weight(
        "balanced",
        classes=np.array(class_names_sorted),
        y=df["class_label"].values,
    )
    raw_weights = np.clip(raw_weights, 1.0, MAX_WEIGHT)
    class_weight = {train_gen.class_indices[c]: w for c, w in zip(class_names_sorted, raw_weights)}
    print(f"  Class weights: { {c: f'{class_weight[train_gen.class_indices[c]]:.2f}' for c in class_names_sorted} }")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(MODELS_DIR / cfg["model_file"])

    # 5. Build model
    model, base = build_efficientnet(len(train_gen.class_indices))

    # ── Phase 1: head only (base frozen) ────────────────────────────────────
    print(f"\n  Phase 1 — training head (base frozen), lr={PHASE1_LR}")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history1 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE1_EPOCHS,
        callbacks=make_callbacks(model_path, patience_es=5, patience_lr=3),
        class_weight=class_weight,
    )

    # ── Phase 2: fine-tune top EfficientNet layers ───────────────────────────
    print(f"\n  Phase 2 — fine-tuning top {abs(FINE_TUNE_FROM)} EfficientNet layers, lr={PHASE2_LR}")
    base.trainable = True
    for layer in base.layers[:FINE_TUNE_FROM]:
        layer.trainable = False
    trainable_count = sum(1 for l in base.layers if l.trainable)
    print(f"  EfficientNet trainable layers: {trainable_count} / {len(base.layers)}")

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE2_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    history2 = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=PHASE2_EPOCHS,
        callbacks=make_callbacks(model_path, patience_es=10, patience_lr=4),
        class_weight=class_weight,
    )

    plot_history(history1, history2, node_name)
    print(f"  Best model saved -> {model_path}")
    return history1, history2


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--node",
        choices=list(NODES.keys()) + ["all"],
        default="all",
        help="Which decision-tree node to train (default: all)",
    )
    args = parser.parse_args()

    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df["filename"] = df["GalaxyID"].astype(str) + ".jpg"

    nodes_to_train = list(NODES.keys()) if args.node == "all" else [args.node]

    for node in nodes_to_train:
        train_node(node, df)

    print("\nAll requested nodes trained.")


if __name__ == "__main__":
    main()
