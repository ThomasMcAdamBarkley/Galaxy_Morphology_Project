"""
train_hierarchical.py — Train 3 binary classifiers mirroring the Galaxy Zoo decision tree.

Decision tree implemented:
  Q1 (clf_top)  : Smooth / Features-Disk / Star-Artifact   — all galaxies
  Q2 (clf_edge) : Edge-on / Not-edge-on                    — Features galaxies only
  Q3 (clf_bar)  : Bar / No bar                             — Not-edge-on disks only

Usage:
    python src/train_hierarchical.py            # train all three
    python src/train_hierarchical.py --node top # train only clf_top
"""
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense,
    Dropout, Input, BatchNormalization,
)
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
# Shared CNN — same architecture for every node
# ---------------------------------------------------------------------------
def build_model(num_classes: int) -> tf.keras.Model:
    return Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        Conv2D(32,  (3, 3), activation="relu", padding="same"),
        BatchNormalization(), MaxPooling2D(2, 2),

        Conv2D(64,  (3, 3), activation="relu", padding="same"),
        BatchNormalization(), MaxPooling2D(2, 2),

        Conv2D(128, (3, 3), activation="relu", padding="same"),
        BatchNormalization(), MaxPooling2D(2, 2),

        Conv2D(256, (3, 3), activation="relu", padding="same"),
        BatchNormalization(), MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax"),
    ])


def make_callbacks(model_path: str):
    return [
        EarlyStopping(monitor="val_loss", patience=7,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(model_path, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=3, min_lr=1e-6, verbose=1),
    ]


def plot_history(history, node_name: str):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(f"Training history — {node_name}")

    ax1.plot(history.history["accuracy"],     label="Train")
    ax1.plot(history.history["val_accuracy"], label="Validation")
    ax1.set_title("Accuracy"); ax1.set_xlabel("Epoch"); ax1.legend()

    ax2.plot(history.history["loss"],     label="Train")
    ax2.plot(history.history["val_loss"], label="Validation")
    ax2.set_title("Loss"); ax2.set_xlabel("Epoch"); ax2.legend()

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

    # 1. Filter dataframe to the relevant sub-population
    if cfg["filter_col"] is not None:
        df = df_full[df_full[cfg["filter_col"]] >= BRANCH_THRESHOLD].copy()
        print(f"  Filtered to {len(df):,} galaxies "
              f"({cfg['filter_col']} >= {BRANCH_THRESHOLD})")
    else:
        df = df_full.copy()
        print(f"  Using full dataset: {len(df):,} galaxies")

    # 2. Assign a hard label via argmax over the node's probability columns
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
    train_gen = train_datagen.flow_from_dataframe(
        df, subset="training", shuffle=True, **common)
    val_gen = val_datagen.flow_from_dataframe(
        df, subset="validation", shuffle=False, **common)

    print(f"  Class indices: {train_gen.class_indices}")

    # 4. Class weights — balance against imbalanced nodes (esp. clf_top Class1.3)
    # Cap at MAX_WEIGHT: raw balanced weights for Class1.3 (~59 samples) reach
    # ~348x which causes gradient explosion. 20x is still strongly corrective.
    MAX_WEIGHT = 20.0
    class_names_sorted = sorted(train_gen.class_indices.keys())
    raw_weights = compute_class_weight(
        "balanced",
        classes=np.array(class_names_sorted),
        y=df["class_label"].values,
    )
    raw_weights = np.clip(raw_weights, 1.0, MAX_WEIGHT)
    class_weight = {train_gen.class_indices[c]: w for c, w in zip(class_names_sorted, raw_weights)}
    print(f"  Class weights: { {c: f'{class_weight[train_gen.class_indices[c]]:.2f}' for c in class_names_sorted} }")

    # 5. Build + compile
    model = build_model(len(train_gen.class_indices))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    # 6. Train
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path = str(MODELS_DIR / cfg["model_file"])

    history = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=make_callbacks(model_path),
        class_weight=class_weight,
    )

    plot_history(history, node_name)
    print(f"  Best model saved -> {model_path}")
    return history


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
