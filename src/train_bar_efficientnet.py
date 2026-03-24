"""
train_bar_efficientnet.py — Two-phase EfficientNetB0 transfer learning for Q3 bar detection.

Replaces the custom 4-block CNN for clf_bar with an ImageNet-pretrained backbone.

Phase 1 (~10 epochs): Train head only with base frozen.
Phase 2 (~30 epochs): Fine-tune top 30 layers of EfficientNet with low lr.

Output: src/models/clf_bar.keras  (drop-in replacement, same path as existing model)
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import yaml
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.utils.class_weight import compute_class_weight
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- Config ---
_cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
IMG_HEIGHT       = _cfg["training"]["img_height"]
IMG_WIDTH        = _cfg["training"]["img_width"]
BATCH_SIZE       = _cfg["training"]["batch_size"]
BRANCH_THRESHOLD = _cfg["training"]["branch_threshold"]
VAL_SPLIT        = _cfg["training"]["validation_split"]
SEED             = _cfg["training"]["random_seed"]

TRAIN_DIR   = str(PROJECT_ROOT / _cfg["paths"]["train_dir"])
CSV_PATH    = str(PROJECT_ROOT / _cfg["paths"]["csv_path"])
MODELS_DIR  = PROJECT_ROOT / _cfg["paths"]["models_dir"]
OUTPUTS_DIR = PROJECT_ROOT / _cfg["paths"]["outputs_dir"]

MODEL_PATH  = str(MODELS_DIR / "clf_bar.keras")
HISTORY_OUT = OUTPUTS_DIR / "history_bar_efficientnet.png"

# Phase hyperparameters
PHASE1_EPOCHS = 10
PHASE1_LR     = 1e-3
PHASE2_EPOCHS = 30
PHASE2_LR     = 1e-5
FINE_TUNE_FROM = -30   # unfreeze last 30 EfficientNet layers in phase 2
MAX_WEIGHT    = 20.0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_generators(df: pd.DataFrame):
    """Build train/val ImageDataGenerator flows for the bar node."""
    # rescale=1/255 matches predict.py; a Rescaling(255.) head layer converts
    # back to [0,255] before EfficientNetB0's internal normalization.
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
    train_gen = train_datagen.flow_from_dataframe(df, subset="training",  shuffle=True,  **common)
    val_gen   = val_datagen.flow_from_dataframe(  df, subset="validation", shuffle=False, **common)
    return train_gen, val_gen


def build_class_weights(df: pd.DataFrame, class_indices: dict) -> dict:
    class_names_sorted = sorted(class_indices.keys())
    raw_weights = compute_class_weight(
        "balanced",
        classes=np.array(class_names_sorted),
        y=df["class_label"].values,
    )
    raw_weights = np.clip(raw_weights, 1.0, MAX_WEIGHT)
    return {class_indices[c]: w for c, w in zip(class_names_sorted, raw_weights)}


def plot_history(h1, h2):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4))
    fig.suptitle("EfficientNetB0 bar classifier — training history")

    p1_len = len(h1.history["accuracy"])
    epochs_p1 = range(1, p1_len + 1)
    epochs_p2 = range(p1_len + 1, p1_len + len(h2.history["accuracy"]) + 1)

    for ax, metric in zip((ax1, ax2), ("accuracy", "loss")):
        ax.plot(epochs_p1, h1.history[metric],         "C0-",  label="Phase 1 train")
        ax.plot(epochs_p1, h1.history[f"val_{metric}"],"C0--", label="Phase 1 val")
        ax.plot(epochs_p2, h2.history[metric],         "C1-",  label="Phase 2 train")
        ax.plot(epochs_p2, h2.history[f"val_{metric}"],"C1--", label="Phase 2 val")
        ax.axvline(p1_len + 0.5, color="gray", linestyle=":", linewidth=1)
        ax.set_title(metric.capitalize())
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=8)

    fig.savefig(str(HISTORY_OUT), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Training curves -> {HISTORY_OUT}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading CSV...")
    df = pd.read_csv(CSV_PATH)
    df["filename"] = df["GalaxyID"].astype(str) + ".jpg"

    # Filter to not-edge-on disks (bar node population)
    df = df[df["Class2.2"] >= BRANCH_THRESHOLD].copy()
    print(f"  Filtered to {len(df):,} galaxies (Class2.2 >= {BRANCH_THRESHOLD})")

    # Hard label via argmax
    df["class_label"] = df[["Class3.1", "Class3.2"]].idxmax(axis=1)
    print(f"  Class distribution:\n{df['class_label'].value_counts().to_string()}\n")

    train_gen, val_gen = make_generators(df)
    print(f"  Class indices: {train_gen.class_indices}")

    class_weight = build_class_weights(df, train_gen.class_indices)
    print(f"  Class weights: { {c: f'{class_weight[train_gen.class_indices[c]]:.2f}' for c in sorted(train_gen.class_indices)} }")

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Phase 1: head only ──────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Phase 1 — training head (base frozen)")
    print(f"{'='*60}")

    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_HEIGHT, IMG_WIDTH, 3),
    )
    base.trainable = False

    model = Sequential([
        tf.keras.layers.Rescaling(scale=255.),  # [0,1] -> [0,255] for EfficientNetB0
        base,
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(len(train_gen.class_indices), activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=PHASE1_LR),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary(line_length=80)

    callbacks_p1 = [
        EarlyStopping(monitor="val_loss", patience=5,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
    ]

    history1 = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=PHASE1_EPOCHS,
        callbacks=callbacks_p1,
        class_weight=class_weight,
    )

    # ── Phase 2: fine-tune top layers ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("  Phase 2 — fine-tuning top EfficientNet layers")
    print(f"{'='*60}")

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

    callbacks_p2 = [
        EarlyStopping(monitor="val_loss", patience=10,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                        save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5,
                          patience=4, min_lr=1e-7, verbose=1),
    ]

    history2 = model.fit(
        train_gen,
        steps_per_epoch=train_gen.samples // BATCH_SIZE,
        validation_data=val_gen,
        validation_steps=val_gen.samples // BATCH_SIZE,
        epochs=PHASE2_EPOCHS,
        callbacks=callbacks_p2,
        class_weight=class_weight,
    )

    plot_history(history1, history2)
    print(f"\n  Best model saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
