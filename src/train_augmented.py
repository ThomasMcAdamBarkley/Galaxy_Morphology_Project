import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# --- CONFIGURATION (config.yaml) ---
_cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
IMG_HEIGHT  = _cfg["training"]["img_height"]
IMG_WIDTH   = _cfg["training"]["img_width"]
BATCH_SIZE  = _cfg["training"]["batch_size"]
EPOCHS      = _cfg["training"]["epochs"]
VAL_SPLIT   = _cfg["training"]["validation_split"]
SEED        = _cfg["training"]["random_seed"]

TRAIN_DIR   = str(PROJECT_ROOT / _cfg["paths"]["train_dir"])
CSV_PATH    = str(PROJECT_ROOT / _cfg["paths"]["csv_path"])
OUTPUTS_DIR = PROJECT_ROOT / _cfg["paths"]["outputs_dir"]
MODEL_OUT   = str(PROJECT_ROOT / "src" / "galaxy_model_augmented.keras")


def build_model(num_classes):
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        # Block 1
        Conv2D(32, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 2
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 3
        Conv2D(128, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        # Block 4
        Conv2D(256, (3, 3), activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax'),
    ])
    return model


def plot_history(history):
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.legend()

    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.legend()

    out = OUTPUTS_DIR / "training_history.png"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Training curves saved to {out}")


def train_model():
    if not Path(TRAIN_DIR).exists():
        print(f"ERROR: Image directory not found at {TRAIN_DIR}")
        return

    print("Loading CSV...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: CSV not found at {CSV_PATH}")
        return

    df['filename'] = df['GalaxyID'].astype(str) + ".jpg"

    # Top-level Galaxy Zoo morphology classes
    classes = ['Class1.1', 'Class1.2', 'Class1.3']
    df['class_label'] = df[classes].idxmax(axis=1)

    print(f"Class distribution:\n{df['class_label'].value_counts()}\n")

    # --- DATA GENERATORS ---
    # Augmentation applied only to training data
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=360,      # Full rotation — galaxies have no "up"
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        zoom_range=0.1,
        fill_mode='nearest',
        validation_split=VAL_SPLIT,
    )

    # Validation gets only rescaling — no augmentation
    val_datagen = ImageDataGenerator(
        rescale=1. / 255,
        validation_split=VAL_SPLIT,
    )

    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=TRAIN_DIR,
        x_col="filename",
        y_col="class_label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=SEED,
    )

    validation_generator = val_datagen.flow_from_dataframe(
        dataframe=df,
        directory=TRAIN_DIR,
        x_col="filename",
        y_col="class_label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=SEED,
    )

    num_classes = len(train_generator.class_indices)
    print(f"Classes: {train_generator.class_indices}")

    # --- MODEL ---
    model = build_model(num_classes)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='categorical_crossentropy',
        metrics=['accuracy'],
    )
    model.summary()

    # --- CALLBACKS ---
    callbacks = [
        EarlyStopping(
            monitor='val_loss',
            patience=7,
            restore_best_weights=True,
            verbose=1,
        ),
        ModelCheckpoint(
            filepath=MODEL_OUT,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=3,
            min_lr=1e-6,
            verbose=1,
        ),
    ]

    # --- TRAINING ---
    print("\nStarting training...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS,
        callbacks=callbacks,
    )

    plot_history(history)
    print(f"\nBest model saved to: {MODEL_OUT}")


if __name__ == "__main__":
    train_model()
