"""
evaluate.py — Run after training to generate metrics and visualizations.

Usage:
    python src/evaluate.py
    python src/evaluate.py --model src/galaxy_model_augmented.keras
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = str(PROJECT_ROOT / "data" / "images_train")
CSV_PATH = str(PROJECT_ROOT / "data" / "training_solutions_rev1.csv")
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
CLASSES = ['Class1.1', 'Class1.2', 'Class1.3']


def load_validation_generator(df):
    val_datagen = ImageDataGenerator(rescale=1. / 255, validation_split=0.2)
    return val_datagen.flow_from_dataframe(
        dataframe=df,
        directory=TRAIN_DIR,
        x_col="filename",
        y_col="class_label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42,
    )


def make_gradcam_heatmap(img_tensor, model):
    last_conv = next(l for l in reversed(model.layers) if isinstance(l, Conv2D))
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = inputs
    conv_out = None
    for layer in model.layers:
        x = layer(x)
        if layer.name == last_conv.name:
            conv_out = x
    grad_model = Model(inputs, [conv_out, x])

    with tf.GradientTape() as tape:
        conv_output, preds = grad_model(img_tensor)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, conv_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = conv_output[0] @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy()


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    fig, ax = plt.subplots(figsize=(7, 6))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title("Validation Confusion Matrix")
    out = OUTPUTS_DIR / "confusion_matrix.png"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Confusion matrix saved to {out}")


def plot_worst_predictions(model, df, val_gen, class_names, n=5):
    val_gen.reset()
    all_images, all_true, all_pred, all_conf = [], [], [], []

    for _ in range(len(val_gen)):
        imgs, labels = next(val_gen)
        preds = model.predict(imgs, verbose=0)
        all_images.append(imgs)
        all_true.extend(np.argmax(labels, axis=1))
        all_pred.extend(np.argmax(preds, axis=1))
        all_conf.extend(np.max(preds, axis=1))

    all_images = np.concatenate(all_images, axis=0)
    all_true = np.array(all_true)
    all_pred = np.array(all_pred)
    all_conf = np.array(all_conf)

    # Wrong predictions sorted by confidence (model was most wrong about these)
    wrong_mask = all_true != all_pred
    wrong_indices = np.where(wrong_mask)[0]
    if len(wrong_indices) == 0:
        print("No wrong predictions found on validation set.")
        return

    sorted_wrong = wrong_indices[np.argsort(all_conf[wrong_indices])[::-1]]
    top_n = sorted_wrong[:n]

    fig, axes = plt.subplots(n, 2, figsize=(10, n * 3))
    fig.suptitle(f"Top {n} High-Confidence Wrong Predictions + Grad-CAM", fontsize=13)

    for i, idx in enumerate(top_n):
        img = all_images[idx]
        img_tensor = np.expand_dims(img, axis=0).astype(np.float32)
        heatmap = make_gradcam_heatmap(img_tensor, model)

        # Overlay heatmap
        heatmap_resized = np.uint8(255 * heatmap)
        jet = plt.get_cmap("jet")
        jet_heatmap = jet(np.arange(256))[:, :3][heatmap_resized]
        jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
        jet_heatmap = jet_heatmap.resize((IMG_WIDTH, IMG_HEIGHT))
        jet_heatmap = np.array(jet_heatmap) / 255.0
        overlay = jet_heatmap * 0.4 + img * 0.6

        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"True: {class_names[all_true[idx]]}\nPred: {class_names[all_pred[idx]]} ({all_conf[idx]:.2%})")
        axes[i, 0].axis('off')

        axes[i, 1].imshow(np.clip(overlay, 0, 1))
        axes[i, 1].set_title("Grad-CAM")
        axes[i, 1].axis('off')

    plt.tight_layout()
    out = OUTPUTS_DIR / "worst_predictions_gradcam.png"
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Worst predictions + Grad-CAM saved to {out}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=str(PROJECT_ROOT / "src" / "galaxy_model_augmented.keras"))
    args = parser.parse_args()

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {args.model}...")
    model = load_model(args.model)

    print("Loading validation data...")
    df = pd.read_csv(CSV_PATH)
    df['filename'] = df['GalaxyID'].astype(str) + ".jpg"
    df['class_label'] = df[CLASSES].idxmax(axis=1)

    val_gen = load_validation_generator(df)
    class_names = list(val_gen.class_indices.keys())

    print("Running predictions on validation set...")
    val_gen.reset()
    y_true, y_pred = [], []
    for _ in range(len(val_gen)):
        imgs, labels = next(val_gen)
        preds = model.predict(imgs, verbose=0)
        y_true.extend(np.argmax(labels, axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    y_true, y_pred = np.array(y_true), np.array(y_pred)

    print("\n--- Classification Report ---")
    report = classification_report(y_true, y_pred, target_names=class_names)
    print(report)

    report_path = OUTPUTS_DIR / "classification_report.txt"
    report_path.write_text(report)
    print(f"Report saved to {report_path}")

    plot_confusion_matrix(y_true, y_pred, class_names)

    print("\nGenerating Grad-CAM for worst predictions...")
    val_gen2 = load_validation_generator(df)
    plot_worst_predictions(model, df, val_gen2, class_names, n=5)

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()
