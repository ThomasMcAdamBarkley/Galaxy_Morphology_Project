"""
predict.py — Run a single galaxy image through the full hierarchical classifier.

Chains the 3 decision-tree nodes in order:
  clf_top  -> Q1: Smooth / Features-Disk / Star-Artifact
  clf_edge -> Q2: Edge-on / Not-edge-on          (only if Features-Disk)
  clf_bar  -> Q3: Bar / No-bar                   (only if Not-edge-on)

Usage:
    python src/predict.py --image data/images_train/100008.jpg
    python src/predict.py --image path/to/galaxy.jpg --save outputs/my_result.png
"""
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, Input
from tensorflow.keras.models import Model
import yaml
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

_cfg        = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
IMG_HEIGHT  = _cfg["training"]["img_height"]
IMG_WIDTH   = _cfg["training"]["img_width"]
MODELS_DIR  = PROJECT_ROOT / _cfg["paths"]["models_dir"]
OUTPUTS_DIR = PROJECT_ROOT / _cfg["paths"]["outputs_dir"]

# Decision-tree structure — mirrors train_hierarchical.py
TREE = [
    {
        "name":     "clf_top",
        "file":     "clf_top.keras",
        "question": "Q1: Galaxy type?",
        "labels":   ["Smooth", "Features-Disk", "Star-Artifact"],
        # Only continue down the tree when this class is predicted
        "branch_on": "Features-Disk",
    },
    {
        "name":     "clf_edge",
        "file":     "clf_edge.keras",
        "question": "Q2: Viewed edge-on?",
        "labels":   ["Edge-on", "Not-edge-on"],
        "branch_on": "Not-edge-on",
    },
    {
        "name":     "clf_bar",
        "file":     "clf_bar.keras",
        "question": "Q3: Bar feature?",
        "labels":   ["Bar", "No-bar"],
        "branch_on": None,   # terminal node
    },
]


def preprocess(image_path: str) -> np.ndarray:
    img = load_img(image_path, target_size=(IMG_HEIGHT, IMG_WIDTH))
    arr = img_to_array(img) / 255.0
    return np.expand_dims(arr, axis=0).astype(np.float32)


def _find_last_conv(model: tf.keras.Model):
    """Find the last Conv2D layer, recursing into nested sub-models."""
    for layer in reversed(model.layers):
        if hasattr(layer, 'layers'):          # nested Functional / Sequential
            found = _find_last_conv(layer)
            if found is not None:
                return found
        if isinstance(layer, Conv2D):
            return layer
    return None


def _layer_contains(parent, target) -> bool:
    """Return True if target layer is anywhere inside parent's sub-layers."""
    for l in getattr(parent, 'layers', []):
        if l is target or _layer_contains(l, target):
            return True
    return False


def gradcam_heatmap(img_tensor: np.ndarray, model: tf.keras.Model) -> np.ndarray:
    last_conv = _find_last_conv(model)
    if last_conv is None:
        raise ValueError("No Conv2D layer found in model.")

    # Rebuild the forward pass as a new functional graph so we can tap
    # last_conv's output regardless of whether model.input is defined
    # (Sequential models without an explicit Input layer don't define it).
    inputs = Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = inputs
    conv_out = None
    for layer in model.layers:
        if hasattr(layer, 'layers') and _layer_contains(layer, last_conv):
            # Nested Functional model (e.g. EfficientNetB0) — expose the
            # target Conv2D output via a temporary sub-model.
            sub = Model(inputs=layer.input, outputs=[last_conv.output, layer.output])
            conv_out, x = sub(x)
        else:
            x = layer(x)
            if layer is last_conv:
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


def overlay_heatmap(img_arr: np.ndarray, heatmap: np.ndarray) -> np.ndarray:
    heatmap_uint8 = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_heatmap = jet(np.arange(256))[:, :3][heatmap_uint8]
    jet_img = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_img = jet_img.resize((IMG_WIDTH, IMG_HEIGHT))
    jet_arr = np.array(jet_img) / 255.0
    return np.clip(jet_arr * 0.4 + img_arr * 0.6, 0, 1)


def run_inference(image_path: str, save_path: str | None = None):
    img_tensor = preprocess(image_path)
    img_display = img_tensor[0]   # (H, W, 3) for plotting

    results = []
    proceed = True

    for node in TREE:
        if not proceed:
            break

        model_path = MODELS_DIR / node["file"]
        if not model_path.exists():
            print(f"  [SKIP] Model not found: {model_path}")
            print(f"         Run: python src/train_hierarchical.py --node "
                  f"{node['name'].replace('clf_', '')}")
            break

        print(f"\n{node['question']}")
        model = load_model(str(model_path))
        preds = model.predict(img_tensor, verbose=0)[0]

        pred_idx   = int(np.argmax(preds))
        pred_label = node["labels"][pred_idx]
        confidence = float(preds[pred_idx])

        heatmap = gradcam_heatmap(img_tensor, model)
        overlay = overlay_heatmap(img_display, heatmap)

        results.append({
            "question":   node["question"],
            "labels":     node["labels"],
            "probs":      preds,
            "pred_label": pred_label,
            "confidence": confidence,
            "overlay":    overlay,
        })

        print(f"  -> {pred_label}  ({confidence:.1%} confidence)")

        # Stop traversal if the predicted class doesn't open the next branch
        if node["branch_on"] is None or pred_label != node["branch_on"]:
            proceed = False

    # ---- Print morphology report ----
    print("\n" + "="*50)
    print("  MORPHOLOGY REPORT")
    print("="*50)
    for r in results:
        print(f"  {r['question']}")
        print(f"    Prediction : {r['pred_label']}  ({r['confidence']:.1%})")
        for label, prob in zip(r["labels"], r["probs"]):
            bar = "#" * int(prob * 30)
            print(f"    {label:<20} {prob:.3f}  {bar}")
    print("="*50)

    # ---- Visualisation ----
    n = len(results)
    fig = plt.figure(figsize=(5 * (n + 1), 4))
    gs  = gridspec.GridSpec(1, n + 1, figure=fig)

    # Original image
    ax0 = fig.add_subplot(gs[0, 0])
    ax0.imshow(img_display)
    ax0.set_title("Input Galaxy", fontweight="bold")
    ax0.axis("off")

    # One panel per node: Grad-CAM + bar chart inset
    for i, r in enumerate(results):
        ax = fig.add_subplot(gs[0, i + 1])
        ax.imshow(r["overlay"])
        ax.set_title(r["question"], fontweight="bold")
        ax.axis("off")

        # Small bar chart as text overlay
        summary = "\n".join(
            f"{lbl}: {p:.1%}" for lbl, p in zip(r["labels"], r["probs"])
        )
        ax.text(
            0.02, 0.02, summary,
            transform=ax.transAxes,
            fontsize=7, color="white",
            verticalalignment="bottom",
            bbox=dict(boxstyle="round,pad=0.2", facecolor="black", alpha=0.6),
        )
        ax.set_xlabel(
            f"-> {r['pred_label']} ({r['confidence']:.1%})",
            fontsize=9, color="white",
            labelpad=2,
        )

    plt.tight_layout()

    if save_path:
        out = Path(save_path)
    else:
        OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
        stem = Path(image_path).stem
        out  = OUTPUTS_DIR / f"predict_{stem}.png"

    fig.savefig(str(out), dpi=150, bbox_inches="tight", facecolor="#111111")
    plt.close(fig)
    print(f"\nVisualization saved -> {out}")


def main():
    parser = argparse.ArgumentParser(
        description="Hierarchical galaxy morphology prediction with Grad-CAM"
    )
    parser.add_argument(
        "--image",
        default=str(PROJECT_ROOT / "data" / "images_train" / "100008.jpg"),
        help="Path to input galaxy image",
    )
    parser.add_argument(
        "--save",
        default=None,
        help="Optional output path for the visualization PNG",
    )
    args = parser.parse_args()

    if not Path(args.image).exists():
        print(f"Error: image not found at {args.image}")
        return

    run_inference(args.image, args.save)


if __name__ == "__main__":
    main()
