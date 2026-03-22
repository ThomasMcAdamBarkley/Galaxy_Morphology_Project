import argparse
import yaml
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D, Input
from pathlib import Path

# --- CONFIGURATION ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent
_cfg = yaml.safe_load((PROJECT_ROOT / "config.yaml").read_text())
MODEL_PATH    = str(PROJECT_ROOT / "src" / "galaxy_model_augmented.keras")
DEFAULT_IMAGE = str(PROJECT_ROOT / "data" / "images_train" / "100008.jpg")
TARGET_SIZE   = (_cfg["training"]["img_height"], _cfg["training"]["img_width"])

def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    # --- RECONSTRUCTION STRATEGY ---
    # Instead of asking the old model for its internals, we rebuild a fresh graph
    # using the existing layers and weights. This bypasses Keras 3 "never called" errors.
    
    # 1. Create a fresh input placeholder
    inputs = Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
    
    # 2. Walk through the layers and connect them manually
    x = inputs
    conv_output = None
    
    for layer in model.layers:
        x = layer(x) # Connect layer
        if layer.name == last_conv_layer_name:
            conv_output = x # Capture the conv output
            
    final_output = x
    
    # 3. Create the Gradient Model from our fresh graph
    # This model outputs [Conv_Features, Prediction]
    grad_model = Model(inputs, [conv_output, final_output])

    # --- CALCULATION ---
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # Calculate gradients
    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def main():
    parser = argparse.ArgumentParser(description="Grad-CAM analysis for galaxy morphology model")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Path to input galaxy image")
    parser.add_argument("--output", default=str(PROJECT_ROOT / "outputs" / "gradcam.png"), help="Output path for heatmap")
    args = parser.parse_args()

    # A. Load Model
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH)

    # B. Load Image
    try:
        img = load_img(args.image, target_size=TARGET_SIZE)
    except FileNotFoundError:
        print(f"Error: Could not find image at {args.image}")
        return

    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    # C. Auto-Detect Last Conv Layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            last_conv_layer_name = layer.name
            break
            
    if not last_conv_layer_name:
        print("Error: No Conv2D layer found.")
        return
        
    print(f"Attaching Reconstruction Grad-CAM to layer: {last_conv_layer_name}")

    # D. Generate Heatmap
    try:
        heatmap = make_gradcam_heatmap(img_tensor, model, last_conv_layer_name)
    except Exception as e:
        print(f"Detailed Error: {e}")
        import traceback
        traceback.print_exc()
        return

    # E. Visualization
    heatmap = np.uint8(255 * heatmap)
    jet = plt.get_cmap("jet")
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap]

    jet_heatmap = tf.keras.preprocessing.image.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img_array.shape[1], img_array.shape[0]))
    jet_heatmap = tf.keras.preprocessing.image.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * 0.4 + (img_array * 255) * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.imsave(str(output_path), superimposed_img)
    print(f"Success! Saved '{output_path}'")

if __name__ == "__main__":
    main()