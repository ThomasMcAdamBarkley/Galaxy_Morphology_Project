import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.layers import Conv2D
from tf_keras_vis.gradcam import Gradcam
from tf_keras_vis.utils.scores import CategoricalScore

# --- CONFIGURATION ---
MODEL_PATH = 'src/galaxy_model_best.keras'
# Your specific image
IMAGE_PATH = r"C:\Dev\Projects\Galaxy_Morphology_Project\data\images_train\100008.jpg"

def analyze_image():
    # 1. Load the Model
    try:
        print(f"Loading model from {MODEL_PATH}...")
        model = load_model(MODEL_PATH)
        
        # --- FIX FOR KERAS 3 ---
        if not hasattr(model, 'output_names'):
            model.output_names = [model.layers[-1].name]

    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Load and Preprocess Image (64x64)
    target_size = (64, 64)
    try:
        img = load_img(IMAGE_PATH, target_size=target_size)
    except FileNotFoundError:
        print(f"Error: Could not find image at {IMAGE_PATH}")
        return

    img_array = img_to_array(img)
    img_array = img_array / 255.0
    img_tensor = np.expand_dims(img_array, axis=0)

    # 3. Auto-detect Layer
    last_conv_layer_name = None
    for layer in reversed(model.layers):
        if isinstance(layer, Conv2D):
            last_conv_layer_name = layer.name
            break
            
    print(f"Attaching Grad-CAM to layer: {last_conv_layer_name}")

    # 4. Setup GradCAM (THE FIX: model_modifier=None)
    # We stop the library from trying to edit the model graph
    gradcam = Gradcam(model, model_modifier=None, clone=False)

    # 5. Generate Heatmap
    preds = model.predict(img_tensor)
    top_class = np.argmax(preds[0])
    print(f"Model predicts class index: {top_class} with confidence {preds[0][top_class]:.2f}")
    
    score = CategoricalScore([top_class])
    
    # Calculate cam
    cam = gradcam(score, img_tensor, penultimate_layer=last_conv_layer_name)
    
    # 6. Visualize
    heatmap = np.uint8(255 * cam[0])
    
    # Resize and colorize
    jet_heatmap = plt.get_cmap("jet")(heatmap)[:, :, :3]
    jet_heatmap = np.uint8(jet_heatmap * 255)
    jet_heatmap = tf.image.resize(jet_heatmap, (img_array.shape[0], img_array.shape[1]))
    jet_heatmap = jet_heatmap.numpy().astype(np.uint8)
    
    original_uint8 = np.uint8(img_array * 255)
    
    # Superimpose
    superimposed_img = jet_heatmap * 0.4 + original_uint8 * 0.6
    superimposed_img = np.clip(superimposed_img, 0, 255).astype(np.uint8)

    output_filename = 'row_3_analysis.png'
    plt.imsave(output_filename, superimposed_img)
    print(f"Success! Check the file '{output_filename}' to see the heatmap.")

if __name__ == "__main__":
    analyze_image()