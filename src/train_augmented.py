import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
import pandas as pd
import os

# --- CONFIGURATION ---
# We use 64x64 to match your previous model's resolution
IMG_HEIGHT = 64
IMG_WIDTH = 64
BATCH_SIZE = 32
EPOCHS = 5  # Start with 5 to test it quickly

# PATHS (Verify these match your folder structure!)
# This assumes your images are in 'data/images_train'
TRAIN_DIR = r"C:\Dev\Projects\Galaxy_Morphology_Project\data\images_train"
CSV_PATH = r"C:\Dev\Projects\Galaxy_Morphology_Project\data\training_solutions_rev1.csv"

def train_model():
    print(f"Checking for data at: {TRAIN_DIR}")
    if not os.path.exists(TRAIN_DIR):
        print("ERROR: Image directory not found!")
        return

    print("Loading CSV Data...")
    try:
        df = pd.read_csv(CSV_PATH)
    except FileNotFoundError:
        print(f"ERROR: CSV file not found at {CSV_PATH}")
        return
    
    # 1. Prepare Dataframe
    # We add '.jpg' to the IDs so they match the filenames on disk
    df['filename'] = df['GalaxyID'].astype(str) + ".jpg"
    
    # 2. Define Classes (Simplified for this test)
    # We will just detect the 3 main shapes for now to prove the augmentation works
    # You can expand this logic later for the full 37 classes
    classes = ['Class1.1', 'Class1.2', 'Class1.3'] 
    df['class_label'] = df[classes].idxmax(axis=1)

    # --- THE FIX: DATA AUGMENTATION ---
    # This is the "magic" part that stops the cheating.
    print("Setting up Augmented Generators...")
    train_datagen = ImageDataGenerator(
        rescale=1./255,         # Normalize pixel values
        rotation_range=90,      # Rotate freely (galaxies have no "up")
        width_shift_range=0.1,  # Shift slightly
        height_shift_range=0.1, 
        horizontal_flip=True,   # Mirror left/right
        vertical_flip=True,     # Mirror up/down (galaxies have no gravity)
        fill_mode='nearest',    # Fill empty space with edge pixels
        validation_split=0.2    # Reserve 20% for testing
    )

    # Training Data Generator
    train_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=TRAIN_DIR,
        x_col="filename",
        y_col="class_label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    # Validation Data Generator
    validation_generator = train_datagen.flow_from_dataframe(
        dataframe=df,
        directory=TRAIN_DIR,
        x_col="filename",
        y_col="class_label",
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    # --- MODEL ARCHITECTURE ---
    print("Building Model...")
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        
        # Layer 1
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Layer 2
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        # Layer 3
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D(2, 2),

        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5), 
        Dense(len(train_generator.class_indices), activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    # --- RUN TRAINING ---
    print("Starting Training (This may take a while)...")
    history = model.fit(
        train_generator,
        steps_per_epoch=train_generator.samples // BATCH_SIZE,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // BATCH_SIZE,
        epochs=EPOCHS
    )

    # Save the fixed model
    output_path = 'src/galaxy_model_augmented.keras'
    model.save(output_path)
    print(f"Training Complete! New model saved to: {output_path}")

if __name__ == "__main__":
    train_model()