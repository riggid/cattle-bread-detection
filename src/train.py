import os
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

# --- SETTINGS ---
data_dir = "data/raw"  # Your dataset folder
img_height, img_width = 224, 224
batch_size = 32
epochs = 30
val_split = 0.2
model_save_path = "models/cattle_cnn_rgb.keras"

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

# --- STEP 1: Convert all images to RGB ---
print("Converting images to RGB...")
converted_count = 0
for root, dirs, files in os.walk(data_dir):
    for file in files:
        if file.lower().endswith((".jpg", ".jpeg", ".png")):
            img_path = os.path.join(root, file)
            try:
                img = Image.open(img_path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                    img.save(img_path)
                    converted_count += 1
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
print(f"✅ Converted {converted_count} images to RGB.")

# --- STEP 2: Load dataset ---
try:
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="training",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="rgb"
    )
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        validation_split=val_split,
        subset="validation",
        seed=123,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        color_mode="rgb"
    )
    
    class_names = train_ds.class_names
    num_classes = len(class_names)
    print("Classes:", class_names)
    print(f"Number of classes: {num_classes}")
    
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Make sure your data directory structure is correct:")
    print("data/raw/")
    print("  ├── class1/")
    print("  │   ├── image1.jpg")
    print("  │   └── image2.jpg")
    print("  └── class2/")
    print("      ├── image3.jpg")
    print("      └── image4.jpg")
    exit(1)

# --- STEP 3: Prefetch and normalize ---
AUTOTUNE = tf.data.AUTOTUNE

# Add normalization layer
normalization_layer = tf.keras.layers.Rescaling(1./255)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- STEP 4: Build model ---
# Clear any previous models from memory
tf.keras.backend.clear_session()

# Try different approaches to fix the shape mismatch
try:
    # Method 1: Use input_tensor approach
    inputs = tf.keras.Input(shape=(img_height, img_width, 3))
    base_model = EfficientNetB0(
        weights="imagenet",
        include_top=False,
        input_tensor=inputs
    )
    base_model.trainable = False
    
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    predictions = layers.Dense(num_classes, activation="softmax")(x)
    
    model = models.Model(inputs=inputs, outputs=predictions)
    
except Exception as e1:
    print(f"Method 1 failed: {e1}")
    try:
        # Method 2: Build without pretrained weights first, then load
        base_model = EfficientNetB0(
            weights=None,
            include_top=False,
            input_shape=(img_height, img_width, 3)
        )
        
        # Load ImageNet weights manually
        base_model_imagenet = EfficientNetB0(weights="imagenet", include_top=False)
        base_model.set_weights(base_model_imagenet.get_weights())
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax")
        ])
        
    except Exception as e2:
        print(f"Method 2 failed: {e2}")
        # Method 3: Use a different base model
        print("Falling back to MobileNetV2...")
        base_model = tf.keras.applications.MobileNetV2(
            weights="imagenet",
            include_top=False,
            input_shape=(img_height, img_width, 3)
        )
        base_model.trainable = False
        
        model = models.Sequential([
            base_model,
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(128, activation="relu"),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax")
        ])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("Model architecture:")
model.summary()

# --- STEP 5: Callbacks ---
checkpoint = ModelCheckpoint(
    model_save_path, 
    save_best_only=True, 
    monitor="val_accuracy", 
    mode="max",
    verbose=1
)
early_stop = EarlyStopping(
    monitor="val_accuracy", 
    patience=5, 
    restore_best_weights=True,
    verbose=1
)

# --- STEP 6: Train ---
print("Starting training...")
try:
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=[checkpoint, early_stop],
        verbose=1
    )
    print(f"✅ Model saved at {model_save_path}")
    
    # Print final results
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    print(f"Final training accuracy: {final_train_acc:.4f}")
    print(f"Final validation accuracy: {final_val_acc:.4f}")
    
except Exception as e:
    print(f"Error during training: {e}")