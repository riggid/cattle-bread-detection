import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import TensorBoard
import os
import json
import datetime

# Paths
DATA_DIR = "data/raw"
MODEL_PATH = "models/cattle_cnn.h5"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
EPOCHS = 5   # increase if you have more data/time

# Data Generators
train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    subset="validation"
)

# Save class names (breed labels)
class_names = list(train_gen.class_indices.keys())
os.makedirs("models", exist_ok=True)
with open(CLASS_NAMES_PATH, "w") as f:
    json.dump(class_names, f)

# Base Model (Transfer Learning)
base_model = MobileNetV2(weights="imagenet", include_top=False, input_shape=(224,224,3))
base_model.trainable = False  # freeze base layers

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
predictions = Dense(train_gen.num_classes, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# TensorBoard setup
log_dir = os.path.join("logs", datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)

# Train
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    callbacks=[tensorboard_cb]
)

# Save model
model.save(MODEL_PATH)
print(f"✅ Model saved at {MODEL_PATH}")
print(f"✅ Class names saved at {CLASS_NAMES_PATH}")
print(f"📊 Run this in a new terminal to launch TensorBoard:\n   tensorboard --logdir logs")
