import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import json

MODEL_PATH = "models/cattle_cnn.h5"
CLASS_NAMES_PATH = "models/class_names.json"
IMG_SIZE = (224, 224)

# Load model & class names
model = tf.keras.models.load_model(MODEL_PATH)
with open(CLASS_NAMES_PATH, "r") as f:
    CLASS_NAMES = json.load(f)

def predict_breed(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    preds = model.predict(img_array)
    predicted_class = np.argmax(preds, axis=1)[0]
    confidence = np.max(preds)

    return CLASS_NAMES[predicted_class], float(confidence)
