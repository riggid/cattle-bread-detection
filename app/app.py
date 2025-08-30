import sys, os
import streamlit as st
from PIL import Image

# Make sure Python can find src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.predict import predict_breed

# Streamlit UI config
st.set_page_config(page_title="🐄 Cattle Breed Recognition", layout="centered")

st.title("🐄 Cattle Breed Recognition")
st.write("Upload a cattle image to identify its breed using AI.")

# File uploader
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Show uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp file for prediction
    temp_path = "temp.jpg"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Run prediction
    breed, confidence = predict_breed(temp_path)

    # Show result
    st.success(f"Prediction: **{breed}** ({confidence*100:.2f}% confidence)")
