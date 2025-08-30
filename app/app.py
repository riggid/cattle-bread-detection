import streamlit as st
from PIL import Image
from src.predict import predict_breed

st.set_page_config(page_title="🐄 Cattle Breed Recognition", layout="centered")

st.title("🐄 Cattle Breed Recognition")
st.write("Upload a cattle image to identify its breed using AI.")

uploaded_file = st.file_uploader("Upload an image", type=["jpg","png","jpeg"])

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Save temp file
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())

    # Predict
    breed, confidence = predict_breed("temp.jpg")
    st.success(f"Prediction: **{breed}** ({confidence*100:.2f}% confidence)")
