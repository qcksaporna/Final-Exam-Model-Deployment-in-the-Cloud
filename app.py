import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Define the weather labels
weather_labels = {0: 'Cloudy', 1: 'Rain', 2: 'Shine', 3: 'Sunrise'}

# Load the trained model
@st.cache_resource
def load_my_model():
    model_path = 'finalmodel'  # Or 'finalmodel.h5' if using HDF5 format
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

st.title("Weather Image Classifier")
st.write("Upload an image and the model will predict the weather.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    if model is not None:
        # Preprocess the image
        img = image.resize((128, 128))
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img_array = np.array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        # Make prediction
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_label = weather_labels[predicted_class]

        st.write(f"Prediction: {predicted_label}")
    else:
        st.warning("Model could not be loaded. Please check the model path.")
