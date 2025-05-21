import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps
import cv2

# Load trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model('final_model.h5')
    return model

model = load_model()

# Define weather class names
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

# Streamlit UI
st.title("Weather Image Classification")
st.write("""
Upload an image and the model will predict the weather condition (Cloudy, Rain, Shine, Sunrise).
""")

file = st.file_uploader("Choose a weather image", type=["jpg", "jpeg", "png"])

# Image preprocessing and prediction
def import_and_predict(image_data, model):
    size = (128, 128)
    image = image_data.convert("RGB")
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    img = np.asarray(image)

    if img.ndim == 2 or img.shape[2] != 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    img_reshape = img.reshape((1, 128, 128, 3)) / 255.0
    prediction = model.predict(img_reshape)
    return prediction

if file is None:
    st.text("Please upload an image file")
else:
    image = Image.open(file)
    st.image(image, use_column_width=True)
    prediction = import_and_predict(image, model)
    result = class_names[np.argmax(prediction)]
    st.success(f"Prediction: {result}")
