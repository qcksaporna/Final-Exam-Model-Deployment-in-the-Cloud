import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Course & Student Info
st.markdown("""
### Course Code: CPE 019  
### Code Title: Emerging Technologies 2 in CpE  
### 2nd Semester | AY 2024-2025  
<hr>
### <u>Final Exam</u>  
**Name:** Christian Kim P. Saporna  
**Section:** CPE32S2  
""", unsafe_allow_html=True)

st.title("☁️ Weather Image Classifier")
st.write("Upload an image and get a weather prediction using a trained CNN model.")

# Load the saved Keras model once
@st.cache_resource
def load_my_model():
    try:
        model = load_model('final_model.h5', compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_my_model()

# Weather class labels
weather_labels = {0: 'Cloudy', 1: 'Rain', 2: 'Shine', 3: 'Sunrise'}

uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="📷 Uploaded Image", use_container_width=True)

    if model is not None:
        try:
            # Preprocess image
            img = image.resize((128, 128))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            img_array = np.array(img) / 255.0  # Normalize
            img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

            predictions = model.predict(img_array)
            pred_class = np.argmax(predictions)
            pred_label = weather_labels[pred_class]
            confidence = np.max(predictions) * 100

            st.markdown(f"### 🧠 Prediction: **{pred_label}**")
            st.markdown(f"### 🔍 Confidence: **{confidence:.2f}%**")

            st.subheader("📊 Class Probabilities:")
            for i, prob in enumerate(predictions[0]):
                st.write(f"- {weather_labels[i]}: {prob * 100:.2f}%")

        except Exception as e:
            st.error(f"Prediction error: {e}")

    else:
        st.warning("⚠️ Model could not be loaded. Please check your model file path.")
