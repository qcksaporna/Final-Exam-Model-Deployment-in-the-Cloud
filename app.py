import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import pandas as pd

# Load the model
model = tf.keras.models.load_model('finalmodel')

# Class labels
class_names = ['Cloudy', 'Rain', 'Shine', 'Sunrise']

st.title("**FINAL SKILL EXAM:** Deep Learning Weather Image Classifier Using CNN")
st.markdown("Name: Rick Melvhin R. Alcoseba")
st.markdown("Section: CPE32S2")
st.markdown("This program is a deep learning-based weather image classifier that uses a Convolutional Neural Network (CNN) to predict weather conditions from images. Users can upload any weather-related image, and the model will classify it as Cloudy, Rain, Shine, or Sunrise.")
st.markdown("-----------------------------------------------------------------------------------------------------")
# Image uploader
uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    img = Image.open(uploaded_file).convert('RGB').resize((224, 224))
    st.image(img, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    try:
        # Get prediction from SavedModel
        infer = model.signatures["serving_default"]
        input_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
        output = infer(input_tensor)
        prediction = list(output.values())[0].numpy()[0]

        # Convert predictions to percent format
        prediction_percent = [f"{p * 100:.2f}%" for p in prediction]

        # Display predictions in a labeled table
        df = pd.DataFrame([prediction_percent], columns=class_names)
        st.markdown("**Prediction Confidence:**")
        st.table(df)

        # Show final result
        predicted_class = class_names[np.argmax(prediction)]
        confidence = np.max(prediction) * 100
        st.success(f"**Prediction:** {predicted_class} ({confidence:.2f}% confidence)")

    except Exception as e:
        st.error(f"Prediction failed:\n\n{e}")
