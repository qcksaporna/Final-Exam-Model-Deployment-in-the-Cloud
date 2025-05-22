import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model once (hardcoded folder name)
model = tf.keras.models.load_model('final_modelxd')

weather_labels = {0: 'Cloudy', 1: 'Rain', 2: 'Shine', 3: 'Sunrise'}

st.title("Weather Image Classifier")
st.write("Upload an image and the model will predict the weather.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    # Preprocess image
    img = image.resize((128, 128))
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)  # batch dimension

    try:
        prediction = model.predict(img_array)
        predicted_class = np.argmax(prediction)
        predicted_label = weather_labels[predicted_class]

        st.write(f"Prediction: **{predicted_label}**")
        st.write(f"Confidence: **{np.max(prediction)*100:.2f}%**")

        st.subheader("All Class Probabilities:")
        for i, prob in enumerate(prediction[0]):
            st.write(f"- {weather_labels[i]}: {prob*100:.2f}%")

    except Exception as e:
        st.error(f"Prediction error: {e}")
