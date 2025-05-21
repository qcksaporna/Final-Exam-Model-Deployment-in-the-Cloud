import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Define the weather labels
weather_labels = {0: 'Cloudy', 1: 'Rain', 2: 'Shine', 3: 'Sunrise'}

@st.cache_resource
def load_my_model():
    """
    Loads the Keras model from the specified path.
    Uses compile=False to prevent issues with model loading for inference.
    """
    # IMPORTANT: Ensure this path and extension (.keras or .h5) match
    # how you saved your model using model.save()
    model_path = 'final_modelxd' # Change to 'final_modelxd.h5' if you saved it as .h5
    try:
        # Crucial: Use compile=False to avoid '_UserObject' object has no attribute 'predict'
        # This tells Keras to load only the model architecture and weights, not the optimizer state.
        model = load_model(model_path, compile=False)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}. Please ensure the model file '{model_path}' exists in the correct directory and was saved using `model.save()`.")
        return None

# Load the model once when the app starts
model = load_my_model()

st.title("Weather Image Classifier")
st.write("Upload an image and the model will predict the weather.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_container_width=True)

    if model is not None:
        # Preprocess the image for prediction
        img = image.resize((128, 128)) # Resize to model's expected input
        if img.mode != 'RGB':
            img = img.convert('RGB') # Ensure 3 channels (RGB)
        
        img_array = np.array(img) / 255.0 # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0) # Add batch dimension (1, 128, 128, 3)

        try:
            # Perform prediction
            prediction = model.predict(img_array)
            predicted_class = np.argmax(prediction)
            predicted_label = weather_labels[predicted_class]
            
            # Display results
            st.write(f"Prediction: **{predicted_label}**")
            st.write(f"Confidence: **{np.max(prediction) * 100:.2f}%**")

            st.subheader("All Class Probabilities:")
            for i, prob in enumerate(prediction[0]):
                st.write(f"- {weather_labels[i]}: {prob*100:.2f}%")

        except Exception as e:
            st.error(f"An unexpected error occurred during prediction: {e}")
            st.error("This might indicate an issue with the model's input shape or internal structure after loading.")
    else:
        st.warning("The model could not be loaded. Please check the model file path and ensure it's a valid Keras model saved with `model.save()`.")
