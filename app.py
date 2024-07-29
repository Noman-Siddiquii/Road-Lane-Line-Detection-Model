import streamlit as st
from PIL import Image
import numpy as np
import cv2  # Assuming you're using OpenCV for image processing
from tensorflow import keras  # Import keras from tensorflow

# Function to preprocess the image (replace with your logic)
def preprocess_image(image):
    # Convert image to RGB (if needed)
    # Resize the image to your model's input size
    image_array = np.array(image)
    image_resized = cv2.resize(image_array, (256, 256))  # Adjust to your model's expected input size
    return image_resized

# Function to plot the predicted image (adapted for Streamlit)
def plot_predicted_image(model, image):
    
    # Preprocess the input image
    preprocessed_image = preprocess_image(image)

    # Make prediction
    predicted_image = model.predict(np.expand_dims(preprocessed_image, axis=0))[0]

    # Assuming the model output is normalized, scale it back to 0-255 range
    grayscale_image = np.average(predicted_image, axis=-1)

    return grayscale_image

# Load your Keras model (assuming you have it saved as 'save_at_6.keras')
loaded_model = keras.models.load_model('save_at_6.keras')

# Streamlit app
st.title("Image-to-Image Prediction App")

uploaded_file = st.file_uploader("Choose an image:", type=["jpg", "png"])

if uploaded_file is not None:
    # Read the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image")

    # Make prediction
    predicted_image = plot_predicted_image(loaded_model, image)

    # Display the predicted image
    st.header("Predicted Image")
    st.image(predicted_image, caption="Predicted Output")
