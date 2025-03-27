import numpy as np
import tensorflow
import streamlit as st
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model
from tensorflow import expand_dims
from tensorflow.nn import sigmoid
from PIL import Image

def prediction():
    score = ""
    class_names = ['Aloevera', 'Amla', 'Bamboo', 'Beans', 'Betel', 'Coffee', 'Coriender', 'Curry', 'Drumstick', 'Eucalyptus', 'Ginger']
    # Load the model
    model = load_model('./model.keras')
    uploaded_file = st.file_uploader("Choose a image [Format - JPG/PNG]", type=["jpg", "png"])
    if uploaded_file:
        st.success("File uploaded successfully!")
        st.image(uploaded_file, use_container_width=True)
        image = Image.open(uploaded_file).convert("RGB")  # Ensure 3 channels (RGB)
        image = image.resize((299, 299))  # Resize to match model input shape
        image_array = img_to_array(image) / 255.0  # Normalize pixel values
        image_array = np.expand_dims(image_array, axis=0) 
        image_array = img_to_array(image)
        image_array = expand_dims(image_array, 0)
        predictions = model.predict(image_array)
        score = sigmoid(predictions[0])
        value = 100 * np.max(score)
        # Predict along with it confidence
        if(predictions.size > 0 and value > 60):
            st.write("This image most likely belongs to {} with a {:.2f} percent confidence."
                .format(class_names[np.argmax(score)], value ))
        else:
            st.error("Prediction failed. The model did not return any output.")
