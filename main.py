import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image

# Load the trained model
model = load_model("_model.h5")  # Change to your actual model file

# Class labels (Change according to your dataset)
class_labels = ["Battery", "Biological", "Brown Glass", "Cardboard", "Clothes", 
                "Green Glass", "Metal", "Paper", "Plastic", "Shoes", "Trash", "White Glass"]

# Streamlit UI
st.title("Waste Classification App 🗑️")
st.write("Upload an image to classify it into one of the waste categories.")

# Upload Image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image_data = Image.open(uploaded_file)
    st.image(image_data, caption="Uploaded Image", use_column_width=True)

    # Preprocess image for prediction
    img = image_data.resize((224, 224))  # Resize to match model input
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0  # Normalize

    # Make Prediction
    prediction = model.predict(img_array)
    predicted_index = np.argmax(prediction)  # Get highest confidence class
    predicted_class = class_labels[predicted_index]
    confidence = prediction[0][predicted_index] * 100  # Convert to percentage

    # Show Result
    st.write("### 🏷️ Predicted Class:", predicted_class)
    st.write(f"### 🎯 Confidence: {confidence:.2f}%")

    # Show all confidence levels in a bar chart
    st.subheader("Confidence Levels")
    confidence_dict = {class_labels[i]: float(prediction[0][i]) for i in range(len(class_labels))}
    st.bar_chart(confidence_dict)
