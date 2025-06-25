import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
import numpy as np
import cv2
from PIL import Image
import tempfile
import os
import requests
import matplotlib.pyplot as plt

# === MUST BE FIRST STREAMLIT COMMAND ===
st.set_page_config(
    page_title="Coffee Leaf Classifier",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# === Custom CSS for brown theme and dark mode ===
st.markdown("""
    <style>
    body {
        background-color: #f7f3ee;
        color: #4e342e;
        font-family: 'Segoe UI', sans-serif;
    }
    .stButton>button {
        background-color: #795548;
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 6px;
    }
    .stProgress .st-bo {
        background-color: #795548 !important;
    }
    .css-1v0mbdj { background-color: #f7f3ee; }
    </style>
""", unsafe_allow_html=True)

# === Download and Load Model ===
@st.cache_resource
def load_model_from_url():
    model_url = "https://huggingface.co/hanzhnn/coffee-leaf-classifier/resolve/main/coffee_leaf_model.h5"
    model_path = os.path.join(tempfile.gettempdir(), "coffee_leaf_model.h5")

    if not os.path.exists(model_path):
        with open(model_path, 'wb') as f:
            response = requests.get(model_url)
            f.write(response.content)

    return tf.keras.models.load_model(model_path, compile=False)

model = load_model_from_url()
class_names = ['Cercospora', 'Healthy', 'Miner', 'Phoma', 'Rust']

# === Grad-CAM Heatmap ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = Model(
        inputs=[model.input],
        outputs=[model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    grads = tape.gradient(class_channel, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# === Image Preprocessing ===
def preprocess_image(img):
    img = img.resize((224, 224))
    img_array = tf.keras.utils.img_to_array(img) / 255.0
    return np.expand_dims(img_array, axis=0)

# === Streamlit UI ===
st.title("☕ Coffee Leaf Disease Classifier")
st.write("Upload an image of a coffee leaf to detect possible diseases.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    img_array = preprocess_image(image)
    predictions = model.predict(img_array)
    top_pred_index = np.argmax(predictions)
    predicted_class = class_names[top_pred_index]
    confidence = float(predictions[0][top_pred_index])

    st.subheader(f"Prediction: `{predicted_class}`")
    st.write(f"Confidence Score: **{confidence:.2f}**")
    st.progress(float(confidence))  # ensure type is float

    # === Grad-CAM visualization ===
    heatmap = make_gradcam_heatmap(img_array, model, "conv2d_2", pred_index=top_pred_index)

    # superimpose on image
    img = np.array(image.resize((224, 224)))
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    st.image(superimposed, caption="Model Focus (Grad-CAM)", use_column_width=True)

st.markdown("---")
st.caption("Developed for Applied AI Prototype – UMS Data Science Project © 2025")

