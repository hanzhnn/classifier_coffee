import streamlit as st
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tempfile
import os
import requests
from tensorflow.keras.models import Model
from PIL import Image

# === Page Config (MUST be first Streamlit command) ===
st.set_page_config(page_title="Coffee Leaf Classifier", layout="centered", initial_sidebar_state="auto")

# === Load Model from HuggingFace ===
@st.cache_resource
def load_model_from_url():
    model_url = "https://huggingface.co/hanzhnn/coffee-leaf-classifier/resolve/main/coffee_leaf_model.h5"
    model_path = os.path.join(tempfile.gettempdir(), "coffee_leaf_model.h5")

    if not os.path.exists(model_path):
        with open(model_path, 'wb') as f:
            response = requests.get(model_url)
            f.write(response.content)

    model = tf.keras.models.load_model(model_path, compile=False)
    _ = model(tf.zeros((1, 224, 224, 3)))  # Build model
    return model

model = load_model_from_url()
class_names = ['Healthy', 'Rust', 'Phoma', 'Cercospora', 'Miner']

# === Grad-CAM Function ===
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

# === Grad-CAM Overlay Function ===
def display_gradcam(img, heatmap, alpha=0.4):
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    colormap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(img, 1 - alpha, colormap, alpha, 0)
    return superimposed

# === Custom Styles ===
st.markdown("""
    <style>
        body {
            background-color: #f6f1eb;
            color: #4b2e2e;
        }
        .css-1d391kg { color: #4b2e2e; }
        .stProgress > div > div > div > div {
            background-color: #8b5e3c;
        }
    </style>
""", unsafe_allow_html=True)

# === App Interface ===
st.title("☕ Coffee Leaf Disease Classifier")
st.write("Upload a coffee leaf image to detect possible diseases using a trained CNN model.")

uploaded_file = st.file_uploader("Upload an image of a coffee leaf", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Leaf", use_column_width=True)

    img_array = tf.image.resize(np.array(image), (224, 224)) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    top_pred_index = np.argmax(prediction[0])
    confidence = float(prediction[0][top_pred_index])
    class_name = class_names[top_pred_index]

    st.subheader("Prediction Result")
    st.markdown(f"<h3 style='color:#4b2e2e'>{class_name}</h3>", unsafe_allow_html=True)
    st.progress(float(confidence))
    st.write(f"Confidence: {confidence:.2f}")

    st.subheader("Model Focus Area (Grad-CAM)")
    try:
        heatmap = make_gradcam_heatmap(img_array, model, "conv2d_2", pred_index=top_pred_index)
        original_img = np.array(image.resize((224, 224)))
        superimposed = display_gradcam(original_img, heatmap)
        st.image(superimposed, caption="Model focus region", use_column_width=True)
    except Exception as e:
        st.warning("Grad-CAM visualization failed. The selected layer may not be compatible.")
        st.text(f"Error: {e}")


