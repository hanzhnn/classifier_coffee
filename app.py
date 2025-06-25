import streamlit as st
import tensorflow as tf
from keras.models import load_model
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import tempfile
import os

# === Grad-CAM Function ===
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        [model.inputs],
        [model.get_layer(last_conv_layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_output = predictions[:, pred_index]

    grads = tape.gradient(class_output, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_gradcam(heatmap, original_img, alpha=0.4):
    heatmap = cv2.resize(heatmap, (original_img.shape[1], original_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(original_img, 1 - alpha, heatmap_color, alpha, 0)
    return superimposed_img

# === Model loading ===
@st.cache_resource
def load_model_from_url():
    model_url = "https://huggingface.co/hanzhnn/coffee-leaf-classifier/resolve/main/coffee_leaf_model.h5"
    model_path = "downloaded_model.h5"
    if not os.path.exists(model_path):
        with open(model_path, "wb") as f:
            response = tf.keras.utils.get_file(origin=model_url, fname=model_path)
    return tf.keras.models.load_model(model_path)

model = load_model_from_url()
class_names = ['Cercospora', 'Healthy', 'Miner', 'Phoma', 'Rust']

# === Streamlit UI ===
st.set_page_config(page_title="Coffee Leaf Classifier", layout="centered", initial_sidebar_state="auto")

# CSS Styling for brown theme
st.markdown("""
    <style>
    html, body, [class*="css"] {
        color: #4b2e2e;
        background-color: #f5f2f0;
    }
    .stButton>button {
        background-color: #8d6e63;
        color: white;
        border: none;
        padding: 0.5em 1em;
        border-radius: 4px;
    }
    </style>
""", unsafe_allow_html=True)

st.title("☕ Coffee Leaf Disease Classifier")
st.markdown("Upload a coffee leaf image to detect the type of disease using a deep learning model.")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    image_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(image_resized) / 255.0, axis=0)

    prediction = model.predict(img_array)[0]
    top_pred_index = np.argmax(prediction)
    top_class = class_names[top_pred_index]
    confidence = float(prediction[top_pred_index])

    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.markdown(f"### Prediction: **{top_class}**")
    st.progress(float(confidence))

    # Grad-CAM visualization
    st.markdown("#### 🔍 Grad-CAM Heatmap")
    heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer_name="conv2d_2", pred_index=top_pred_index)
    image_cv = np.array(image.resize((224, 224)))
    superimposed = overlay_gradcam(heatmap, image_cv)
    st.image(superimposed, caption="Model Focus Area (Grad-CAM)", use_column_width=True)

# Optional: Footer
st.markdown("<br><sub>Created by Han • Deployed on Streamlit</sub>", unsafe_allow_html=True)
