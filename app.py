import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.cm as cm

# === Page config (must be first) ===
st.set_page_config(page_title="Coffee Leaf Classifier", layout="centered")

# === Custom CSS styling ===
st.markdown("""
    <style>
    body {
        background-color: #f7f3f0;
        color: #4b2e2e;
    }
    .stApp {
        background-color: #f7f3f0;
        font-family: 'Segoe UI', sans-serif;
    }
    h1, h2, h3, h4 {
        color: #5b3c2c;
    }
    .st-bb {
        background-color: #f7f3f0 !important;
    }
    .stProgress > div > div > div > div {
        background-color: #8b5e3c;
    }
    </style>
""", unsafe_allow_html=True)

# === Load model ===
model_path = os.path.join(os.getcwd(), 'model', 'coffee_leaf_model.keras')
model = tf.keras.models.load_model(model_path)
class_names = list(model.class_names) if hasattr(model, 'class_names') else ['Healthy', 'Rust', 'Phoma', 'Cercospora', 'Miner']

st.title("☕🌿 Coffee Leaf Disease Classifier")
st.markdown("Upload a clear image of a coffee leaf below. The model will identify possible diseases and highlight the leaf regions it focused on using Grad-CAM.")

uploaded_file = st.file_uploader("📤 Upload a coffee leaf image", type=["jpg", "jpeg", "png"])

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    img_tensor = tf.convert_to_tensor(img_array)

    conv_layer = model.get_layer(last_conv_layer_name)

    grad_model = tf.keras.models.Model(
        inputs=model.input,
        outputs=[conv_layer.output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_tensor)
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

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    preds = model.predict(img_array)
    top_pred_index = np.argmax(preds[0])
    predicted_label = class_names[top_pred_index]
    confidence = preds[0][top_pred_index]

    st.subheader(f"🧠 Prediction: **{predicted_label}** ({confidence*100:.2f}% confidence)")

    st.text("📊 Class Probabilities:")
    for i, prob in enumerate(preds[0]):
        st.markdown(f"<span style='color:#5b3c2c'>{class_names[i]}:</span> **{prob*100:.2f}%**", unsafe_allow_html=True)

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, "last_conv", pred_index=top_pred_index)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    superimposed = heatmap_colored * 0.4 + np.array(img_resized)/255.0
    st.subheader("🔍 Grad-CAM Heatmap")
    st.image(np.clip(superimposed, 0.0, 1.0), caption="Model focus region", use_column_width=True)


