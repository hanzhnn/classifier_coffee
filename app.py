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

# Set Streamlit page config
st.set_page_config(
    page_title="Coffee Leaf Disease Classifier ☕🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add custom CSS for aesthetics
st.markdown("""
    <style>
        body {
            background-color: #dfcfaf;
            color: #3e2723;
        }
        .stApp {
            background-color: #dfcfaf;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3, .stMarkdown {
            color: #3e2723;
        }
        .css-18e3th9 {
            padding-top: 2rem;
        }
        .dark-mode {
            background-color: #584723 !important;
            color: #f5f0e6 !important;
        }
    </style>
""", unsafe_allow_html=True)

# Toggle dark mode
dark_mode = st.toggle("🌙 Dark Mode")
if dark_mode:
    st.markdown("<style>body, .stApp { background-color: #1e1e1e; color: #f5f0e6; }</style>", unsafe_allow_html=True)

# Animation
st.markdown("### 🚀 Upload a coffee leaf image to classify and visualize disease region")

# === Load model ===
model_path = os.path.join(os.getcwd(), 'model', 'coffee_leaf_model.keras')
model = tf.keras.models.load_model(model_path)
class_names = list(model.class_names) if hasattr(model, 'class_names') else ['Healthy', 'Rust', 'Phoma', 'Cercospora', 'Miner']

st.title("Coffee Leaf Disease Classifier ☕🌿")
uploaded_file = st.file_uploader("📂 Upload a coffee leaf image", type=["jpg", "jpeg", "png"])


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

    with st.spinner("🔍 Analyzing image..."):
        preds = model.predict(img_array)
        top_pred_index = np.argmax(preds[0])
        predicted_label = class_names[top_pred_index]
        confidence = preds[0][top_pred_index]

    st.success(f"🧠 Prediction: **{predicted_label}** ({confidence*100:.2f}% confidence)")
    st.text("All class probabilities:")
    for i, prob in enumerate(preds[0]):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")

    # Grad-CAM
    heatmap = make_gradcam_heatmap(img_array, model, "last_conv", pred_index=top_pred_index)
    heatmap_resized = cv2.resize(heatmap, (224, 224))
    heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
    superimposed = heatmap_colored * 0.4 + np.array(img_resized) / 255.0
    st.subheader("🔬 Grad-CAM Heatmap")
    st.image(np.clip(superimposed, 0.0, 1.0), caption="Model focus region", use_column_width=True)


