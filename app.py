import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import matplotlib.cm as cm

# === Page Configuration ===
st.set_page_config(
    page_title="Coffee Leaf Classifier ☕🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# === Custom CSS Styling for Aesthetics ===
def apply_custom_style(dark_mode=False):
    primary = "#4E342E"  # Dark brown
    background = "#D7CCC8" if not dark_mode else "#2E2E2E"
    font_color = "#2E2E2E" if not dark_mode else "#FFFFFF"
    box_color = "#FFF3E0" if not dark_mode else "#424242"
    st.markdown(f"""
        <style>
        body {{
            background-color: {background};
        }}
        .reportview-container {{
            background-color: {background};
            color: {font_color};
        }}
        .block-container {{
            max-width: 90%;
            padding: 2rem;
        }}
        .stButton button {{
            background-color: {primary};
            color: white;
            border-radius: 8px;
            padding: 0.5rem 1.2rem;
            transition: all 0.3s ease-in-out;
        }}
        .stButton button:hover {{
            background-color: #6D4C41;
        }}
        .stFileUploader > div {{
            background-color: {box_color};
            padding: 1rem;
            border-radius: 10px;
        }}
        .title-text {{
            text-align: center;
            font-size: 2.5rem;
            font-weight: bold;
            color: {primary};
        }}
        .subtitle-text {{
            text-align: center;
            font-size: 1.2rem;
            color: {font_color};
            margin-bottom: 1.5rem;
        }}
        </style>
    """, unsafe_allow_html=True)

# === Toggle Dark Mode ===
dark_mode = st.sidebar.checkbox("🌙 Dark Mode")
apply_custom_style(dark_mode)

# === Load model ===
model_path = os.path.join(os.getcwd(), 'model', 'coffee_leaf_model.keras')
model = tf.keras.models.load_model(model_path)
class_names = list(model.class_names) if hasattr(model, 'class_names') else ['Healthy', 'Rust', 'Phoma', 'Cercospora', 'Miner']

# === Title Section ===
st.markdown("<div class='title-text'>Coffee Leaf Disease Classifier ☕🌿</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle-text'>Upload a coffee leaf image to detect potential diseases using AI</div>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])


# === Grad-CAM Utility ===
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


# === Prediction Process ===
if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    img_resized = image.resize((224, 224))
    img_array = np.expand_dims(np.array(img_resized) / 255.0, axis=0)

    with st.spinner("🔎 Analyzing the leaf..."):
        preds = model.predict(img_array)
        top_pred_index = np.argmax(preds[0])
        predicted_label = class_names[top_pred_index]
        confidence = float(preds[0][top_pred_index])

        st.success(f"🧠 **Prediction:** {predicted_label} ({confidence*100:.2f}%)")

        # Show all class probabilities
        st.subheader("📊 Prediction Probabilities")
        for i, prob in enumerate(preds[0]):
            st.write(f"**{class_names[i]}**: {prob*100:.2f}%")
            st.progress(float(prob))

        # Grad-CAM
        st.subheader("🔍 Grad-CAM Heatmap")
        heatmap = make_gradcam_heatmap(img_array, model, "conv2d_2", pred_index=top_pred_index)
        heatmap_resized = cv2.resize(heatmap, (224, 224))
        heatmap_colored = cm.jet(heatmap_resized)[:, :, :3]
        superimposed = heatmap_colored * 0.4 + np.array(img_resized)/255.0
        st.image(np.clip(superimposed, 0.0, 1.0), caption="Model attention on input", use_column_width=True)


