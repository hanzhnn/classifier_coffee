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

# Set Streamlit page config
st.set_page_config(
    page_title="Coffee Leaf Disease Classifier ☕🌿",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Add custom CSS for improved readability with brown aesthetic
st.markdown("""
    <style>
        html, body, [class*="st-"] {
            background-color: #f5f0e6;
            color: #3e2723;
            font-family: 'Segoe UI', sans-serif;
        }
        h1, h2, h3, .stMarkdown, .stText, .stSubheader, .stCaption {
            color: #3e2723;
        }
        .dark-mode body, .dark-mode [class*="st-"] {
            background-color: #1e1e1e !important;
            color: #f5f0e6 !important;
        }
        .dark-mode h1, .dark-mode h2, .dark-mode h3, .dark-mode .stMarkdown,
        .dark-mode .stText, .dark-mode .stSubheader, .dark-mode .stCaption {
            color: #f5f0e6 !important;
        }
    </style>
""", unsafe_allow_html=True)




