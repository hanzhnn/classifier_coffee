# Coffee Leaf Disease Classifier ☕🌿

This Streamlit app classifies coffee leaf diseases using a trained CNN model and visualizes model attention using Grad-CAM.

## Features
- Upload a coffee leaf image
- Predicts disease (e.g., Rust, Phoma)
- Visual explanation via Grad-CAM heatmap

## How to Run
1. Install dependencies from `requirements.txt`
2. Add the trained model to the `/model` folder (see below)
3. Run locally:
```bash
streamlit run app.py
```

## Deployment
This app is ready to deploy on [Streamlit Cloud](https://streamlit.io/cloud).

## Model
Place your trained model file here:
```
/model/coffee_leaf_model.keras
```

---

Made with ❤️ for coffee farmers and computer vision learners.
