
import json
from pathlib import Path

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

# ==========================================
# PROJECT PATHS
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "cat_dog_model.keras"
INFO_PATH = BASE_DIR / "models" / "model_info.json"

# ==========================================
# PAGE CONFIGURATION
# ==========================================

st.set_page_config(
    page_title="PawVision AI",
    page_icon="🐾",
    layout="centered"
)

# ==========================================
# PREMIUM DARK THEME
# ==========================================

st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

.stApp {
    background:
        radial-gradient(ellipse at 15% 5%, rgba(111, 66, 193, 0.19), transparent 35%),
        radial-gradient(ellipse at 90% 35%, rgba(37, 99, 235, 0.13), transparent 35%),
        #080b16;
    color: #f8fafc;
    font-family: 'Inter', sans-serif;
}

.block-container {
    max-width: 780px;
    padding-top: 3rem;
    padding-bottom: 3rem;
}

header[data-testid="stHeader"] {
    background: transparent;
}

h1, h2, h3, p, label {
    color: #f8fafc;
}

.hero {
    text-align: center;
    padding: 20px 0 32px 0;
}

.logo {
    display: inline-block;
    font-size: 14px;
    font-weight: 700;
    letter-spacing: 3px;
    color: #c4b5fd;
    background: rgba(139, 92, 246, 0.12);
    border: 1px solid rgba(167, 139, 250, 0.25);
    padding: 9px 18px;
    border-radius: 30px;
    margin-bottom: 22px;
}

.hero-title {
    font-size: 46px;
    font-weight: 800;
    letter-spacing: -2px;
    line-height: 1.15;
    color: #ffffff;
    margin-bottom: 14px;
}

.gradient-text {
    background: linear-gradient(90deg, #a78bfa, #60a5fa);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-description {
    color: #94a3b8;
    font-size: 15px;
    line-height: 1.8;
    max-width: 530px;
    margin: auto;
}

.section-heading {
    color: #e2e8f0;
    font-size: 18px;
    font-weight: 700;
    margin: 25px 0 12px 0;
}

.upload-card {
    background: rgba(22, 27, 48, 0.75);
    border: 1px solid rgba(139, 92, 246, 0.28);
    border-radius: 20px;
    padding: 22px;
    box-shadow: 0 10px 40px rgba(0,0,0,0.2);
}

[data-testid="stFileUploader"] {
    background: rgba(15, 20, 38, 0.8);
    border: 1px dashed #6551a5;
    border-radius: 14px;
    padding: 16px;
}

[data-testid="stFileUploader"] section {
    background: transparent;
}

[data-testid="stFileUploader"] label,
[data-testid="stFileUploader"] small,
[data-testid="stFileUploader"] span {
    color: #cbd5e1 !important;
}

[data-testid="stFileUploader"] button {
    background: #31245d;
    color: white;
    border: 1px solid #7055bc;
    border-radius: 9px;
}

.stButton > button {
    background: linear-gradient(100deg, #7c3aed, #4f46e5);
    color: white;
    border: none;
    border-radius: 12px;
    height: 52px;
    font-size: 16px;
    font-weight: 700;
    box-shadow: 0 5px 22px rgba(124, 58, 237, 0.28);
    transition: all 0.25s ease;
}

.stButton > button:hover {
    background: linear-gradient(100deg, #8b5cf6, #6366f1);
    border: none;
    color: white;
    transform: translateY(-2px);
    box-shadow: 0 8px 28px rgba(124, 58, 237, 0.4);
}

[data-testid="stImage"] {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid #303653;
}

.result-card {
    background: linear-gradient(135deg, rgba(91, 33, 182, 0.22), rgba(30, 64, 175, 0.18));
    border: 1px solid rgba(167, 139, 250, 0.4);
    border-radius: 20px;
    padding: 25px;
    text-align: center;
    margin-top: 25px;
    box-shadow: 0 10px 35px rgba(79, 70, 229, 0.12);
}

.result-label {
    font-size: 12px;
    font-weight: 700;
    letter-spacing: 3px;
    color: #a5b4fc;
}

.result-value {
    font-size: 42px;
    font-weight: 800;
    color: #ffffff;
    margin-top: 12px;
}

.confidence-card {
    background: rgba(22, 27, 48, 0.8);
    border: 1px solid #303653;
    border-radius: 16px;
    padding: 20px;
    margin-top: 18px;
}

[data-testid="stMetricLabel"] {
    color: #a5b4fc !important;
}

[data-testid="stMetricValue"] {
    color: #c4b5fd !important;
    font-weight: 800;
}

[data-testid="stProgressBar"] > div {
    background: #252b43;
}

[data-testid="stProgressBar"] > div > div {
    background: linear-gradient(90deg, #7c3aed, #60a5fa);
}

[data-testid="stExpander"] {
    background: rgba(22, 27, 48, 0.8);
    border: 1px solid #303653;
    border-radius: 12px;
}

[data-testid="stExpander"] summary,
[data-testid="stExpander"] summary p {
    color: #e2e8f0 !important;
}

.stAlert {
    border-radius: 12px;
}

.footer {
    text-align: center;
    color: #64748b;
    font-size: 12px;
    padding-top: 38px;
    letter-spacing: 1px;
}

hr {
    border-color: #252b43;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# HEADER
# ==========================================

st.markdown("""
<div class="hero">
    <div class="logo">✦ CAT VS DOG</div>
    <div class="hero-title">
     <span class="gradient-text"></span>
    </div>
    <div class="hero-description">
        Intelligent image recognition that identifies cats
        and dogs using deep learning.
        Upload an image and discover.
    </div>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ==========================================
# LOAD MODEL
# ==========================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

if model is None:
    st.error("Trained model not found. Check your models folder.")
    st.stop()

# ==========================================
# IMAGE UPLOAD
# ==========================================

st.markdown(
    '<div class="section-heading">01 &nbsp; / &nbsp; Upload Image</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="upload-card">', unsafe_allow_html=True)

image_file = st.file_uploader(
    "Choose your image",
    type=["jpg", "jpeg", "png", "webp"]
)

st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# IMAGE PROCESSING
# ==========================================

if image_file is not None:

    try:
        image = Image.open(image_file).convert("RGB")

        st.markdown(
            '<div class="section-heading">02 &nbsp; / &nbsp; Preview</div>',
            unsafe_allow_html=True
        )

        st.image(
            image,
            caption="Uploaded image",
            use_container_width=True
        )

        info = {}

        if INFO_PATH.exists():
            info = json.loads(
                INFO_PATH.read_text(encoding="utf-8")
            )

        image_size = int(info.get("image_size", 160))
        threshold = float(info.get("threshold", 0.5))

        resized = image.resize((image_size, image_size))

        arr = np.asarray(resized, dtype=np.float32) / 255.0

        # MobileNetV2 preprocessing
        arr = arr * 2.0 - 1.0

        arr = np.expand_dims(arr, axis=0)

        # ==========================================
        # CLASSIFICATION
        # ==========================================

        st.markdown(
            '<div class="section-heading">03 &nbsp; / &nbsp;  Analysis</div>',
            unsafe_allow_html=True
        )

        if st.button(
            "✦  ANALYZE IMAGE",
            type="primary",
            use_container_width=True
        ):

            probability_dog = float(
                model.predict(arr, verbose=0)[0][0]
            )

            predicted = (
                "Dog" if probability_dog >= threshold else "Cat"
            )

            confidence = (
                probability_dog
                if predicted == "Dog"
                else 1.0 - probability_dog
            )

            emoji = "🐶" if predicted == "Dog" else "🐱"

            # ==========================================
            # RESULT
            # ==========================================

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        ✦ &nbsp;  PREDICTION
                    </div>
                    <div class="result-value">
                        {emoji} &nbsp; {predicted}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="confidence-card">',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-heading">Prediction Confidence</div>',
                unsafe_allow_html=True
            )

            st.metric(
                "Model Confidence",
                f"{confidence * 100:.2f}%"
            )

            st.progress(float(confidence))

            st.markdown('</div>', unsafe_allow_html=True)

            # ==========================================
            # CLASS PROBABILITIES
            # ==========================================

            with st.expander("View detailed class probabilities"):

                st.write(
                    f"🐱 Cat: {(1-probability_dog)*100:.2f}%"
                )

                st.write(
                    f"🐶 Dog: {probability_dog*100:.2f}%"
                )

            if confidence < 0.60:
                st.warning(
                    "The AI is not very certain about this image. "
                    "Try uploading a clearer image."
                )

    except Exception as exc:

        st.error(
            f"Could not process this image. "
            f"Please upload a valid image. Details: {exc}"
        )

# ==========================================
# FOOTER
# ==========================================

st.markdown("""
<div class="footer">
    PAWVISION AI &nbsp; • &nbsp; POWERED BY TENSORFLOW
    AND MOBILENETV2
</div>
""", unsafe_allow_html=True)
