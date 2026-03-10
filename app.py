import streamlit as st
import numpy as np
import pandas as pd
from deepface import DeepFace
from PIL import Image

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="AI Face Expression Analyzer",
    page_icon="😊",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

.title {
    text-align:center;
    font-size:45px;
    font-weight:bold;
    color:#00FFFF;
}

.subtitle {
    text-align:center;
    font-size:18px;
    color:#CCCCCC;
}

.result-box {
    padding:20px;
    border-radius:10px;
    background-color:#1f1f1f;
}

</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<p class="title">😊 AI Face Expression Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect human emotions using Artificial Intelligence</p>', unsafe_allow_html=True)

st.write("")

# ---------------- HERO SECTION ----------------
st.image(
"https://cdn-icons-png.flaticon.com/512/4140/4140048.png",
width=120
)

st.markdown("""
### 🤖 AI Emotion Detection System

This application uses **Deep Learning** to analyze facial expressions  
and detect human emotions.

### Features
- Webcam emotion detection
- Emotion probability dashboard
- Interactive AI interface
""")

st.write("")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Control Panel")

camera_on = st.sidebar.toggle("Start Camera")

st.sidebar.markdown("---")

st.sidebar.markdown("### 📘 Project Details")

st.sidebar.write("""
**Project:** Face Expression Analyzer  

**Technologies Used:**
- Streamlit
- DeepFace
- TensorFlow
- OpenCV
- NumPy

**Model:**
CNN-based facial emotion recognition model.

**Emotions Detected:**
Happy, Sad, Angry, Fear, Surprise, Neutral
""")

# ---------------- PAGE LAYOUT ----------------
col1, col2 = st.columns([2,1])

# ---------------- CAMERA SECTION ----------------
with col1:

    st.subheader("📷 Camera")

    if camera_on:
        img_file = st.camera_input("Take a Picture")
    else:
        st.info("Turn on the camera from the sidebar")
        img_file = None

# ---------------- EMOTION DETECTION ----------------
if img_file is not None:

    image = Image.open(img_file)
    img = np.array(image)

    with st.spinner("Analyzing Emotion..."):

        result = DeepFace.analyze(
            img,
            actions=['emotion'],
            enforce_detection=False
        )

    if isinstance(result, list):
        result = result[0]

    emotion = result['dominant_emotion']
    emotions = result['emotion']

# ---------------- RESULT SECTION ----------------
    with col2:

        st.subheader("📊 Emotion Result")

        st.success(f"Dominant Emotion: {emotion.upper()}")

        st.write("")

        # Emotion Chart
        df = pd.DataFrame(
            emotions.items(),
            columns=["Emotion","Score"]
        )

        st.bar_chart(df.set_index("Emotion"))

        st.write("")

        # Emotion Progress Bars
        st.markdown("### 📈 Emotion Confidence")

        for emotion_name, score in emotions.items():
            st.write(f"{emotion_name} : {round(score,2)} %")
            st.progress(int(score))

# ---------------- AI EXPLANATION ----------------
st.write("")

with st.expander("📘 How this AI model works"):
    st.write("""
This system uses a pretrained **Convolutional Neural Network (CNN)** model
from the DeepFace framework to detect facial emotions.

### Steps:

1. Capture image from webcam
2. Detect face in the image
3. Extract facial features
4. Apply deep learning model
5. Predict human emotion

The model analyzes facial muscles and expressions to classify emotions.
""")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & Deep Learning")
