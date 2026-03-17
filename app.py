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

# ---------------- CACHE MODEL ----------------
@st.cache_resource
def load_model():
    return DeepFace.build_model("Emotion")

model = load_model()

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main {
    background-color: #0E1117;
}
.title {
    text-align: center;
    font-size: 45px;
    font-weight: bold;
    color: #00FFFF;
}
.subtitle {
    text-align:center;
    font-size:18px;
    color:#CCCCCC;
}
.result-box {
    padding:20px;
    border-radius:12px;
    background-color:#1f1f1f;
}
</style>
""", unsafe_allow_html=True)

# ---------------- TITLE ----------------
st.markdown('<p class="title">😊 AI Face Expression Analyzer</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Detect human emotions using AI</p>', unsafe_allow_html=True)

st.write("")

# ---------------- SIDEBAR ----------------
st.sidebar.title("⚙ Control Panel")
camera_on = st.sidebar.toggle("Start Camera")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📘 About Project")
st.sidebar.write("""
This AI application detects human emotions from facial expressions.

Technology used:
- Streamlit
- DeepFace
- Deep Learning (CNN)
""")

# ---------------- LAYOUT ----------------
col1, col2 = st.columns([2,1])

# ---------------- CAMERA ----------------
with col1:
    st.subheader("📷 Camera")

    if camera_on:
        img_file = st.camera_input("Take a Picture")
    else:
        st.info("Turn on the camera from sidebar")
        img_file = None

# ---------------- PROCESS IMAGE ----------------
if img_file is not None:

    image = Image.open(img_file)
    img = np.array(image)

    with st.spinner("Analyzing Emotion..."):
        try:
            result = DeepFace.analyze(
                img,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = result[0]['dominant_emotion']
            emotions = result[0]['emotion']

        except Exception as e:
            st.error("Error detecting face. Try another image.")
            st.stop()

# ---------------- RESULT ----------------
    with col2:

        st.subheader("📊 Emotion Result")

        st.success(f"Dominant Emotion: {emotion.upper()}")

        df = pd.DataFrame(
            emotions.items(),
            columns=["Emotion","Score"]
        )

        st.bar_chart(df.set_index("Emotion"))

        st.markdown("### Emotion Probabilities")

        for key, value in emotions.items():
            st.write(f"{key} : {round(value,2)} %")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made with ❤️ using Streamlit & Deep Learning")
