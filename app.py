import streamlit as st
import cv2
import numpy as np
from deepface import DeepFace
from PIL import Image

st.set_page_config(page_title="Face Expression Analyzer", layout="centered")

st.title("😊 AI Face Expression Analyzer")
st.write("Detect your emotions using your webcam")

# Session state for camera
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# Buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("Start Camera"):
        st.session_state.camera_on = True

with col2:
    if st.button("Stop Camera"):
        st.session_state.camera_on = False


# Webcam
if st.session_state.camera_on:

    img_file = st.camera_input("Take a picture")

    if img_file is not None:

        # Convert image
        image = Image.open(img_file)
        img = np.array(image)

        st.image(img, caption="Captured Image")

        try:
            result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)

            emotion = result[0]['dominant_emotion']
            emotions = result[0]['emotion']

            st.success(f"Dominant Emotion: {emotion}")

            st.subheader("Emotion Probabilities")

            for key, value in emotions.items():
                st.write(f"{key} : {round(value,2)} %")

        except Exception as e:
            st.error("Face not detected clearly. Try again.")
