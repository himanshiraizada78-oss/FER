import streamlit as st
import av
import cv2
from deepface import DeepFace
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

st.title("😃 Live Face Expression Analyzer")

# session state for camera
if "camera_on" not in st.session_state:
    st.session_state.camera_on = False

# buttons
col1, col2 = st.columns(2)

with col1:
    if st.button("▶ Start Camera"):
        st.session_state.camera_on = True

with col2:
    if st.button("⏹ Stop Camera"):
        st.session_state.camera_on = False


class EmotionDetector(VideoTransformerBase):

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")

        try:
            result = DeepFace.analyze(
                img,
                actions=['emotion'],
                enforce_detection=False
            )

            emotion = result[0]['dominant_emotion']

            cv2.putText(
                img,
                emotion,
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

        except:
            pass

        return img


# start webcam only if button pressed
if st.session_state.camera_on:

    webrtc_streamer(
        key="emotion",
        video_transformer_factory=EmotionDetector
    )
