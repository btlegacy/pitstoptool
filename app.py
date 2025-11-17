import streamlit as st
import tempfile
from video_analyzer import analyze_video

st.set_page_config(page_title="Pit Stop Analyzer", layout="wide")

st.title("🏎️ Pit Stop Analysis Tool")

st.write(
    "Upload a video of a pit stop to analyze the duration the car is stationary."
)

uploaded_file = st.file_uploader(
    "Choose a video file", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    st.video(uploaded_file)

    # Use a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.getvalue())
        video_path = tfile.name

        if st.button("Analyze Pit Stop"):
            with st.spinner("Analyzing video... This may take a few moments."):
                try:
                    stationary_time = analyze_video(video_path)
                    st.success(f"**Pit stop duration (car stationary): {stationary_time:.2f} seconds**")
                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
