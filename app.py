import streamlit as st
import tempfile
import os
from video_analyzer import analyze_video, analyze_video_with_debug

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
    
    generate_debug_video = st.checkbox("Generate debug video", help="This will create a video showing the analysis process. It will take longer.")

    # Use a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.getvalue())
        video_path = tfile.name

        if st.button("Analyze Pit Stop"):
            with st.spinner("Analyzing video... This may take a few moments."):
                try:
                    if generate_debug_video:
                        debug_output_path = os.path.join(tempfile.gettempdir(), "debug_video.mp4")
                        stationary_time = analyze_video_with_debug(video_path, debug_output_path)
                        
                        st.success(f"**Pit stop duration (car stationary): {stationary_time:.2f} seconds**")
                        
                        st.subheader("Analysis Debug Video")
                        st.video(debug_output_path)

                        with open(debug_output_path, "rb") as file:
                            st.download_button(
                                label="Download Debug Video",
                                data=file,
                                file_name="pitstop_analysis_debug.mp4",
                                mime="video/mp4"
                            )

                    else:
                        stationary_time = analyze_video(video_path)
                        st.success(f"**Pit stop duration (car stationary): {stationary_time:.2f} seconds**")

                except Exception as e:
                    st.error(f"An error occurred during analysis: {e}")
