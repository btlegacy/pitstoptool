import streamlit as st
import tempfile
import os
import cv2
import numpy as np
from video_analyzer import analyze_video, analyze_video_with_debug

st.set_page_config(page_title="Pit Stop Analyzer", layout="wide")

st.title("🏎️ Pit Stop Analysis Tool")

st.write(
    "Upload a video of a pit stop to analyze the duration the car is stationary."
)

uploaded_file = st.file_uploader(
    "Choose a video file", type=["mp4", "mov", "avi"]
)

# --- ROI Controls in Sidebar ---
st.sidebar.header("ROI & Analysis Settings")

st.sidebar.info("Adjust the sliders to draw a box around the car to isolate it from pit crew motion.")

# New default ROI, moved up
roi_top = st.sidebar.slider("ROI Top", 0.0, 1.0, 0.15, 0.01)
roi_bottom = st.sidebar.slider("ROI Bottom", 0.0, 1.0, 0.45, 0.01)
roi_left = st.sidebar.slider("ROI Left", 0.0, 1.0, 0.25, 0.01)
roi_right = st.sidebar.slider("ROI Right", 0.0, 1.0, 0.75, 0.01)

roi_percentage = [roi_top, roi_bottom, roi_left, roi_right]

generate_debug_video = st.sidebar.checkbox("Generate debug video", help="This will create a video showing the analysis process. It will take longer.")


if uploaded_file is not None:
    # Use a temporary file to store the uploaded video
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.getvalue())
        video_path = tfile.name
    
    # --- Display ROI Preview ---
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Video Preview")
        st.video(uploaded_file)
        
    with col2:
        st.subheader("ROI Preview on First Frame")
        
        # Get the first frame
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_height, frame_width, _ = frame.shape
            
            # Calculate ROI coordinates in pixels
            roi_coords = [
                int(frame_height * roi_top),
                int(frame_height * roi_bottom),
                int(frame_width * roi_left),
                int(frame_width * roi_right),
            ]

            # Draw the ROI rectangle on the frame
            preview_frame = frame.copy()
            cv2.rectangle(preview_frame, (roi_coords[2], roi_coords[0]), (roi_coords[3], roi_coords[1]), (0, 255, 0), 2)
            
            st.image(preview_frame, channels="BGR", use_column_width=True)
        else:
            st.warning("Could not read the first frame of the video.")


    if st.button("Analyze Pit Stop"):
        with st.spinner("Analyzing video... This may take a few moments."):
            try:
                if generate_debug_video:
                    debug_output_path = os.path.join(tempfile.gettempdir(), "debug_video.mp4")
                    stationary_time = analyze_video_with_debug(video_path, debug_output_path, roi_percentage)
                    
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
                    stationary_time = analyze_video(video_path, roi_percentage)
                    st.success(f"**Pit stop duration (car stationary): {stationary_time:.2f} seconds**")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
