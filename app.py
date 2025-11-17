import streamlit as st
import tempfile
import os
import cv2
from video_analyzer import analyze_video_with_debug

st.set_page_config(page_title="Pit Stop Analyzer", layout="wide")

st.title("Pit Stop Analysis Tool")

st.write(
    "Upload a video of a pit stop to analyze the duration the car is stationary."
)

# --- ROI Controls in Sidebar ---
st.sidebar.header("Analysis Settings")

with st.sidebar.expander("Car ROI Settings", expanded=True):
    st.info("Draw a box around the area where the car will be when stopped.")
    roi_top = st.slider("Car ROI Top", 0.0, 1.0, 0.15, 0.01)
    roi_bottom = st.slider("Car ROI Bottom", 0.0, 1.0, 0.45, 0.01)
    roi_left = st.slider("Car ROI Left", 0.0, 1.0, 0.25, 0.01)
    roi_right = st.slider("Car ROI Right", 0.0, 1.0, 0.75, 0.01)
    car_roi_percentage = [roi_top, roi_bottom, roi_left, roi_right]

with st.sidebar.expander("Signboard ROI Settings"):
    st.info("Draw a small box around the pit signboard that the car hits.")
    sign_roi_top = st.slider("Sign ROI Top", 0.0, 1.0, 0.20, 0.01)
    sign_roi_bottom = st.slider("Sign ROI Bottom", 0.0, 1.0, 0.40, 0.01)
    sign_roi_left = st.slider("Sign ROI Left", 0.0, 1.0, 0.65, 0.01)
    sign_roi_right = st.slider("Sign ROI Right", 0.0, 1.0, 0.75, 0.01)
    sign_roi_percentage = [sign_roi_top, sign_roi_bottom, sign_roi_left, sign_roi_right]

generate_debug_video = st.sidebar.checkbox("Generate debug video", help="This will create a video showing the analysis process. It will take longer.")

uploaded_file = st.file_uploader(
    "Choose a video file", type=["mp4", "mov", "avi"]
)

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tfile:
        tfile.write(uploaded_file.getvalue())
        video_path = tfile.name
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Video Preview")
        st.video(uploaded_file)
        
    with col2:
        st.subheader("ROI Preview on First Frame")
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()

        if ret:
            frame_height, frame_width, _ = frame.shape
            
            preview_frame = frame.copy()
            # Draw Car ROI
            car_coords = [int(frame_height * roi_top), int(frame_height * roi_bottom), int(frame_width * roi_left), int(frame_width * roi_right)]
            cv2.rectangle(preview_frame, (car_coords[2], car_coords[0]), (car_coords[3], car_coords[1]), (0, 255, 0), 2)
            cv2.putText(preview_frame, "Car ROI", (car_coords[2], car_coords[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Draw Signboard ROI
            sign_coords = [int(frame_height * sign_roi_top), int(frame_height * sign_roi_bottom), int(frame_width * sign_roi_left), int(frame_width * sign_roi_right)]
            cv2.rectangle(preview_frame, (sign_coords[2], sign_coords[0]), (sign_coords[3], sign_coords[1]), (255, 0, 0), 2)
            cv2.putText(preview_frame, "Sign ROI", (sign_coords[2], sign_coords[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            
            st.image(preview_frame, channels="BGR", use_column_width=True)
        else:
            st.warning("Could not read the first frame of the video.")

    if st.button("Analyze Pit Stop"):
        with st.spinner("Analyzing video... This will take a few moments."):
            try:
                # We only need one analysis function now, as the debug info is more critical.
                debug_output_path = os.path.join(tempfile.gettempdir(), "debug_video.mp4")
                stationary_time = analyze_video_with_debug(video_path, debug_output_path, car_roi_percentage, sign_roi_percentage)
                
                st.success(f"**Pit stop duration: {stationary_time:.2f} seconds**")
                
                if generate_debug_video:
                    st.subheader("Analysis Debug Video")
                    st.video(debug_output_path)
                    with open(debug_output_path, "rb") as file:
                        st.download_button("Download Debug Video", file, "pitstop_analysis_debug.mp4", "video/mp4")
                else:
                    st.info("To see the visual analysis, check the 'Generate debug video' box and run again.")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
