import streamlit as st
import tempfile
import os
import cv2
from video_analyzer import analyze_video_with_yolo

st.set_page_config(page_title="Pit Stop Analyzer (YOLO)", layout="wide")

st.title("🏎️ Pit Stop Analysis Tool (YOLOv8)")

st.write(
    "Upload a video of a pit stop. This tool uses YOLOv8 object detection to identify the car and measure pit stop times."
)

# --- Controls in Sidebar ---
st.sidebar.header("Analysis Settings")

with st.sidebar.expander("ROI Settings", expanded=True):
    st.info("The Car ROI defines the pit box. The Tire Change ROI helps pinpoint the stop.")
    # Car ROI
    roi_top = st.slider("Car ROI Top", 0.0, 1.0, 0.04, 0.01)
    roi_bottom = st.slider("Car ROI Bottom", 0.0, 1.0, 0.45, 0.01)
    roi_left = st.slider("Car ROI Left", 0.0, 1.0, 0.25, 0.01)
    roi_right = st.slider("Car ROI Right", 0.0, 1.0, 0.75, 0.01)
    car_roi_percentage = [roi_top, roi_bottom, roi_left, roi_right]
    
    # This ROI is now used to confirm the stop position
    tire_change_roi_top = st.slider("Tire Change ROI Top", 0.0, 1.0, 0.20, 0.01)
    tire_change_roi_bottom = st.slider("Tire Change ROI Bottom", 0.0, 1.0, 0.40, 0.01)
    tire_change_roi_left = st.slider("Tire Change ROI Left", 0.0, 1.0, 0.69, 0.01)
    tire_change_roi_right = st.slider("Tire Change ROI Right", 0.0, 1.0, 0.84, 0.01)
    tire_change_roi_percentage = [tire_change_roi_top, tire_change_roi_bottom, tire_change_roi_left, tire_change_roi_right]


generate_debug_video = st.sidebar.checkbox("Generate debug video", True, help="This will create a video showing the analysis process. Required for this version.")

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
            # Draw Tire Change ROI
            tire_coords = [int(frame_height * tire_change_roi_top), int(frame_height * tire_change_roi_bottom), int(frame_width * tire_change_roi_left), int(frame_width * tire_change_roi_right)]
            cv2.rectangle(preview_frame, (tire_coords[2], tire_coords[0]), (tire_coords[3], tire_coords[1]), (255, 0, 0), 2)
            cv2.putText(preview_frame, "Tire Change ROI", (tire_coords[2], tire_coords[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)
            st.image(preview_frame, channels="BGR", use_column_width=True)
        else:
            st.warning("Could not read the first frame of the video.")

    if st.button("Analyze Pit Stop"):
        with st.spinner("Analyzing video with YOLOv8... This will take longer and download the model on first run."):
            try:
                debug_output_path = os.path.join(tempfile.gettempdir(), "debug_video.mp4")
                analysis_results = analyze_video_with_yolo(video_path, debug_output_path, car_roi_percentage, tire_change_roi_percentage)
                
                tire_change_time = analysis_results.get("tire_change_time", 0.0)
                total_pit_time = analysis_results.get("total_pit_time", 0.0)

                st.success(f"**Tire Change Time: {tire_change_time:.2f} seconds**")
                st.success(f"**Total Pit Stop Time: {total_pit_time:.2f} seconds**")
                
                st.subheader("Analysis Debug Video")
                video_file = open(debug_output_path, 'rb')
                video_bytes = video_file.read()
                st.video(video_bytes)
                st.download_button("Download Debug Video", video_bytes, "pitstop_analysis_debug.mp4", "video/mp4")

            except Exception as e:
                st.error(f"An error occurred during analysis: {e}")
                st.exception(e)
