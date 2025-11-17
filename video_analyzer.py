import cv2
import numpy as np

def analyze_video(video_path):
    """
    Analyzes the video to determine the time the car is stationary.

    This function uses frame differencing within a defined Region of Interest (ROI)
    to detect motion. It identifies the first frame where motion stops and the
    first frame where motion resumes to calculate the duration.

    Args:
        video_path (str): The path to the video file.

    Returns:
        float: The duration in seconds that the car was stationary.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise Exception("Error: Could not determine video FPS.")

    # --- Region of Interest (ROI) Definition ---
    # Based on the sample video, we define a ROI to focus on the car
    # and ignore pit crew movement. This is defined as a percentage of the
    # frame dimensions to work with different video resolutions.
    # Format: [startY, endY, startX, endX]
    roi_percentage = [0.35, 0.65, 0.25, 0.75]

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    roi = [
        int(frame_height * roi_percentage[0]),
        int(frame_height * roi_percentage[1]),
        int(frame_width * roi_percentage[2]),
        int(frame_width * roi_percentage[3]),
    ]

    # --- Motion Detection Logic ---
    prev_frame = None
    stationary_start_frame = None
    stationary_end_frame = None
    is_stationary = False
    
    # This threshold determines how much change is considered "motion".
    # It may need tuning for different video qualities or lighting.
    MOTION_THRESHOLD = 5.0  

    frame_number = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Crop the frame to the ROI
        roi_frame = frame[roi[0]:roi[1], roi[2]:roi[3]]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is not None:
            # Calculate the difference between the current and previous frame
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            
            # Calculate the percentage of changed pixels in the ROI
            motion_score = (np.sum(thresh) / 255) / (thresh.shape[0] * thresh.shape[1]) * 100

            if motion_score < MOTION_THRESHOLD:
                if not is_stationary:
                    # Car has just become stationary
                    is_stationary = True
                    stationary_start_frame = frame_number
            else:
                if is_stationary:
                    # Car has just started moving again
                    stationary_end_frame = frame_number
                    # We have our measurement, so we can stop processing
                    break
        
        prev_frame = gray
        frame_number += 1

    cap.release()

    if stationary_start_frame is not None and stationary_end_frame is not None:
        stationary_frames = stationary_end_frame - stationary_start_frame
        stationary_duration = stationary_frames / fps
        return stationary_duration
    elif stationary_start_frame is not None and stationary_end_frame is None:
        # This means the car stopped and the video ended before it moved again.
        stationary_frames = frame_number - stationary_start_frame
        stationary_duration = stationary_frames / fps
        return stationary_duration
    else:
        # If no stationary period was detected
        raise Exception("Could not detect a stationary period for the car in the video.")
