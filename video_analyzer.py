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

    # Region of Interest Definition
    roi_percentage = [0.35, 0.65, 0.25, 0.75]
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = [
        int(frame_height * roi_percentage[0]),
        int(frame_height * roi_percentage[1]),
        int(frame_width * roi_percentage[2]),
        int(frame_width * roi_percentage[3]),
    ]

    prev_frame = None
    stationary_start_frame = None
    stationary_end_frame = None
    is_stationary = False
    MOTION_THRESHOLD = 5.0
    frame_number = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        roi_frame = frame[roi[0]:roi[1], roi[2]:roi[3]]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is not None:
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = (np.sum(thresh) / 255) / (thresh.shape[0] * thresh.shape[1]) * 100

            if motion_score < MOTION_THRESHOLD:
                if not is_stationary:
                    is_stationary = True
                    stationary_start_frame = frame_number
            else:
                if is_stationary:
                    stationary_end_frame = frame_number
                    break
        
        prev_frame = gray
        frame_number += 1

    cap.release()

    if stationary_start_frame is not None and stationary_end_frame is not None:
        return (stationary_end_frame - stationary_start_frame) / fps
    elif stationary_start_frame is not None:
        return (frame_number - stationary_start_frame) / fps
    else:
        raise Exception("Could not detect a stationary period for the car in the video.")


def analyze_video_with_debug(video_path, output_path):
    """
    Analyzes the video and generates a debug video with overlays.

    Args:
        video_path (str): The path to the input video file.
        output_path (str): The path to save the output debug video.

    Returns:
        float: The duration in seconds that the car was stationary.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    roi_percentage = [0.35, 0.65, 0.25, 0.75]
    roi = [
        int(frame_height * roi_percentage[0]),
        int(frame_height * roi_percentage[1]),
        int(frame_width * roi_percentage[2]),
        int(frame_width * roi_percentage[3]),
    ]
    
    # Colors for overlays
    ROI_COLOR = (0, 255, 0)  # Green
    TEXT_COLOR = (255, 255, 255) # White
    STATUS_MOVING_COLOR = (0, 0, 255) # Red
    STATUS_STATIONARY_COLOR = (0, 255, 0) # Green

    prev_frame = None
    stationary_start_frame = None
    stationary_end_frame = None
    is_stationary = False
    MOTION_THRESHOLD = 5.0
    frame_number = 0
    motion_score = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw ROI box on the original frame
        cv2.rectangle(frame, (roi[2], roi[0]), (roi[3], roi[1]), ROI_COLOR, 2)

        roi_frame = frame[roi[0]:roi[1], roi[2]:roi[3]]
        gray = cv2.cvtColor(roi_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_frame is not None:
            frame_delta = cv2.absdiff(prev_frame, gray)
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            motion_score = (np.sum(thresh) / 255) / (thresh.shape[0] * thresh.shape[1]) * 100

            if motion_score < MOTION_THRESHOLD:
                if not is_stationary:
                    is_stationary = True
                    stationary_start_frame = frame_number
            else:
                if is_stationary and stationary_end_frame is None:
                    stationary_end_frame = frame_number
        
        # --- Add overlays to the frame ---
        status_text = "Stationary" if is_stationary else "Moving"
        status_color = STATUS_STATIONARY_COLOR if is_stationary else STATUS_MOVING_COLOR
        
        # Motion Score and Status
        cv2.putText(frame, f"Motion Score: {motion_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
        cv2.putText(frame, f"Status: {status_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

        # Timer
        if is_stationary:
            current_stationary_frames = frame_number - stationary_start_frame
            current_stationary_time = current_stationary_frames / fps
            cv2.putText(frame, f"Stationary Time: {current_stationary_time:.2f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)

        out.write(frame)
        prev_frame = gray
        frame_number += 1

    cap.release()
    out.release()
    
    if stationary_start_frame is not None and stationary_end_frame is not None:
        return (stationary_end_frame - stationary_start_frame) / fps
    elif stationary_start_frame is not None:
        return (frame_number - stationary_start_frame) / fps
    else:
        # To avoid an error if no stationary period is found, return 0 for debug mode
        return 0.0
