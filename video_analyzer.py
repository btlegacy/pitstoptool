import cv2
import numpy as np

def analyze_video(video_path, roi_percentage):
    """
    Analyzes the video to determine the time the car is stationary using a user-defined ROI.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps == 0:
        raise Exception("Error: Could not determine video FPS.")

    # Region of Interest from parameters
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    roi = [
        int(frame_height * roi_percentage[0]), # Top
        int(frame_height * roi_percentage[1]), # Bottom
        int(frame_width * roi_percentage[2]),  # Left
        int(frame_width * roi_percentage[3]),  # Right
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
                if is_stationary and stationary_end_frame is None:
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


def analyze_video_with_debug(video_path, output_path, roi_percentage):
    """
    Analyzes the video and generates a debug video using a user-defined ROI.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # Region of Interest from parameters
    roi = [
        int(frame_height * roi_percentage[0]), # Top
        int(frame_height * roi_percentage[1]), # Bottom
        int(frame_width * roi_percentage[2]),  # Left
        int(frame_width * roi_percentage[3]),  # Right
    ]
    
    ROI_COLOR = (0, 255, 0)
    TEXT_COLOR = (255, 255, 255)
    STATUS_MOVING_COLOR = (0, 0, 255)
    STATUS_STATIONARY_COLOR = (0, 255, 0)

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
        
        status_text = "Stationary" if is_stationary else "Moving"
        status_color = STATUS_STATIONARY_COLOR if is_stationary else STATUS_MOVING_COLOR
        
        cv2.putText(frame, f"Motion Score: {motion_score:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, TEXT_COLOR, 2)
        cv2.putText(frame, f"Status: {status_text}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, status_color, 2)

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
        return 0.0
