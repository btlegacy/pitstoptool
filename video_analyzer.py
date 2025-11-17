import cv2
import numpy as np

def get_motion_score(frame1, frame2, sensitivity):
    """Calculates a motion score between two frames with adjustable sensitivity."""
    if frame1 is None or frame2 is None:
        return 0
    frame_delta = cv2.absdiff(frame1, frame2)
    thresh = cv2.threshold(frame_delta, sensitivity, 255, cv2.THRESH_BINARY)[1]
    motion_score = (np.sum(thresh) / 255) / (thresh.shape[0] * thresh.shape[1]) * 100
    return motion_score

def analyze_video_with_debug(video_path, output_path, car_roi_percentage, sign_roi_percentage, thresholds):
    """
    Analyzes video using a multi-stage approach with user-defined ROIs and thresholds.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    car_roi = [int(frame_height * p) for p in car_roi_percentage[0:2]] + [int(frame_width * p) for p in car_roi_percentage[2:4]]
    sign_roi = [int(frame_height * p) for p in sign_roi_percentage[0:2]] + [int(frame_width * p) for p in sign_roi_percentage[2:4]]

    class State:
        WAITING_FOR_CAR = "Waiting for car"
        CAR_ARRIVED = "Car arrived, waiting for stop"
        CAR_STATIONARY = "Car stationary"
        CAR_DEPARTED = "Car departed"
    
    current_state = State.WAITING_FOR_CAR
    
    prev_car_gray, prev_sign_gray = None, None
    stationary_start_frame, stationary_end_frame = None, None
    frame_number = 0
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        car_roi_frame = frame[car_roi[0]:car_roi[1], car_roi[2]:car_roi[3]]
        sign_roi_frame = frame[sign_roi[0]:sign_roi[1], sign_roi[2]:sign_roi[3]]
        
        car_gray = cv2.GaussianBlur(cv2.cvtColor(car_roi_frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        sign_gray = cv2.GaussianBlur(cv2.cvtColor(sign_roi_frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)

        car_motion = get_motion_score(prev_car_gray, car_gray, thresholds["sensitivity"])
        sign_motion = get_motion_score(prev_sign_gray, sign_gray, thresholds["sensitivity"])

        # State Machine Logic
        if current_state == State.WAITING_FOR_CAR:
            if car_motion > thresholds["arrival"]:
                current_state = State.CAR_ARRIVED
        elif current_state == State.CAR_ARRIVED:
            if sign_motion > thresholds["hit"]:
                current_state = State.CAR_STATIONARY
                stationary_start_frame = frame_number
        elif current_state == State.CAR_STATIONARY:
            if car_motion > thresholds["departure"]:
                current_state = State.CAR_DEPARTED
                stationary_end_frame = frame_number

        # Drawing for Debug Video
        cv2.rectangle(frame, (car_roi[2], car_roi[0]), (car_roi[3], car_roi[1]), (0, 255, 0), 2)
        cv2.rectangle(frame, (sign_roi[2], sign_roi[0]), (sign_roi[3], sign_roi[1]), (255, 0, 0), 2)
        cv2.putText(frame, f"State: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Car Motion: {car_motion:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Sign Motion: {sign_motion:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        if current_state in [State.CAR_STATIONARY, State.CAR_DEPARTED] and stationary_start_frame is not None:
            timer_frames = (frame_number if stationary_end_frame is None else stationary_end_frame) - stationary_start_frame
            timer_seconds = timer_frames / fps
            # Changed the label to "Tire Change Time"
            cv2.putText(frame, f"Tire Change Time: {timer_seconds:.2f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        out.write(frame)
        prev_car_gray, prev_sign_gray = car_gray, sign_gray
        frame_number += 1

    cap.release()
    out.release()
    
    if stationary_start_frame is not None and stationary_end_frame is not None:
        return (stationary_end_frame - stationary_start_frame) / fps
    else:
        if stationary_start_frame is not None:
            return (frame_number - stationary_start_frame) / fps
        return 0.0
