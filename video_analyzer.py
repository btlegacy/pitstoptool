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
    Analyzes video using a multi-stage approach, returning tire change and total pit stop time.
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
        CAR_ARRIVED = "Car in stall"
        CAR_STATIONARY = "Tire change"
        CAR_LEAVING = "Car leaving stall"
        ANALYSIS_COMPLETE = "Analysis Complete"
    
    current_state = State.WAITING_FOR_CAR
    
    prev_car_gray, prev_sign_gray = None, None
    tire_change_start_frame, tire_change_end_frame = None, None
    total_pit_start_frame, total_pit_end_frame = None, None
    frame_number = 0
    
    # Threshold to decide when the car has fully left the ROI
    CAR_EXIT_THRESHOLD = 0.5 

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

        # --- State Machine Logic ---
        if current_state == State.WAITING_FOR_CAR:
            if car_motion > thresholds["arrival"]:
                current_state = State.CAR_ARRIVED
                total_pit_start_frame = frame_number
        elif current_state == State.CAR_ARRIVED:
            if sign_motion > thresholds["hit"]:
                current_state = State.CAR_STATIONARY
                tire_change_start_frame = frame_number
        elif current_state == State.CAR_STATIONARY:
            if car_motion > thresholds["departure"]:
                current_state = State.CAR_LEAVING
                tire_change_end_frame = frame_number
        elif current_state == State.CAR_LEAVING:
             if car_motion < CAR_EXIT_THRESHOLD:
                current_state = State.ANALYSIS_COMPLETE
                total_pit_end_frame = frame_number

        # --- Drawing for Debug Video ---
        cv2.rectangle(frame, (car_roi[2], car_roi[0]), (car_roi[3], car_roi[1]), (0, 255, 0), 2)
        cv2.rectangle(frame, (sign_roi[2], sign_roi[0]), (sign_roi[3], sign_roi[1]), (255, 0, 0), 2)
        cv2.putText(frame, f"State: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Car Motion: {car_motion:.2f}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        cv2.putText(frame, f"Sign Motion: {sign_motion:.2f}", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        # Tire Change Timer
        if tire_change_start_frame is not None:
            end_frame = tire_change_end_frame if tire_change_end_frame is not None else frame_number
            timer_seconds = (end_frame - tire_change_start_frame) / fps
            cv2.putText(frame, f"Tire Change Time: {timer_seconds:.2f}s", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Total Pit Stop Timer
        if total_pit_start_frame is not None:
            end_frame = total_pit_end_frame if total_pit_end_frame is not None else frame_number
            timer_seconds = (end_frame - total_pit_start_frame) / fps
            cv2.putText(frame, f"Total Pit Stop Time: {timer_seconds:.2f}s", (10, 190), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)
        prev_car_gray, prev_sign_gray = car_gray, sign_gray
        frame_number += 1

    cap.release()
    out.release()
    
    results = {}
    # Calculate final tire change time
    if tire_change_start_frame is not None:
        end_frame = tire_change_end_frame if tire_change_end_frame is not None else frame_number
        results["tire_change_time"] = (end_frame - tire_change_start_frame) / fps
    else:
        results["tire_change_time"] = 0.0

    # Calculate final total pit time
    if total_pit_start_frame is not None:
        end_frame = total_pit_end_frame if total_pit_end_frame is not None else frame_number
        results["total_pit_time"] = (end_frame - total_pit_start_frame) / fps
    else:
        results["total_pit_time"] = 0.0
        
    return results
