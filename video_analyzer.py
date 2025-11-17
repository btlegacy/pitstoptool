import cv2
import numpy as np
from ultralytics import YOLO
from collections import deque

def does_intersect(box1, box2):
    """Check if two bounding boxes intersect."""
    return not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])

def analyze_video_with_yolo(video_path, output_path, car_roi_percentage, tire_change_roi_percentage):
    """
    Analyzes video using YOLOv8 object detection.
    """
    # Load a pre-trained YOLOv8 model
    model = YOLO('yolov8n.pt')

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise Exception("Error: Could not open video file.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    # --- Define ROIs in pixels ---
    car_roi = [int(frame_height * car_roi_percentage[0]), int(frame_height * car_roi_percentage[1]), int(frame_width * car_roi_percentage[2]), int(frame_width * car_roi_percentage[3])]
    tire_change_roi = [int(frame_height * tire_change_roi_percentage[0]), int(frame_height * tire_change_roi_percentage[1]), int(frame_width * tire_change_roi_percentage[2]), int(frame_width * tire_change_roi_percentage[3])]

    class State:
        WAITING_FOR_CAR = "Waiting for car"
        CAR_IN_STALL = "Car in stall"
        TIRE_CHANGE = "Tire change"
        CAR_LEAVING = "Car leaving"
        ANALYSIS_COMPLETE = "Analysis Complete"
    
    current_state = State.WAITING_FOR_CAR
    
    tire_change_start_frame, tire_change_end_frame = None, None
    total_pit_start_frame, total_pit_end_frame = None, None
    frame_number = 0

    # Store recent car positions to check for movement
    car_positions = deque(maxlen=5)
    MOVEMENT_THRESHOLD = 5.0  # Pixels

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Run YOLO detection
        results = model(frame, verbose=False)
        
        car_detected_in_roi = False
        car_box = None

        # Process detections
        for r in results:
            for box in r.boxes:
                # Check if the detected object is a car, truck, or bus with high confidence
                if model.names[int(box.cls)] in ['car', 'truck', 'bus'] and box.conf > 0.5:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detected_box = [x1, y1, x2, y2]
                    
                    # Check if the detected car intersects with the main Car ROI
                    if does_intersect(detected_box, [car_roi[2], car_roi[0], car_roi[3], car_roi[1]]):
                        car_detected_in_roi = True
                        car_box = detected_box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                        cv2.putText(frame, f"Car: {box.conf[0]:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,0,255), 2)
                        break # Process first car found in ROI
            if car_detected_in_roi: break

        # --- State Machine Logic ---
        if current_state == State.WAITING_FOR_CAR:
            if car_detected_in_roi:
                current_state = State.CAR_IN_STALL
                total_pit_start_frame = frame_number
        
        elif current_state == State.CAR_IN_STALL:
            if car_box:
                car_positions.append(np.mean([car_box[0], car_box[2]])) # Append center X
                # Check for stop by seeing if recent positions are stable
                if len(car_positions) == car_positions.maxlen:
                    pos_std_dev = np.std(car_positions)
                    if pos_std_dev < MOVEMENT_THRESHOLD:
                        current_state = State.TIRE_CHANGE
                        tire_change_start_frame = frame_number

        elif current_state == State.TIRE_CHANGE:
            if car_box:
                car_positions.append(np.mean([car_box[0], car_box[2]]))
                pos_std_dev = np.std(car_positions)
                if pos_std_dev >= MOVEMENT_THRESHOLD:
                    current_state = State.CAR_LEAVING
                    tire_change_end_frame = frame_number
        
        elif current_state == State.CAR_LEAVING:
            if not car_detected_in_roi: # Car has left the main ROI
                current_state = State.ANALYSIS_COMPLETE
                total_pit_end_frame = frame_number

        # --- Drawing for Debug Video ---
        cv2.rectangle(frame, (car_roi[2], car_roi[0]), (car_roi[3], car_roi[1]), (0, 255, 0), 2)
        cv2.rectangle(frame, (tire_change_roi[2], tire_change_roi[0]), (tire_change_roi[3], tire_change_roi[1]), (255, 0, 0), 2)
        cv2.putText(frame, f"State: {current_state}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
        
        if tire_change_start_frame is not None:
            end = tire_change_end_frame if tire_change_end_frame is not None else frame_number
            cv2.putText(frame, f"Tire Change Time: {(end - tire_change_start_frame) / fps:.2f}s", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        if total_pit_start_frame is not None:
            end = total_pit_end_frame if total_pit_end_frame is not None else frame_number
            cv2.putText(frame, f"Total Pit Stop Time: {(end - total_pit_start_frame) / fps:.2f}s", (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        out.write(frame)
        frame_number += 1

    cap.release()
    out.release()
    
    # --- Final Calculations ---
    results = {}
    end_frame_tire = tire_change_end_frame if tire_change_end_frame is not None else frame_number if tire_change_start_frame else None
    end_frame_total = total_pit_end_frame if total_pit_end_frame is not None else frame_number if total_pit_start_frame else None

    results["tire_change_time"] = ((end_frame_tire - tire_change_start_frame) / fps) if tire_change_start_frame and end_frame_tire else 0.0
    results["total_pit_time"] = ((end_frame_total - total_pit_start_frame) / fps) if total_pit_start_frame and end_frame_total else 0.0
        
    return results
