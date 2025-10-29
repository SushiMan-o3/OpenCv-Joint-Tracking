import cv2 as cv
import numpy as np
import mediapipe as mp
import json
from typing import Optional, Tuple, Sequence


def landmark_to_pixel_xy(landmarks: Sequence, landmark_index: int, 
                         image_width: int, image_height: int, 
                         min_visibility: float = 0.4) -> Optional[Tuple[int, int]]:
    """
    Convert a MediaPipe NORMALIZED landmark (x,y in [0,1]) into integer pixel coords (x_px, y_px).
    Returns None if the landmark's visibility is below min_visibility.
    """
    p = landmarks[landmark_index]
    vis = getattr(p, "visibility", 0.0) or 0.0
    if vis < min_visibility:
        return None
    else:
        # map normalized → pixel, clamped to image bounds
        x_px = int(np.clip(p.x * image_width,  0, image_width  - 1))
        y_px = int(np.clip(p.y * image_height, 0, image_height - 1))
        return (x_px, y_px)


def process_video(input_path: str) -> None:
    """
    Processses a video to detect and annotate human poses using MediaPipe then
    displays the annotated video in a window. Once the video is show, a json file 
    containing the pose landmarks for each frame is saved inside /Outputs folder
    and a video window is displayed.
    """
    cap = cv.VideoCapture(input_path)

    mp_pose = mp.solutions.pose.Pose(min_detection_confidence=0.7, min_tracking_confidence=0.7) 
    mp_drawing = mp.solutions.drawing_utils # used for drawing the figures


    if not cap.isOpened():
        print(f"Cannot open video: {input_path}")
        return
    
    ret, frame = cap.read()
    h, w = frame.shape[:2]

    output = cv.VideoWriter("Output/output.mp4",
                             cv.VideoWriter_fourcc(*'mp4v'),
                             30,
                             (w, h)
                            )
        
    curr_index = 0
    landmark_data = {}

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        results = mp_pose.process(rgb) # runs a pose model for the frame

        landmarks = {}

        # Draw landmarks if present
        if results.pose_landmarks is not None:
            # draw using the same results
            mp_drawing.draw_landmarks(
                frame, # draws image on the frame itself
                results.pose_landmarks, # for each of the frame gitwork
                mp.solutions.pose.POSE_CONNECTIONS, # for the lines connecting them all 
                mp_drawing.DrawingSpec(color=(180,180,180), thickness=4, circle_radius=2), # for the dots
                mp_drawing.DrawingSpec(color=(180,180,180), thickness=6, circle_radius=2) # for the lines
            )

            # Collecting landmark data for the current frame
            lm_list = results.pose_landmarks.landmark
            for lm in mp.solutions.pose.PoseLandmark:
                xy = landmark_to_pixel_xy(lm_list, lm.value, w, h, min_visibility=0.4)
                landmarks[lm.name] = xy
        else:
            for lm in mp.solutions.pose.PoseLandmark:
                landmarks[lm.name] = None


        landmark_data[curr_index] = landmarks

        # saving the processed frame to output video
        output.write(frame)
        # Display the resulting frame
        cv.imshow('frame', frame)
        
        # quits the loop if q is pressed
        if cv.waitKey(1) == ord('q'):
            break

        curr_index += 1
    
    # When everything done, release the capture
    cap.release()
    output.release()
    cv.destroyAllWindows()

    # Saving the landmarks data to a json file
    with open("Output/data.json", "w") as f:
        json.dump(landmark_data, f, indent=4)
