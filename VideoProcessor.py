import cv2 as cv
import numpy as np
import math
import mediapipe as mp
from typing import Optional, Tuple, Sequence


def getAngle(a: Tuple[int, int], b: Tuple[int, int], c: Tuple[int, int]) -> float:
    """
    Calculate the angle ABC formed by three points.
    """
    ab = (b[0] - a[0], b[1] - a[1])
    bc = (c[0] - b[0], c[1] - b[1])
    dot_product = ab[0] * bc[0] + ab[1] * bc[1]
    mag_ab = math.hypot(ab[0], ab[1])
    mag_bc = math.hypot(bc[0], bc[1])
    if mag_ab == 0 or mag_bc == 0:
        return 0.0
    
    cos_angle = dot_product / (mag_ab * mag_bc)

    return math.degrees(math.acos(np.clip(cos_angle, -1.0, 1.0)))

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


def draw_joint_point(image_bgr: np.ndarray, point_xy: Optional[Tuple[int, int]],
                     bgr_color: Tuple[int, int, int], radius: int = 8, 
                     label_text: Optional[str] = None) -> None:
    """
    Draw a filled circle at a joint location, and an optional text label next to it.
    Does nothing if point_xy is None.
    """
    if point_xy is None:
        return
    else:
        cv.circle(image_bgr, point_xy, radius, bgr_color, thickness=-1)

        if label_text:
            cv.putText(image_bgr, label_text,(point_xy[0] + 8, point_xy[1] - 8),
                    cv.FONT_HERSHEY_SIMPLEX, 0.6, bgr_color, 2, cv.LINE_AA )


def draw_line_segment(image_bgr: np.ndarray, point_a_xy: Optional[Tuple[int, int]],
                      point_b_xy: Optional[Tuple[int, int]], bgr_color: Tuple[int, int, int],
                      thickness: int = 6) -> None:
    """
    Draw a straight line between two points. Skips if either point is None.
    """
    if point_a_xy is None or point_b_xy is None:
        return
    else:
        cv.line(image_bgr, point_a_xy, point_b_xy, bgr_color, thickness, cv.LINE_AA)


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
                             (w, h))

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        # Draw landmarks if present
        results = mp_pose.process(frame) # runs a pose model for the frame
        
        mp_drawing.draw_landmarks(
            frame, # draws image on the frame itself
            results.pose_landmarks, # for each of the frame gitwork
            mp.solutions.pose.POSE_CONNECTIONS, # for the lines connecting them all 
            mp_drawing.DrawingSpec(color=(180,180,180), thickness=4, circle_radius=2), # for the dots
            mp_drawing.DrawingSpec(color=(180,180,180), thickness=6, circle_radius=2) # for the lines
        )
        
        # saving the processed frame to output video
        output.write(frame)
        # Display the resulting frame
        cv.imshow('frame', frame)
        
        # quits the loop if q is pressed
        if cv.waitKey(1) == ord('q'):
            break
    
    # When everything done, release the capture
    cap.release()
    output.release()
    cv.destroyAllWindows()
