from VideoProcessor import process_video
import math
import numpy as np
from typing import Tuple


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

def main():
    input_video_path = "Input/input.mp4"
    process_video(input_video_path)

if __name__ == "__main__":
    main()