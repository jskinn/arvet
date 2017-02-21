from enum import Enum


class VisionSystemType(Enum):
    LOOP_CLOSURE_DETECTION = 0,
    MONOCULAR_SLAM = 1,
    VISUAL_ODOMETRY = 2
