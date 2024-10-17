from enum import Enum

class ImageType(Enum):
    PANORAMA = 1
    CANNY_RESULT = 2
    DoG_RESULT = 3
    TEMP_C = 4
    TEMP_D = 5
    HUMAN_DETECTION = 6