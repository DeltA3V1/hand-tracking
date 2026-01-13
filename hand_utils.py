import math
class HandUtils:
    
    FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20
    }

    @staticmethod
    def truncate(num, decimals=0):
        if decimals < 0:
            raise ValueError("decimals must be non-negative")
        factor = 10 ** decimals
        return int(num * factor) / factor

    @staticmethod
    def distance(p1, p2, roundTo=0):
        return HandUtils.truncate(math.hypot(p2[0]-p1[0], p2[1]-p1[1]), roundTo)
    
    @staticmethod
    def lm_to_pixel(landmark, w, h):
        return int(landmark.x * w), int(landmark.y * h)
