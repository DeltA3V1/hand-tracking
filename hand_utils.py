import math
class HandUtils:
    
    FINGER_TIPS = {
    "thumb": 4,
    "index": 8,
    "middle": 12,
    "ring": 16,
    "pinky": 20
    }

    NUMBER_MAP = {
        2: 1,      # index
        6: 2,      # index + middle
        14: 3,     # index + middle + ring
        30: 4,     # index + middle + ring + pinky
        31: 5,     # all fingers
        1: 6,      # thumb
        17: 6,     # thumb + pinky
        3: 7,      # thumb + index
        7: 8,      # thumb + index + middle
        15: 9      # thumb + index + middle + ring
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

    @staticmethod
    def thumb_extended(hand, handedness="Right"):
        tip = hand[4]
        ip  = hand[3]
        mcp = hand[2]
        wrist = hand[0]
        index_mcp = hand[5]

        # Vector from MCP → tip
        vx = tip.x - mcp.x
        vy = tip.y - mcp.y

        # Vector from wrist → MCP
        wx = mcp.x - wrist.x
        wy = mcp.y - wrist.y

        # Dot product to see if thumb points outward
        dot = vx * wx + vy * wy

        # Length of thumb
        length = math.hypot(vx, vy)

        # Is thumb pointing outward?
        outward = dot > 0 and length > 0.04

        # Is thumb crossing palm?
        PALM_MARGIN = 0.04   # tweak between 0.02 and 0.06

        if handedness == "Right":
            not_across_palm = tip.x < index_mcp.x - PALM_MARGIN
        else:
            not_across_palm = tip.x > index_mcp.x + PALM_MARGIN


        return outward and not_across_palm

    @staticmethod
    def get_finger_states(hand_landmarks):
        fingers = {}

        # Index, middle, ring, pinky
        fingers["index"]  = hand_landmarks[8].y < hand_landmarks[6].y
        fingers["middle"] = hand_landmarks[12].y < hand_landmarks[10].y
        fingers["ring"]   = hand_landmarks[16].y < hand_landmarks[14].y
        fingers["pinky"]  = hand_landmarks[20].y < hand_landmarks[18].y

        # Thumb (assume right hand for now)
        fingers["thumb"] = HandUtils.thumb_extended(hand_landmarks)

        return fingers

    @staticmethod
    def finger_code(f):
        code = 0
        if f["thumb"]:  code |= 1
        if f["index"]:  code |= 2
        if f["middle"]: code |= 4
        if f["ring"]:   code |= 8
        if f["pinky"]:  code |= 16
        return code

    @staticmethod
    def recognize_number(f):
        code = HandUtils.finger_code(f)
        return HandUtils.NUMBER_MAP.get(code)