from hand_utils import HandUtils
class HandCalc:

    @staticmethod
    def minus(hand_landmarks, fingers):
        return (
            fingers["index"] and
            not any([fingers["middle"], fingers["ring"], fingers["pinky"]]) and
            hand_landmarks[8].x < hand_landmarks[6].x - 0.05
        )
    @staticmethod
    def plus(hand_landmarks, fingers):
        return (
        fingers["index"] and fingers["middle"] and
        not any([fingers["ring"], fingers["pinky"]]) and
        hand_landmarks[12].y < hand_landmarks[10].y - 0.05 and
        hand_landmarks[8].y < hand_landmarks[6].y - 0.05
    )
    @staticmethod
    def multiply(hand_landmarks, fingers):
        return (
            fingers["index"] and fingers["middle"] and fingers["ring"] and
            not fingers["pinky"] and
            HandUtils.finger_direction(hand_landmarks, 8, 6) in ["left", "right"] and
            HandUtils.finger_direction(hand_landmarks, 12, 10) in ["left", "right"]
        )
    @staticmethod
    def divide(hand_landmarks, fingers):
        return (
            fingers["index"] and fingers["middle"] and fingers["ring"] and
            not fingers["pinky"] and
            HandUtils.finger_direction(hand_landmarks, 8, 6) in ["left", "right"] and
            HandUtils.finger_direction(hand_landmarks, 12, 10) in ["left", "right"]
        )
    
    @staticmethod
    def recognize(hand_landmarks, fingers):
        if not any(fingers.values()):
            return "="
        if HandCalc.minus(hand_landmarks, fingers):
            return "-"
        if HandCalc.plus(hand_landmarks, fingers):
            return "+"
        if HandCalc.multiply(hand_landmarks, fingers):
            return "*"
        if HandCalc.divide(hand_landmarks, fingers):
            return "/"

        return None