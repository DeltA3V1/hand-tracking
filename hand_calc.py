from hand_utils import HandUtils
class HandCalc:

    @staticmethod
    def only(hand_fingers, allowed):
        return all(
            (finger in allowed) == state
            for finger, state in hand_fingers.items()
        )

    @staticmethod
    def minus(hand, f):
        return (
            HandCalc.only(f, {"index"}) and
            HandUtils.finger_direction(hand, 8, 6) == "left"
        )

    @staticmethod
    def plus(hand, f):
        return (
            HandCalc.only(f, {"index", "middle"}) and
            HandUtils.finger_direction(hand, 8, 6) == "left" and
            HandUtils.finger_direction(hand, 12, 10) == "left"
        )

    @staticmethod
    def multiply(hand, f):
        return (
            HandCalc.only(f, {"index", "middle", "ring"}) and
            HandUtils.finger_direction(hand, 8, 6) == "left" and
            HandUtils.finger_direction(hand, 12, 10) == "left" and
            HandUtils.finger_direction(hand, 16, 14) == "left"
        )


    @staticmethod
    def divide(hand, f):
        return (
            HandCalc.only(f, {"index", "middle", "ring", "pinky"}) and
            HandUtils.finger_direction(hand, 8, 6) == "left" and
            HandUtils.finger_direction(hand, 12, 10) == "left" and
            HandUtils.finger_direction(hand, 16, 14) == "left" and
            HandUtils.finger_direction(hand, 20, 18) == "left"
        )

    
    @staticmethod
    def recognize_op(hand, f):
        if not any(f.values()):
            return "="
        if HandCalc.minus(hand, f):
            return "-"
        if HandCalc.plus(hand, f):
            return "+"
        if HandCalc.multiply(hand, f):
            return "*"
        if HandCalc.divide(hand, f):
            return "/"

        return None