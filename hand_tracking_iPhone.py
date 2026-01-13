# hand_tracking_iPhone_fixed.py
import threading
import time
from pathlib import Path
import pyautogui

import cv2
import mediapipe as mp
from mediapipe.tasks.python.vision import (
    HandLandmarker,
    HandLandmarkerOptions,
    RunningMode,
)
from mediapipe.tasks.python.core.base_options import BaseOptions

from hand_utils import HandUtils


# --- CONFIG -----------------------------------------------------------------
MODEL_PATH = str(Path(__file__).parent / "hand_landmarker.task")
CAM_INDEX = 2             # set to your detected Camo index (0/1/2...). change if needed
FPS_APPROX = 20
LINE_THICKNESS = 4
BONE_THICKNESS = 2
TEXT_SIZE = 1
PINCH_THRESHOLD = 100  # distance in pixels, adjust for your camera resolution
CAM_BOX_MARGIN = 0.2  # x% margin on each side
# PERFORMANCE TIPS:
# - Lower FPS_APPROX (e.g., 15) reduces processing frequency
# - Increase BONE_THICKNESS/LINE_THICKNESS for less detailed drawing (GPU may not benefit but looks cleaner)
# - Use CAM_INDEX wisely; some cameras are slower than others
# ---------------------------------------------------------------------------


# Hard-coded MediaPipe hand connections (standard MediaPipe hand graph)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index finger
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle finger
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring finger
    (13, 17), (17, 18), (18, 19), (19, 20), # Little finger
    (0, 17)                                 # Palm base
]


# Shared buffer written by callback and read by main thread
_latest_frame = None
_latest_lock = threading.Lock()
_last_pinch = {}

# Smoothed landmark positions: one dict per hand
_smoothed = {}   # key = (hand_index, landmark_index) -> (x, y)
SMOOTH_ALPHA = 0.65

# Cached number tracking (persist text while updating less frequently)
_last_number = None
_last_number_pos = None

#Toggles
DRAW = True
DRAW_PINCH_LINE = False
SHOW_BBOX = True
DRAW_LANDMARKS = True
TRACK_NUMBERS = False
_click_state = False  # track whether we are currently "holding click"

# Mouse control
MOUSE_CONTROL = False
SCREEN_W, SCREEN_H = pyautogui.size()
pyautogui.PAUSE = 0      # remove small automatic pause
pyautogui.FAILSAFE = False

# Shared mouse target (updated by callback, consumed by mouse thread)
_mouse_target = None     # (x, y) in screen coords
_mouse_lock = threading.Lock()
_mouse_thread = None
_mouse_thread_stop = False

# Throttle parameters
MOUSE_THREAD_HZ = 30  # Reduced from 60 Hz (30 Hz is plenty for mouse)
MOUSE_MOVE_THRESHOLD = 5   # Increased threshold to ignore more tiny moves

# Frame counter for throttling expensive operations
_frame_counter = 0
TEXT_RENDER_INTERVAL = 5  # Render text every 5 frames (was 3)
LANDMARK_DRAW_INTERVAL = 1  # Draw landmarks every N frames (1 = every frame, 2 = every other)


def smooth_point(hand_id, lm_id, x, y):
    key = (hand_id, lm_id)

    if key not in _smoothed:
        _smoothed[key] = (x, y)
        return x, y

    prev_x, prev_y = _smoothed[key]
    sx = int(SMOOTH_ALPHA * x + (1 - SMOOTH_ALPHA) * prev_x)
    sy = int(SMOOTH_ALPHA * y + (1 - SMOOTH_ALPHA) * prev_y)

    _smoothed[key] = (sx, sy)
    return sx, sy



def hand_result_callback(result, output_image, timestamp_ms):
    """
    Called by MediaPipe in LIVE_STREAM mode.
    output_image is a mediapipe Image. Convert -> numpy, draw, store.
    """
    global _latest_frame, _frame_counter, _mouse_target
    _frame_counter += 1
    # Convert MediaPipe Image -> writable NumPy RGB array
    try:
        frame_rgb = output_image.numpy_view().copy()  # RGB, HxWx3, writable copy
    except Exception as e:
        # If conversion fails, skip drawing this frame
        print("Callback: couldn't numpy_view() the output_image:", e)
        return

    h, w = frame_rgb.shape[:2]

    # Draw landmarks and connections on the RGB frame
    if getattr(result, "hand_landmarks", None):
        for hand_id, hand_landmarks in enumerate(result.hand_landmarks):
            xs, ys = [], []
            smoothed_points = {}
           
            for idx, landmark in enumerate(hand_landmarks):
                x, y = HandUtils.lm_to_pixel(landmark, w, h)
                sx, sy = smooth_point(hand_id, idx, x, y)
                smoothed_points[idx] = (sx, sy)
                xs.append(sx)
                ys.append(sy)
                
                # Only draw landmarks every Nth frame to reduce CPU load
                if DRAW and DRAW_LANDMARKS and _frame_counter % LANDMARK_DRAW_INTERVAL == 0:
                    cv2.circle(frame_rgb, (sx, sy), 4, (0, 255, 0), -1)
                    cv2.putText(frame_rgb, str(idx), (sx+5, sy-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), TEXT_SIZE)

            # --- Step 4: draw connections using smoothed points ---
            for start_idx, end_idx in HAND_CONNECTIONS:
                s = hand_landmarks[start_idx]
                e = hand_landmarks[end_idx]

                sx, sy = smoothed_points[start_idx]
                ex, ey = smoothed_points[end_idx]

                # Only draw connections if drawing is enabled (skip entirely if not)
                if DRAW:
                    cv2.line(frame_rgb, (sx, sy), (ex, ey), (0, 255, 255), BONE_THICKNESS)

            # Move mouse using index fingertip (runs once per hand after landmarks are processed)
            if MOUSE_CONTROL and 8 in smoothed_points:
                index_x, index_y = smoothed_points[8]  # index tip

                # Control box (optional)
                min_x = int(CAM_BOX_MARGIN * w)
                max_x = int((1 - CAM_BOX_MARGIN) * w)
                min_y = int(CAM_BOX_MARGIN * h)
                max_y = int((1 - CAM_BOX_MARGIN) * h)

                clamped_x = max(min_x, min(index_x, max_x))
                clamped_y = max(min_y, min(index_y, max_y))

                # Normalize inside control box
                nx = (clamped_x - min_x) / (max_x - min_x)
                ny = (clamped_y - min_y) / (max_y - min_y)

                # optional flip
                #nx = 1 - nx

                # Map to screen
                screen_x = int(nx * SCREEN_W)
                screen_y = int(ny * SCREEN_H)

                with _mouse_lock:
                    _mouse_target = (screen_x, screen_y)


            # Always compute smoothed points for landmarks
            # --- CLICK LOGIC using thumb + middle finger ---
            if MOUSE_CONTROL and 4 in smoothed_points and 12 in smoothed_points:
                thumb_x, thumb_y = smoothed_points[4]
                middle_x, middle_y = smoothed_points[12]

                pinch_dist = HandUtils.distance((thumb_x, thumb_y), (middle_x, middle_y))

                global _click_state
                if pinch_dist < PINCH_THRESHOLD and not _click_state:
                    pyautogui.mouseDown()  # start click
                    _click_state = True
                elif pinch_dist >= PINCH_THRESHOLD and _click_state:
                    pyautogui.mouseUp()    # release click
                    _click_state = False

                # Optional: draw line for debugging
                if DRAW and DRAW_PINCH_LINE:
                    cv2.line(frame_rgb, (thumb_x, thumb_y), (middle_x, middle_y), (0, 0, 255), LINE_THICKNESS)
                    if _frame_counter % TEXT_RENDER_INTERVAL == 0:
                        mid_x = (thumb_x + middle_x) // 2
                        mid_y = (thumb_y + middle_y) // 2
                        cv2.putText(frame_rgb, str(int(pinch_dist)), (mid_x, mid_y),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), TEXT_SIZE)

            # Draw bounding box only if enabled
            if DRAW and SHOW_BBOX and xs:
                # --- bounding box (only once per hand, after all points) ---
                    cv2.rectangle(frame_rgb,
                                    (min(xs), min(ys)),
                                    (max(xs), max(ys)),
                                    (0, 255, 255), BONE_THICKNESS)
                    
            # --- Number tracking ---
            if DRAW and TRACK_NUMBERS:
                # Only update recognition every N frames to reduce CPU load
                if _frame_counter % TEXT_RENDER_INTERVAL == 0:
                    global _last_number, _last_number_pos
                    _last_number = HandUtils.recognize_number(HandUtils.get_finger_states(hand_landmarks))
                    _last_number_pos = (min(xs), min(ys)-10)
                
                # Draw the number every frame using cached value
                if _last_number is not None and _last_number_pos is not None:
                    cv2.putText(frame_rgb, f"Num: {_last_number}", _last_number_pos,
                                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), TEXT_SIZE + 5)

    # Convert RGB back to BGR for OpenCV             
    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
    # Store into shared buffer (thread-safe)
    with _latest_lock:
        _latest_frame = frame_bgr


def mouse_worker():
    global _mouse_thread_stop
    last_pos = None
    delay = 1.0 / MOUSE_THREAD_HZ
    while not _mouse_thread_stop:
        with _mouse_lock:
            target = _mouse_target

        if target is not None:
            tx, ty = target
            # Only move if different enough
            if last_pos is None or abs(tx - last_pos[0]) > MOUSE_MOVE_THRESHOLD or abs(ty - last_pos[1]) > MOUSE_MOVE_THRESHOLD:
                try:
                    pyautogui.moveTo(tx, ty, duration=0)  # instantaneous jump
                except Exception:
                    pass
                last_pos = (tx, ty)
        time.sleep(delay)



def main():
    global _latest_frame, DRAW_PINCH_LINE, SHOW_BBOX, DRAW_LANDMARKS, MOUSE_CONTROL, DRAW, TRACK_NUMBERS
    
    # start mouse thread
    _mouse_thread_stop = False
    _mouse_thread = threading.Thread(target=mouse_worker, daemon=True)
    _mouse_thread.start()

    # Create HandLandmarker in LIVE_STREAM with required callback
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=RunningMode.LIVE_STREAM,
        num_hands=2,
        result_callback=hand_result_callback,
    )
    hand_landmarker = HandLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Cannot open camera index {CAM_INDEX}. Run list_cams.py to find indices.")
        hand_landmarker.close()
        return

    print("'b' to toggle bounding box, 'z' to toggle thumb-index line.")
    print("Starting. Press 'q' in the window to quit.")

    frame_timestamp = 0  # milliseconds, incremental
    frame_interval_ms = int(1000 / FPS_APPROX)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Frame grab failed, stopping.")
                break

            # Convert BGR -> RGB for MediaPipe
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Wrap as mediapipe Image (required by detect_async)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            # Send to Landmarker (async) with timestamp
            hand_landmarker.detect_async(mp_image, timestamp_ms=frame_timestamp)
            frame_timestamp += frame_interval_ms

            # Display the most recent processed frame if available; otherwise show camera
            with _latest_lock:
                show = _latest_frame.copy() if _latest_frame is not None else frame

            cv2.imshow("Hand Tracking", show)

            # Key cmds
            key = cv2.waitKey(1) & 0xFF

            #Quit 'q'
            if key == ord("q"):
                break
            
            #Track numbers 'n'
            if key == ord("n"):
                TRACK_NUMBERS = not TRACK_NUMBERS
                print("Number Tracking:", "ON" if TRACK_NUMBERS else "OFF")

            #Mouse control 'm'
            if key == ord("m"):
                MOUSE_CONTROL = not MOUSE_CONTROL
                DRAW = not DRAW
                print("Mouse:", "ON" if MOUSE_CONTROL else "OFF")

            #Draw landmarks 'l'
            if key == ord("l"):
                DRAW_LANDMARKS = not DRAW_LANDMARKS
                print("Landmarks:", "ON" if DRAW_LANDMARKS else "OFF")

            #Bounding box 'b'
            if key == ord("b"):
                SHOW_BBOX = not SHOW_BBOX
                print("Bounding box:", "ON" if SHOW_BBOX else "OFF")

            #Thumb to index 'z'
            if key == ord("z"):
                DRAW_PINCH_LINE = not DRAW_PINCH_LINE
                print("Pinch line:", "ON" if DRAW_PINCH_LINE else "OFF")


            # small sleep to yield CPU (keeps loop pacing closer to FPS_APPROX)
            time.sleep(frame_interval_ms / 1000.0)

    finally:
        # stop mouse thread
        _mouse_thread_stop = True
        if _mouse_thread is not None:
            _mouse_thread.join(timeout=0.5)

        cap.release()
        cv2.destroyAllWindows()
        hand_landmarker.close()


if __name__ == "__main__":
    main()
