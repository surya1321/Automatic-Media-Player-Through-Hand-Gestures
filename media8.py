import cv2
import mediapipe as mp
import pyautogui
import collections
import logging
import signal
import sys
import time

# Configuration
DETECTION_CONFIDENCE = 0.8
TRACKING_CONFIDENCE = 0.5
BUFFER_SIZE = 5
FRAME_SKIP = 3  # Process every N-th frame

# Initialize logging
logging.basicConfig(filename='gesture_control.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Initialize Mediapipe hands and drawing modules
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Capture video from the webcam
cap = cv2.VideoCapture(0)

# Check if the camera is available
if not cap.isOpened():
    logging.error("Error: Camera not accessible")
    print("Error: Camera not accessible")
    exit()

# Gesture tracking buffer
gesture_buffer = collections.deque(maxlen=BUFFER_SIZE)

# Check if all finger tips are above their respective knuckles
def is_open_palm(landmarks):
    return all(landmarks[i].y < landmarks[i - 3].y for i in [8, 12, 16, 20])

# Check if all finger tips are close to their knuckles
def is_fist(landmarks):
    return all(abs(landmarks[i].y - landmarks[i - 3].y) < 0.05 for i in [8, 12, 16, 20])

# Detect gestures based on hand landmarks
def detect_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    distance = ((thumb_tip.x - index_tip.x) ** 2 + (thumb_tip.y - index_tip.y) ** 2) ** 0.5

    if is_fist(landmarks):
        return "stop"  # Closed fist to stop playback
    elif is_open_palm(landmarks):
        return "play_pause"  # Open palm for play/pause
    elif distance < 0.05:
        return "play_pause"  # Thumb and index finger close together
    elif thumb_tip.y < index_tip.y:
        return "volume_up"  # Thumb below index finger for volume up
    else:
        return "volume_down"  # Thumb above index finger for volume down

# Execute gesture actions only if stable over several frames
def execute_gesture(gesture):
    gesture_buffer.append(gesture)
    
    # Only act if the same gesture is consistently detected over all frames in the buffer
    if len(gesture_buffer) == gesture_buffer.maxlen and len(set(gesture_buffer)) == 1:
        try:
            if gesture == "play_pause":
                pyautogui.press('space')  # Play/Pause
                logging.info("Play/Pause")
            elif gesture == "volume_up":
                pyautogui.press('volumeup')  # Volume Up
                logging.info("Volume Up")
            elif gesture == "volume_down":
                pyautogui.press('volumedown')  # Volume Down
                logging.info("Volume Down")
            elif gesture == "stop":
                pyautogui.press('volumemute')  # Stop/Play
                logging.info("Stop/Play")
        except Exception as e:
            logging.error(f"Failed to execute gesture {gesture}: {e}")

# Handle user interrupt (Ctrl+C) gracefully
def signal_handler(sig, frame):
    logging.info('Exiting...')
    print('Exiting...')
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)

# Main loop
try:
    with mp_hands.Hands(min_detection_confidence=DETECTION_CONFIDENCE, min_tracking_confidence=TRACKING_CONFIDENCE) as hands:
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                logging.error("Error: Unable to capture video frame")
                print("Error: Unable to capture video frame")
                break

            # Skip frames to improve performance
            frame_count += 1
            if frame_count % FRAME_SKIP != 0:
                continue

            # Flip the frame horizontally and convert the color to RGB
            frame = cv2.flip(frame, 1)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process the frame to find hands
            results = hands.process(rgb_frame)

            # Draw landmarks on detected hands
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    # Detect the gesture
                    gesture = detect_gesture(hand_landmarks.landmark)

                    # Execute the gesture if it is stable
                    execute_gesture(gesture)

                    # Display the detected gesture on the frame
                    text = {
                        "play_pause": "Play/Pause",
                        "volume_up": "Volume Up",
                        "volume_down": "Volume Down",
                        "stop": "Stop/Play"
                    }.get(gesture, "")

                    if text:
                        cv2.putText(frame, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Display the frame
            cv2.imshow('Media Player Gesture Control', frame)

            # Exit on pressing 'q'
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

except Exception as e:
    logging.error(f"An error occurred: {e}")
    print(f"An error occurred: {e}")

finally:
    cap.release()
    cv2.destroyAllWindows()
