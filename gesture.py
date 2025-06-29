import cv2
import mediapipe as mp
import pyautogui
import time

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Initialize webcam
cap = cv2.VideoCapture(0)

# Variables for gesture detection
prev_x = 0
gesture_threshold = 60  # Adjust this for sensitivity
last_gesture_time = time.time()
cooldown = 1  # seconds

print("Ready! Use your middle finger to swipe left or right to switch windows.")

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip for mirror image and convert to RGB
    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get x-coordinate of middle fingertip (landmark 12)
            x = hand_landmarks.landmark[12].x
            width = img.shape[1]
            curr_x = int(x * width)

            # Calculate delta movement
            delta_x = curr_x - prev_x

            # Gesture detected if movement is large enough and cooldown passed
            if abs(delta_x) > gesture_threshold and time.time() - last_gesture_time > cooldown:
                if delta_x > 0:
                    print("Swipe Right → Next Window")
                    pyautogui.hotkey('alt', 'tab')
                else:
                    print("Swipe Left ← Previous Window")
                    pyautogui.hotkey('alt', 'shift', 'tab')

                last_gesture_time = time.time()

            prev_x = curr_x

    # Show video with landmarks
    cv2.imshow("Gesture-Controlled Window Switcher", img)

    # Exit on pressing 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("Exiting...")
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
