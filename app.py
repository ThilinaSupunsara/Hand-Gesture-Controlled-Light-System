import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import base64
import threading
import time

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your_secret_key_here' # Change this to a strong, unique key
socketio = SocketIO(app, cors_allowed_origins="*") # Allow all origins for development, refine for production

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Finger tip landmarks (for counting extended fingers)
# IDs of the tip landmarks for each finger (thumb, index, middle, ring, pinky)
TIP_IDS = [mp_hands.HandLandmark.THUMB_TIP, mp_hands.HandLandmark.INDEX_FINGER_TIP,
           mp_hands.HandLandmark.MIDDLE_FINGER_TIP, mp_hands.HandLandmark.RING_FINGER_TIP,
           mp_hands.HandLandmark.PINKY_TIP]

# Indices of the base landmarks for each finger (for comparison)
# We compare the tip to the joint directly below it to determine if it's extended
# Thumb base is the carpometacarpal joint, others are metacarpophalangeal joints (knuckles)
BASE_IDS = [mp_hands.HandLandmark.THUMB_IP, # Interphalangeal joint for thumb
            mp_hands.HandLandmark.INDEX_FINGER_MCP, # Metacarpophalangeal for index
            mp_hands.HandLandmark.MIDDLE_FINGER_MCP,
            mp_hands.HandLandmark.RING_FINGER_MCP,
            mp_hands.HandLandmark.PINKY_MCP]

# Global variable to store the number of lifted fingers
current_lifted_fingers = 0
video_feed_active = False # Control video processing thread

def count_fingers(hand_landmarks):
    fingers = []

   
    if hand_landmarks.landmark[TIP_IDS[0]].y < hand_landmarks.landmark[BASE_IDS[0]].y:
         # A more robust thumb check might involve comparing x-coords relative to wrist or another reference
         # depending on hand orientation. For a simple up-down, this is okay.
        fingers.append(1) # Thumb is up
    else:
        fingers.append(0) # Thumb is down


    # Other 4 fingers (Index, Middle, Ring, Pinky)
    for i in range(1, 5): # Iterate from index finger to pinky
        if hand_landmarks.landmark[TIP_IDS[i]].y < hand_landmarks.landmark[BASE_IDS[i]].y:
            fingers.append(1) # Finger is up
        else:
            fingers.append(0) # Finger is down

    return sum(fingers)


def generate_frames():
    global current_lifted_fingers, video_feed_active
    cap = cv2.VideoCapture(0) # 0 for default webcam

    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    video_feed_active = True
    print("Video feed started...")

    while video_feed_active:
        success, frame = cap.read()
        if not success:
            break

        # Flip the frame horizontally for a mirror effect (common for webcams)
        frame = cv2.flip(frame, 1)
        
        # Convert the BGR image to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame with MediaPipe Hands
        results = hands.process(rgb_frame)

        num_fingers = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                num_fingers = count_fingers(hand_landmarks) # Count fingers for each hand
                # If you only want to count one hand, you might break here or average

        if num_fingers != current_lifted_fingers:
            current_lifted_fingers = num_fingers
            socketio.emit('finger_count_update', {'count': current_lifted_fingers})
            print(f"Detected {current_lifted_fingers} fingers.")

        # Encode frame as JPEG for streaming to browser
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()
        encoded_frame = base64.b64encode(frame_bytes).decode('utf-8')
        socketio.emit('video_frame', {'image': encoded_frame})

        # Add a small delay to avoid overwhelming the CPU/network
        time.sleep(0.01)

    cap.release()
    hands.close()
    print("Video feed stopped.")

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')
    global video_feed_active
    if not video_feed_active:
        # Start video processing in a separate thread when a client connects
        thread = threading.Thread(target=generate_frames)
        thread.daemon = True # Allow the main program to exit even if this thread is running
        thread.start()

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')
    


if __name__ == '__main__':
    # Use 0.0.0.0 to make it accessible from other devices on the network
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)