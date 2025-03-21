import mediapipe as mp
import cv2
import numpy as np

# Set up mediapipe instance
mp_holistic = mp.solutions.holistic # Holistic models
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

# Define a function to detect key points
def mediapipe_detection(image, model):
    image =cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR TO RGB
    image.flags.writeable = False
    results = model.process(image)                 # Make prediction
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB TO BGR
    return image, results


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_CONTOURS,
                             mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
                             mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
                             )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                             )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                             )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                             mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                             mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                             )
    
def extract_keypoints(results):
    # Pose
    if results.pose_landmarks:
        pose_landmarks = np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in results.pose_landmarks.landmark]).flatten() # Reshape to a single dimension array
    else:
        pose_landmarks = np.zeros(33*4)

    # Face
    if results.face_landmarks:
        face_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.face_landmarks.landmark]).flatten()
    else:
        face_landmarks = np.zeros(468*3)

    # Left Hand
    if results.left_hand_landmarks:
        left_hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand_landmarks = np.zeros(21*3)

    # Right Hand
    if results.right_hand_landmarks:
        right_hand_landmarks = np.array([[landmark.x, landmark.y, landmark.z] for landmark in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand_landmarks = np.zeros(21*3)

    return np.concatenate([pose_landmarks, face_landmarks, left_hand_landmarks, right_hand_landmarks])

