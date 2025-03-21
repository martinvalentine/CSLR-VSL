import cv2
import numpy as np
from src.utils import mediapipe_utils as mp_utils
from src.models import model3_exp as model3

# Load the models
model = model3.load_model(model3.model, '/home/martinvalentine/Desktop/sign-language-lstm/models/exp3.h5')

actions = np.array(['hello', 'thanks', 'goodbye', 'me', 'you'])

# Start realtime video prediction
# 1. NEW detection variables
sequence = [] # For storing 30 frames in order to make a prediction on
sentence = [] # Concatenate detections history
threshold = 0.8

# Apply the function to the webcam
cap = cv2.VideoCapture(0)
# Set mediapipe models
with mp_utils.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        # Read feed
        ret, frame = cap.read()

        # Make detections
        image, results = mp_utils.mediapipe_detection(frame, holistic)
        print(results)

        # Draw landmarks
        mp_utils.draw_styled_landmarks(image, results)

        # 2. PREDICTION LOGIC
        keypoints = mp_utils.extract_keypoints(results)
        # sequence.append(keypoints)
        sequence.insert(0,keypoints)
        # sequence = sequence[-30:] # Grab the last 30 frames to make prediction
        sequence = sequence[:40]

        if len(sequence) == 40:
            input_sequence =  np.expand_dims(sequence, axis=0) # Explain in bellow
            print("Model input shape:", input_sequence.shape)  # (1, 30, 1662)

            # Predict sign language action
            res = model.predict(input_sequence)

            # Reshape output if necessary
            res = res[0]  # Extract first batch prediction
            res = res.flatten()  # Ensure it is a 1D array

            # Determine the most likely action
            predicted_action = actions[np.argmax(res)]
            print("Predicted action:", predicted_action)

        # 3. VISUALIZATION LOGIC
            if res[np.argmax(res)] > threshold:
                if len(sentence) > 0:
                    if actions[np.argmax(res)] != sentence[-1]:  # Avoid consecutive duplicates
                        sentence.append(actions[np.argmax(res)])
                else:
                    sentence.append(actions[np.argmax(res)])

                if len(sentence) > 5:
                    sentence = sentence[-5:]  # Keep only the last 5 words

        cv2.rectangle(image, (0,0), (640,40), (245, 117, 16), -1)
        cv2.putText(image, ' '.join(sentence), (3, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        # Show to screen
        cv2.imshow('Raw Webcam Feed', image)

        # Break if 'q' is pressed
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
