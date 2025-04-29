import os
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.framework.formats.landmark_pb2 import NormalizedLandmark, NormalizedLandmarkList

def draw_hand_landmarks_on_image(image, landmarks, is_right=True):
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands

    hand_landmarks = [
        NormalizedLandmark(x=landmarks[i], y=landmarks[i+1], z=landmarks[i+2])
        for i in range(0, len(landmarks), 3)
    ]
    hand_landmarks_proto = NormalizedLandmarkList(landmark=hand_landmarks)

    mp_drawing.draw_landmarks(
        image,
        hand_landmarks_proto,
        mp_hands.HAND_CONNECTIONS,
        mp_drawing.DrawingSpec(color=(0, 255, 0) if is_right else (0, 0, 255), thickness=2, circle_radius=2),
        mp_drawing.DrawingSpec(color=(255, 255, 255), thickness=2),
    )

def visualize_landmarks_on_images(img_folder_path, npy_folder_path, result_folder_root_path):
    folder_name = os.path.basename(os.path.normpath(img_folder_path))
    result_folder_path = os.path.join(result_folder_root_path, folder_name + "_visualize")
    os.makedirs(result_folder_path, exist_ok=True)

    for fname in sorted(os.listdir(img_folder_path)):
        if not fname.endswith(".jpg"):
            continue

        img_path = os.path.join(img_folder_path, fname)
        npy_path = os.path.join(npy_folder_path, fname.replace(".jpg", ".npy"))
        out_path = os.path.join(result_folder_path, fname)

        image = cv2.imread(img_path)
        if image is None:
            print(f"[SKIP] Can't read image: {img_path}")
            continue

        if not os.path.exists(npy_path):
            print(f"[SKIP] Missing .npy file: {npy_path} → saving raw image.")
            cv2.imwrite(out_path, image)
            continue

        keypoints = np.load(npy_path)

        if np.all(keypoints == 0):
            print(f"[SKIP] Zero landmarks: {npy_path} → saving raw image.")
            cv2.imwrite(out_path, image)
            continue

        try:
            keypoints = keypoints.reshape(2, 21, 3)
        except:
            print(f"[SKIP] Invalid shape in file: {npy_path} → saving raw image.")
            cv2.imwrite(out_path, image)
            continue

        draw_hand_landmarks_on_image(image, keypoints[0].flatten(), is_right=False)  # Left
        draw_hand_landmarks_on_image(image, keypoints[1].flatten(), is_right=True)   # Right
        cv2.imwrite(out_path, image)

    print(f"[DONE] Visualization completed for: {result_folder_path}")

if __name__ == "__main__":
    img_folder_path = "/home/martinvalentine/Desktop/CSLR-VSL/data/interim/VSL_no_resize/frames/VSL_Benchmark/train/Signer3_Bạn-đọc_sách-thích_2"
    npy_folder_path = "/home/martinvalentine/Desktop/CSLR-VSL/data/processed/VSL_Benchmark_landmarks/train/Signer3_Bạn-đọc_sách-thích_2"
    result_folder_root_path = "/home/martinvalentine/Desktop/CSLR-VSL/data/processed/VSL_Benchmark_landmarks/mediapipe_visualize"

    visualize_landmarks_on_images(img_folder_path, npy_folder_path, result_folder_root_path)
