import cv2
import os

# Input folder containing videos
input_folder = "/home/martinvalentine/Desktop/sign-language-lstm/data/raw/person2/a"
output_base_folder = "output_frames"

# Get all video files in the folder
video_files = [f for f in os.listdir(input_folder) if f.endswith(('.mp4', '.avi', '.mkv'))]
video_files.sort()  # Sort to maintain order

print(f"Found {len(video_files)} videos in '{input_folder}'")

# Process each video
for idx, video_file in enumerate(video_files):
    video_path = os.path.join(input_folder, video_file)
    output_folder = os.path.join(output_base_folder, str(idx))  # Create numbered folder for each video
    os.makedirs(output_folder, exist_ok=True)

    print(f"Processing Video {idx}: {video_file}")

    # Open video
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get FPS (normally 30 FPS)
    print(f"  FPS: {fps}")

    frame_count = 0
    saved_frame_count = 0

    # Read and save every 2nd frame
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break  # Stop if no more frames

        if frame_count % 3 == 0:  # Capture every 3rd frame
            frame_filename = os.path.join(output_folder, f"{saved_frame_count}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_frame_count += 1

        frame_count += 1  # Always increment the frame counter

    # Release resources
    cap.release()
    print(f"Saved {saved_frame_count} images to '{output_folder}'")

print("All videos processed successfully!")
