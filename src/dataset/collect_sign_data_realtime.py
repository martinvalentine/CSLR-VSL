import cv2
import os
import time

# Path to store videos
IMAGES_PATH = "/home/martinvalentine/Desktop/sign-language-lstm/data/raw/person2"

# Labels (categories for data)
labels = ['a', 'ă', 'â', 'b', 'c', 'd', 'đ', 'e', 'ê']
# labels = ['f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n']
# labels = ['o', 'ô', 'p', 'q', 'r', 's', 't', 'u', 'v']
# labels = ['w', 'x', 'y', 'z']

# Video settings
video_duration = 5  # Record each video for 5 seconds
fps = 30  # Frames per second
frame_width = 640
frame_height = 480

# Create folders for each label
for label in labels:
    label_path = os.path.join(IMAGES_PATH, label)
    os.makedirs(label_path, exist_ok=True)

print("Folders created for each label")

# Initialize webcam
cap = cv2.VideoCapture(0)
cap.set(3, frame_width)  # Set width
cap.set(4, frame_height)  # Set height

# Capture a video for each label
for label in labels:
    print(f"Recording video for label: {label}")
    time.sleep(2)  # Pause before recording

    # Define video filename and path
    video_filename = os.path.join(IMAGES_PATH, label, f"{label}_{int(time.time())}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_filename, fourcc, fps, (frame_width, frame_height))

    start_time = time.time()

    while time.time() - start_time < video_duration:
        ret, frame = cap.read()
        if not ret:
            print("Error capturing frame")
            break

        out.write(frame)  # Write frame to video file

        # Show frame (optional)
        cv2.imshow("Recording", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):  # Press 'q' to quit
            break

    out.release()
    print(f"Saved video: {video_filename}")

# Release resources
cap.release()
cv2.destroyAllWindows()
print("Video collection completed")
