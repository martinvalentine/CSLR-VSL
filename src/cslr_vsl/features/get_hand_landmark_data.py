import mediapipe as mp
import cv2
import numpy as np
import os
import argparse
from collections import defaultdict
from tqdm import tqdm
import csv
import multiprocessing
# No longer need functools.partial in this new structure

# Set up mediapipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils # Not directly used in extraction but good to keep if needed later

# Global variable to hold the Holistic model in each worker process
# This will be initialized by the pool's initializer
holistic_model = None

def worker_init():
    """
    Initializes the MediaPipe Holistic model in each worker process.
    This function is called once per worker when the pool starts.
    """
    global holistic_model
    # Set detection and tracking confidence as needed
    holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

def mediapipe_detection(image, model):
    """
    Processes an image with the provided MediaPipe model.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = model.process(image)
    image.flags.writeable = True
    # No need to convert back to BGR as we only need results, not the annotated image
    return results

def extract_keypoints(results):
    """
    Extracts left hand and right hand landmarks from MediaPipe results.
    Handles cases where hands are not detected by returning zeros.
    Returns a concatenated numpy array of landmarks.
    """
    left_missing = results.left_hand_landmarks is None
    right_missing = results.right_hand_landmarks is None

    # Left hand landmarks
    if not left_missing:
        # Extract x, y, z. Use zeros if left hand is missing.
        left_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.left_hand_landmarks.landmark]).flatten()
    else:
        left_hand = np.zeros(21 * 3, dtype=np.float32)  # 21 hand landmarks, 3 values (x, y, z)

    # Right hand landmarks
    if not right_missing:
        # Extract x, y, z. Use zeros if right hand is missing.
        right_hand = np.array([[lm.x, lm.y, lm.z] for lm in results.right_hand_landmarks.landmark]).flatten()
    else:
        right_hand = np.zeros(21 * 3, dtype=np.float32)  # 21 hand landmarks, 3 values (x, y, z)

    # Return the concatenated keypoints (left hand + right hand)
    return np.concatenate([left_hand, right_hand])

# Other functions (worker_init, mediapipe_detection, etc.) remain the same as in your code.

def process_single_frame(args):
    """
    Processes a single frame to extract keypoints using the initialized Holistic model.
    This function is designed to be run by a worker process.
    Args:
        args (tuple): A tuple containing (input_path, output_path, vertical_flip, folder_key_for_summary).
    Returns:
        tuple: (folder_key_for_summary, is_zero). folder_key is used to aggregate results in the main process.
    """
    input_path, output_path, vertical_flip, folder_key = args
    is_zero = True # Assume zeroed initially in case of error

    try:
        global holistic_model
        if holistic_model is None:
             # Fallback: Should not happen with initializer, but good practice
             holistic_model = mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5)

        frame = cv2.imread(input_path)
        if frame is None:
             print(f"Warning: Could not read image file {input_path}")
             return folder_key, is_zero # Return zeroed status

        if vertical_flip:
            frame = cv2.flip(frame, 0)

        # Use the model initialized in worker_init
        results = mediapipe_detection(frame, holistic_model)
        keypoints = extract_keypoints(results)

        np.save(output_path, keypoints)
        is_zero = np.all(keypoints == 0)

    except Exception as e:
        # Log the error but continue processing other frames
        print(f"Error processing {input_path}: {e}")
        is_zero = True # Treat errors as zeroed/failed frames

    return folder_key, is_zero

def process_videos(input_folder, output_folder, vertical_flip=False, summary_path='summary.csv', max_workers=None):
    """
    Orchestrates the parallel processing of frames.
    Finds all image files, creates output paths, and uses a multiprocessing pool
    to process each frame concurrently.
    """
    # Find all frame files and prepare task arguments
    all_frame_tasks = []
    print(f"Scanning input folder: {input_folder}")
    for root, _, files in os.walk(input_folder):
        frame_files = sorted([f for f in files if f.endswith('.jpg')])
        if frame_files:
            relative_folder = os.path.relpath(root, input_folder)
            output_dir = os.path.join(output_folder, relative_folder)
            os.makedirs(output_dir, exist_ok=True) # Ensure output directory exists

            # Create a key for the summary based on the last two folder levels
            # Handles cases where relative_folder is just a single level or empty
            folder_key_parts = os.path.normpath(relative_folder).split(os.sep)
            folder_key = os.path.join(*folder_key_parts[-2:]) if len(folder_key_parts) > 1 else folder_key_parts[-1] if folder_key_parts else 'root'


            for file in frame_files:
                input_path = os.path.join(root, file)
                output_path = os.path.join(output_dir, file.replace('.jpg', '.npy'))
                # Append the task arguments as a tuple
                all_frame_tasks.append((input_path, output_path, vertical_flip, folder_key))

    if not all_frame_tasks:
        print("No image files found in the input folder.")
        return

    print(f"Found {len(all_frame_tasks)} frames to process.")

    # Determine the number of worker processes
    if max_workers is None:
        # Use a conservative number of workers to avoid freezing the PC
        # For example, half of the logical cores, but at least 1
        num_workers = max(multiprocessing.cpu_count() // 2, 1)
        print(f"Using default max_workers: {num_workers}")
    else:
         num_workers = max_workers
         print(f"Using specified max_workers: {num_workers}")


    summary_dict = defaultdict(lambda: {'total': 0, 'saved': 0, 'zeroed': 0})

    # Create and run the multiprocessing pool
    # The 'initializer=worker_init' ensures holistic_model is created in each worker
    with multiprocessing.Pool(processes=num_workers, initializer=worker_init) as pool:
        # Use imap_unordered to get results as they complete
        for folder_key, is_zero in tqdm(pool.imap_unordered(process_single_frame, all_frame_tasks), total=len(all_frame_tasks), desc="Processing Frames"):
            # Aggregate results based on the folder key
            summary_dict[folder_key]['total'] += 1
            if is_zero:
                summary_dict[folder_key]['zeroed'] += 1
            else:
                summary_dict[folder_key]['saved'] += 1

    print("\nFinished processing frames.")

    # Save summary
    try:
        with open(summary_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Try to infer header based on key structure, or use a generic one
            header = ['Folder', 'Total Frames', 'Valid Frames', 'Zero Frames']
            if all(os.sep in key for key in summary_dict.keys()):
                 header = ['Split', 'Sentence', 'Total Frames', 'Valid Frames', 'Zero Frames']

            writer.writerow(header)

            # Sort keys for consistent summary output
            for key in sorted(summary_dict.keys()):
                stats = summary_dict[key]
                row = [key, stats['total'], stats['saved'], stats['zeroed']]
                if os.sep in key and len(key.split(os.sep)) == 2:
                     row = key.split(os.sep) + [stats['total'], stats['saved'], stats['zeroed']]
                writer.writerow(row)

        print(f"[INFO] Summary saved to {summary_path}")

    except Exception as e:
        print(f"Error saving summary to {summary_path}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extracts MediaPipe keypoints from image frames in parallel.")
    parser.add_argument('--input_folder', type=str, required=True,
                        help='Root folder containing the image frames (e.g., structure like split/sentence/frame.jpg).')
    parser.add_argument('--output_folder', type=str, required=True,
                        help='Root folder to save the extracted keypoints (.npy files). Output structure mirrors input.')
    parser.add_argument('--vertical_flip', action='store_true',
                        help='Apply vertical flip to frames before processing.')
    parser.add_argument('--summary_path', type=str, default='summary.csv',
                        help='Path to save the processing summary CSV file.')
    parser.add_argument('--max_workers', type=int, default=None,
                        help='Maximum number of worker processes to use. Defaults to half of CPU cores.')
    args = parser.parse_args()

    # Ensure the output folder exists before starting
    os.makedirs(args.output_folder, exist_ok=True)

    process_videos(
        input_folder=args.input_folder,
        output_folder=args.output_folder,
        vertical_flip=args.vertical_flip,
        summary_path=args.summary_path,
        max_workers=args.max_workers # Pass the argument
    )