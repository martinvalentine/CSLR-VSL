import os
import re
import csv
import cv2
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def crop_to_square(image):
    """
    Crops an image to a 1080x1080 square, assuming the input is 1920x1080.

    Args:
        image (numpy.ndarray): The input image.

    Returns:
        numpy.ndarray or None: The cropped image, or None if input dimensions are incorrect.
    """
    if image is None:
        logging.error("Could not load image for cropping.")
        return None

    height, width = image.shape[:2]

    if width != 1920 or height != 1080:
        logging.error(f"Image dimensions are {width}x{height}, expected 1920x1080 for cropping.")
        return None

    # Crop 420 pixels from both left and right sides
    cropped_img = image[:, 420:1500]  # Keep height unchanged, crop width
    return cropped_img

def extract_frames_from_video(video_path: Path, output_path: Path):
    """
    Extracts and crops all frames from a single video.

    Args:
        video_path: Path to the video file.
        output_path: Path to the directory where cropped frames will be saved.

    Returns:
        int: The number of frames extracted and saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return 0

    output_path.mkdir(parents=True, exist_ok=True)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    extracted_frame_count = 0
    for frame_index in range(frame_count):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            logging.warning(f"Could not read frame {frame_index} from {video_path}")
            continue

        cropped_frame = crop_to_square(frame)
        if cropped_frame is not None:
            frame_filename = output_path / f"frame_{frame_index:04d}.png"
            cv2.imwrite(str(frame_filename), cropped_frame)
            extracted_frame_count += 1

    cap.release()
    return extracted_frame_count

def process_single_video(video_file: Path, sentence_id: int, signer_id: int, trial_index: int, gloss_label: str, output_root: Path, split_output_path: Path):
    """
    Processes a single video: extracts and crops frames, and returns an annotation entry.

    Args:
        video_file: Path to the video file.
        sentence_id: ID of the sentence.
        signer_id: ID of the signer.
        trial_index: Index of the trial.
        gloss_label: Gloss label for the sentence.
        output_root: Root output directory.
        split_output_path: Output path for the current split.

    Returns:
        list: An annotation entry [video_id, frames_path, gloss_label] or None if processing fails.
    """
    video_id = f"S{sentence_id:06d}_P{signer_id:04d}_T{trial_index+1:02d}"
    video_output_path = split_output_path / video_id

    try:
        frame_count = extract_frames_from_video(video_file, video_output_path)
        relative_path = video_output_path.relative_to(output_root)
        logging.info(f"Extracted and cropped {frame_count} frames from {video_file} → {video_output_path}")
        return [video_id, f"{relative_path}", gloss_label]
    except Exception as e:
        logging.error(f"Error processing {video_file}: {e}")
        return None

def process_sentence_folder(sentence_folder: Path, sentence_index: int, output_dir: Path, split: str):
    """
    Processes all signer folders and video files within a sentence folder.

    Args:
        sentence_folder: Path to the sentence folder.
        sentence_index: Index of the sentence.
        output_dir: Root output directory.
        split: The current split (e.g., "train", "test").

    Returns:
        list: A list of annotation entries for the videos in this sentence folder.
    """
    annotations = []
    sentence_id = sentence_index + 1
    gloss_label = sentence_folder.name
    split_output_path = Path(output_dir) / split

    for signer_folder in sorted(sentence_folder.iterdir()):
        if not signer_folder.is_dir():
            continue

        signer_match = re.search(r'Signer(\d+)', str(signer_folder))
        signer_id = int(signer_match.group(1)) if signer_match else 0

        for trial_index, video_file in enumerate(sorted(signer_folder.iterdir())):
            if video_file.suffix.lower() in {".mp4", ".mov", ".avi"}:
                annotation = process_single_video(video_file, sentence_id, signer_id, trial_index, gloss_label, output_dir, split_output_path)
                if annotation:
                    annotations.append(annotation)
    return annotations

def process_and_extract_frames_optimized(root_dir: Path, output_dir: Path, csv_dir: Path, split: str, num_processes: int = None):
    """
    Optimized function to extract and crop frames from videos and generate an annotation CSV file using multiprocessing.
    """
    root_path = root_dir / split
    output_path = output_dir / split
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir
    csv_path.mkdir(parents=True, exist_ok=True)

    all_annotations = []
    sentence_folders = sorted([f for f in root_path.iterdir() if f.is_dir()])

    if num_processes is None:
        num_processes = cpu_count()

    with Pool(processes=num_processes) as pool:
        tasks = []
        for sentence_index, sentence_folder in enumerate(sentence_folders):
            tasks.append((sentence_folder, sentence_index, str(output_dir), split)) # Pass output_dir as string for multiprocessing

        results = pool.starmap(process_sentence_folder, tasks)
        for annotations in results:
            all_annotations.extend(annotations)

    # Save annotations to CSV
    csv_file_path = csv_path / f"{split}_annotations.csv"
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Video_ID", "Frames_Path", "Gloss_Label"])
        writer.writerows(all_annotations)
    logging.info(f"Annotations for {split} saved to {csv_file_path}")

def count_train_videos(root_dir: Path):
    """
    Counts the total number of video files in the training split.
    """
    count = 0
    train_path = root_dir / "train"
    for sentence_folder in train_path.iterdir():
        if sentence_folder.is_dir():
            for signer_folder in sentence_folder.iterdir():
                if signer_folder.is_dir():
                    for video_file in signer_folder.iterdir():
                        if video_file.suffix.lower() in {".mp4", ".mov", ".avi"}:
                            count += 1
    return count

if __name__ == "__main__":
    root_dir = Path("../../data/raw/VSL_Sample")
    output_dir = Path("../../data/interim/VSL_Sample") # Changed output directory to indicate cropped images
    csv_dir = Path("../../data/processed/VSL_Sample/csv")

    # Determine the number of processes to use
    num_processes = cpu_count()

    for split in ["train", "test", "dev"]:
        process_and_extract_frames_optimized(root_dir, output_dir, csv_dir, split, num_processes)

    logging.info("Frame extraction and annotation saving complete for all splits!")