import os
import re
import csv
import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool, cpu_count
import logging
from functools import partial
import time

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def crop_to_square(image):
    """
    Crops an image to a 1080x1080 square, assuming the input is 1920x1080.
    Optimized to avoid unnecessary checks for most frames.
    """
    if image is None:
        return None

    # Only check dimensions on first frame or if there's an error later
    height, width = image.shape[:2]

    if width != 1920 or height != 1080:
        logging.warning(f"Image dimensions are {width}x{height}, expected 1920x1080 for cropping.")
        return None

    # Use array slicing (more efficient than creating a new array)
    return image[:, 420:1500]


def extract_frames_from_video(video_path, output_path, sample_rate=1):
    """
    Extracts and crops frames from a video at a given sample rate.

    Args:
        video_path: Path to the video file.
        output_path: Path to the directory where cropped frames will be saved.
        sample_rate: Extract every nth frame (default=1 for all frames).

    Returns:
        int: The number of frames extracted and saved.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Could not open video file: {video_path}")
        return 0

    output_path.mkdir(parents=True, exist_ok=True)

    # Read video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Preallocate arrays for better memory management
    extracted_frame_count = 0
    frames_to_extract = range(0, frame_count, sample_rate)

    # Process frames in batches for better efficiency
    batch_size = 10
    for i in range(0, len(frames_to_extract), batch_size):
        batch_indices = frames_to_extract[i:i + batch_size]

        for frame_index in batch_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                continue

            cropped_frame = crop_to_square(frame)
            if cropped_frame is not None:
                frame_filename = output_path / f"frame_{frame_index:04d}.png"
                # Use optimized writing parameters
                cv2.imwrite(
                    str(frame_filename),
                    cropped_frame,
                    [cv2.IMWRITE_PNG_COMPRESSION, 3]  # Balanced compression/speed
                )
                extracted_frame_count += 1

    cap.release()
    return extracted_frame_count


def process_single_video(args):
    """
    Processes a single video: extracts and crops frames, and returns an annotation entry.
    Modified to accept a single argument tuple for better multiprocessing compatibility.
    """
    video_file, sentence_id, signer_id, trial_index, gloss_label, output_root, split_output_path, sample_rate = args

    video_id = f"S{sentence_id:06d}_P{signer_id:04d}_T{trial_index + 1:02d}"
    video_output_path = split_output_path / video_id

    try:
        frame_count = extract_frames_from_video(video_file, video_output_path, sample_rate)
        relative_path = video_output_path.relative_to(Path(output_root))
        logging.info(f"Extracted and cropped {frame_count} frames from {video_file}")
        return [video_id, f"{relative_path}", gloss_label]
    except Exception as e:
        logging.error(f"Error processing {video_file}: {e}")
        return None


def collect_video_tasks(root_dir, output_dir, split, sample_rate=1):
    """
    Collects all video processing tasks for the given split.

    Returns:
        list: A list of task tuples for multiprocessing.
    """
    tasks = []
    root_path = root_dir / split
    split_output_path = Path(output_dir) / split

    sentence_folders = sorted([f for f in root_path.iterdir() if f.is_dir()])

    for sentence_index, sentence_folder in enumerate(sentence_folders):
        sentence_id = sentence_index + 1
        gloss_label = sentence_folder.name

        for signer_folder in sorted(sentence_folder.iterdir()):
            if not signer_folder.is_dir():
                continue

            signer_match = re.search(r'Signer(\d+)', str(signer_folder))
            signer_id = int(signer_match.group(1)) if signer_match else 0

            for trial_index, video_file in enumerate(sorted(signer_folder.iterdir())):
                if video_file.suffix.lower() in {".mp4", ".mov", ".avi"}:
                    tasks.append((
                        video_file,
                        sentence_id,
                        signer_id,
                        trial_index,
                        gloss_label,
                        output_dir,
                        split_output_path,
                        sample_rate
                    ))

    return tasks


def process_and_extract_frames_optimized(root_dir, output_dir, csv_dir, split, num_processes=None, sample_rate=1,
                                         chunk_size=1):
    """
    Optimized function to extract and crop frames from videos using dynamic workload distribution.

    Args:
        root_dir: Root directory containing the video data.
        output_dir: Directory to save extracted frames.
        csv_dir: Directory to save CSV annotations.
        split: Data split (train, test, dev).
        num_processes: Number of CPU processes to use.
        sample_rate: Extract every nth frame.
        chunk_size: Chunk size for task distribution.
    """
    start_time = time.time()

    # Create output directories
    output_path = Path(output_dir) / split
    output_path.mkdir(parents=True, exist_ok=True)
    csv_path = Path(csv_dir)
    csv_path.mkdir(parents=True, exist_ok=True)

    # Collect all tasks
    tasks = collect_video_tasks(root_dir, output_dir, split, sample_rate)
    logging.info(f"Collected {len(tasks)} videos to process for {split} split")

    # Process videos in parallel
    if num_processes is None:
        num_processes = max(1, cpu_count() - 1)  # Leave one CPU free for system tasks

    logging.info(f"Processing with {num_processes} processes")

    all_annotations = []
    with Pool(processes=num_processes) as pool:
        # Use imap_unordered for better load balancing
        for result in pool.imap_unordered(process_single_video, tasks, chunksize=chunk_size):
            if result:
                all_annotations.append(result)

    # Save annotations to CSV
    csv_file_path = csv_path / f"{split}_annotations.csv"
    with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Video_ID", "Frames_Path", "Gloss_Label"])
        writer.writerows(all_annotations)

    elapsed_time = time.time() - start_time
    logging.info(f"Split {split} processed in {elapsed_time:.2f} seconds with {len(all_annotations)} videos")
    logging.info(f"Annotations for {split} saved to {csv_file_path}")


def get_optimal_parameters(total_videos):
    """
    Calculate optimal parameters based on the dataset size and system.

    Returns:
        tuple: (chunk_size, sample_rate)
    """
    system_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 3)  # RAM in GB
    cpus = cpu_count()

    # Estimate optimal chunk size based on number of CPUs and videos
    chunk_size = max(1, min(5, total_videos // (cpus * 10)))

    # Determine if we should sample frames based on system memory
    if system_memory < 8:  # Less than 8GB RAM
        sample_rate = 2  # Extract every other frame
    else:
        sample_rate = 1  # Extract all frames

    return chunk_size, sample_rate


def count_videos(root_dir, split):
    """
    Counts the total number of video files in the specified split.
    """
    count = 0
    split_path = root_dir / split
    if not split_path.exists():
        return 0

    for sentence_folder in split_path.iterdir():
        if sentence_folder.is_dir():
            for signer_folder in sentence_folder.iterdir():
                if signer_folder.is_dir():
                    for video_file in signer_folder.iterdir():
                        if video_file.suffix.lower() in {".mp4", ".mov", ".avi"}:
                            count += 1
    return count


if __name__ == "__main__":
    # Configure paths
    root_dir = Path("../../data/raw/VSL/VSL_Benchmark")
    output_dir = Path("../../data/interim/VSL_Benchmark")
    csv_dir = Path("../../data/processed/VSL_Benchmark/csv")

    # Count total videos to determine optimal parameters
    total_videos = sum(count_videos(root_dir, split) for split in ["train", "test", "dev"])
    chunk_size, sample_rate = get_optimal_parameters(total_videos)

    # Determine optimal number of processes (leave some resources for the system)
    available_cpus = cpu_count()
    num_processes = max(1, available_cpus - 1)

    logging.info(
        f"Starting processing with {num_processes} processes, chunk size {chunk_size}, sample rate {sample_rate}")

    # Process each split
    for split in ["train", "test", "dev"]:
        if (root_dir / split).exists():
            process_and_extract_frames_optimized(
                root_dir,
                output_dir,
                csv_dir,
                split,
                num_processes=num_processes,
                sample_rate=sample_rate,
                chunk_size=chunk_size
            )
        else:
            logging.warning(f"Split directory {split} not found, skipping")

    logging.info("Frame extraction and annotation saving complete for all splits!")