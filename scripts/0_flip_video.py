import cv2
import logging
from pathlib import Path
from multiprocessing import Pool, cpu_count

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def flip_video(video_path, output_path, flip_code=1):
    """
    Flip a single video and save it to the output path.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logging.error(f"Failed to open video: {video_path}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path = output_path.with_suffix(".mp4")

    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        flipped_frame = cv2.flip(frame, flip_code)
        out.write(flipped_frame)

    cap.release()
    out.release()
    logging.info(f"Flipped video saved: {output_path}")


def process_all_videos(input_dir, output_dir):
    """
    Process and flip all video files inside signer folders under sentence folders.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    video_files = list(input_dir.rglob("*.MOV")) + list(input_dir.rglob("*.mp4"))
    logging.info(f"Found {len(video_files)} videos to process...")

    tasks = []
    for video_path in video_files:
        relative_path = video_path.relative_to(input_dir)
        output_video_path = output_dir / relative_path
        tasks.append((video_path, output_video_path))

    with Pool(processes=max(1, cpu_count() - 1)) as pool:
        pool.starmap(flip_video, tasks)


if __name__ == "__main__":
    input_video_dir = "/home/martinvalentine/Desktop/CSLR-VSL/data/raw/VSL/minnor"
    output_video_dir = "/home/martinvalentine/Desktop/CSLR-VSL/data/raw/VSL/flipped_videos"

    process_all_videos(input_video_dir, output_video_dir)
    logging.info("All videos processed and flipped successfully!")
