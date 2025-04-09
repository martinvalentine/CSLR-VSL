import cv2
from pathlib import Path
from collections import defaultdict

def get_video_duration(video_path):
    """
    Return duration of video in seconds (float).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return frames / fps if fps > 0 else 0.0

def collect_video_info(parent_dir, extensions=(".mov", ".mp4")):
    """
    Walk Sentence→Signer→videos and collect durations.
    Returns:
      - video_lengths[sentence][signer][video_filename] = duration_in_seconds
      - total_video_count
    """
    parent = Path(parent_dir)
    exts = {e.lower() for e in extensions}

    video_lengths = defaultdict(lambda: defaultdict(dict))
    total_video_count = 0

    for sentence_dir in parent.iterdir():
        if not sentence_dir.is_dir():
            continue
        sentence = sentence_dir.name

        for signer_dir in sentence_dir.iterdir():
            if not signer_dir.is_dir():
                continue
            signer = signer_dir.name

            for f in signer_dir.iterdir():
                if not f.is_file() or f.suffix.lower() not in exts:
                    continue
                dur = get_video_duration(f)
                video_lengths[sentence][signer][f.name] = dur
                total_video_count += 1

    return video_lengths, total_video_count

if __name__ == "__main__":
    base_path = "/home/martinvalentine/Desktop/CSLR-VSL/data/raw/VSL/VSL_Benchmark"

    splits = ["train", "test", "dev"]

    grand_total = 0

    for split in splits:
        print(f"\n=== Processing split: {split.upper()} ===")
        split_path = Path(base_path) / split
        if not split_path.exists():
            print(f"  Split folder '{split}' does not exist at path: {split_path}")
            continue

        lengths, total_videos = collect_video_info(split_path)

        print("=== Video Lengths (MM:SS) per Sentence → Signer → Video ===")
        for sentence in sorted(lengths):
            print(f"\n{sentence}:")
            for signer in sorted(lengths[sentence]):
                print(f"  {signer}:")
                for video, dur in lengths[sentence][signer].items():
                    mins, secs = divmod(int(dur), 60)
                    print(f"    {video:20s} : {mins:02d}:{secs:02d}")

        print(f"\n=== Unique Sentences in {split.upper()} ===")
        unique_sentences = sorted(lengths.keys())
        print("| ID | Sentence |")
        print("|----|----------|")
        for idx, s in enumerate(unique_sentences, start=1):
            print(f"| {idx}  | {s} |")

        print(f"\n=== Total Videos in {split.upper()}: {total_videos} ===")
        grand_total += total_videos

    print(f"\n=== Grand Total Videos Across All Splits: {grand_total} ===")
