import cv2
from pathlib import Path
from collections import defaultdict
import re


# TODO: Adjust your path to the video files before running this file
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


def collect_video_info(parent_dir, extensions=(".mov", ".mp4", ".MOV")):
    """
    Walk Sentence→Signer→videos and collect durations.
    Returns:
      - video_lengths[sentence][signer][video_filename] = duration_in_seconds
      - videos_with_invalid_format = list of videos that don't match the expected pattern
    """
    parent = Path(parent_dir)
    exts = {e.lower() for e in extensions}

    expected_pattern = re.compile(r"^Signer[A-Z0-9]+_.+_\d+\.(mp4|mov)$", re.IGNORECASE)

    video_lengths = defaultdict(lambda: defaultdict(dict))
    signer_video_count = defaultdict(lambda: defaultdict(int))  # Track video count for each signer
    total_video_count = 0
    videos_with_invalid_format = []  # Store videos that don't match the expected pattern

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
                # Check if the video filename matches the expected pattern
                if not expected_pattern.match(f.name):
                    videos_with_invalid_format.append(f)
                    continue
                dur = get_video_duration(f)
                video_lengths[sentence][signer][f.name] = dur
                signer_video_count[sentence][signer] += 1
                total_video_count += 1

    return video_lengths, signer_video_count, total_video_count, videos_with_invalid_format


def group_sentences_by_signer(video_lengths, signer_video_count):
    """
    Groups sentences by the combination of signers, disregarding the order.
    Returns:
      - grouped_sentences[signer_group] = [sentences]
    """
    grouped_sentences = defaultdict(list)

    for sentence, signers in signer_video_count.items():
        signer_group = frozenset(signers.keys())  # Use frozenset to group by signer set (ignoring order)
        grouped_sentences[signer_group].append(sentence)

    return grouped_sentences


if __name__ == "__main__":
    root = "/home/martinvalentine/Desktop/CSLR-VSL/data/raw/Sample_V1"
    # root = "/home/martinvalentine/Desktop/VSL/VSL_Benchmark"
    lengths, signer_video_count, total_videos, invalid_videos = collect_video_info(root)

    # 1) Detailed print of every video length and frequency of each signer per sentence
    print("=== Video Lengths (MM:SS) per Sentence → Signer → Video ===")
    for idx, sentence in enumerate(sorted(lengths), start=1):
        print(f"\n{idx}. {sentence}:")
        for signer in sorted(lengths[sentence]):
            print(f"  {signer}: {signer_video_count[sentence][signer]} video(s):")
            for video, dur in lengths[sentence][signer].items():
                mins, secs = divmod(int(dur), 60)
                print(f"    {video:20s} : {mins:02d}:{secs:02d}")

    # 2) Markdown table of unique sentences with IDs
    unique_sentences = sorted(lengths.keys())
    print("\n=== Unique Sentences ===")
    print("| ID | Sentence |")
    print("|----|----------|")
    for idx, s in enumerate(unique_sentences, start=1):
        print(f"| {idx:2d} | {s} |")

    # 3) Group sentences by signer combinations (ignoring order)
    grouped_sentences = group_sentences_by_signer(lengths, signer_video_count)
    print("\n=== Grouped Sentences by Signer Combination ===")
    for signer_group, sentences in grouped_sentences.items():
        print(f"\nSigner Group: {', '.join(signer_group)}")
        for idx, sentence in enumerate(sentences, start=1):
            print(f"  {idx}. {sentence}")

    # 4) Total video count
    print(f"\n=== Total Videos Processed: {total_videos} ===")

    # 5) Log videos with invalid filenames
    if invalid_videos:
        print("\n=== Videos with Invalid Filename Format ===")
        for video in invalid_videos:
            print(f"  {video}")
    else:
        print("\nNo videos with invalid filename format found.")
