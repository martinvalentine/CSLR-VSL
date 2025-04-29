import cv2
from pathlib import Path
from collections import defaultdict
import re

def get_video_duration(video_path):
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return 0.0
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    frames = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0
    cap.release()
    return frames / fps if fps > 0 else 0.0

def numeric_key(filename):
    nums = re.findall(r"(\d+)", filename)
    return int(nums[-1]) if nums else 0

def collect_video_info(parent_dir, extensions=(".mov", ".mp4")):
    parent = Path(parent_dir)
    exts = {e.lower() for e in extensions}

    video_lengths = defaultdict(dict)
    total_video_count = 0
    video_names = set()
    mismatched_filenames = []

    expected_pattern = re.compile(r"^Signer[A-Z0-9]+_.+_\d+\.(mp4|mov)$", re.IGNORECASE)

    for sentence_dir in sorted(parent.iterdir()):
        if not sentence_dir.is_dir():
            continue
        sentence = sentence_dir.name

        for f in sentence_dir.iterdir():
            if not f.is_file() or f.suffix.lower() not in exts:
                continue

            if not expected_pattern.match(f.name):
                mismatched_filenames.append(str(f.relative_to(parent)))

            dur = get_video_duration(f)
            video_lengths[sentence][f.name] = dur
            total_video_count += 1
            video_names.add(f.name)

    return video_lengths, total_video_count, video_names, mismatched_filenames

def extract_signer_name(video_filename):
    match = re.match(r"^(Signer[A-Z0-9]+)_", video_filename, re.IGNORECASE)
    return match.group(1) if match else "UNKNOWN"

if __name__ == "__main__":
    base_path = "/home/martinvalentine/Desktop/CSLR-VSL/data/raw/VSL_V2"
    splits = ["train", "test", "dev"]
    grand_total = 0
    all_video_names = defaultdict(set)
    all_mismatches = []
    signer_split_counts = defaultdict(lambda: defaultdict(int))

    for split in splits:
        print(f"\n=== Processing split: {split.upper()} ===")
        split_path = Path(base_path) / split
        if not split_path.exists():
            print(f"  Split folder '{split}' does not exist at path: {split_path}")
            continue

        lengths, total_videos, video_names, mismatches = collect_video_info(split_path)

        # 1) Detailed print of every video length
        print("=== Video Lengths (MM:SS) per Sentence → Video ===")
        for sentence in sorted(lengths):
            print(f"\n{sentence}:")
            for video in sorted(lengths[sentence], key=numeric_key):
                dur = lengths[sentence][video]
                mins, secs = divmod(int(dur), 60)
                print(f"    {video:30s} : {mins:02d}:{secs:02d}")

                signer = extract_signer_name(video)
                signer_split_counts[signer][split] += 1

        # 2) Markdown table of unique sentences with IDs
        unique_sentences = sorted(lengths.keys())
        print(f"\n=== Unique Sentences in {split.upper()} ===")
        print("| ID | Sentence |")
        print("|----|----------|")
        for idx, s in enumerate(unique_sentences, start=1):
            print(f"| {idx}  | {s} |")

        # 3) Total video count for this split
        print(f"\n=== Total Videos in {split.upper()}: {total_videos} ===")
        grand_total += total_videos

        # 4) Track video names by split
        for vname in video_names:
            all_video_names[vname].add(split)

        # 5) Accumulate mismatches
        if mismatches:
            all_mismatches.extend((split, m) for m in mismatches)

    print(f"\n=== Grand Total Videos Across All Splits: {grand_total} ===")

    # Final: Print signer video counts
    print("\n=== Number of Videos per Signer per Split (Sorted by Total Descending) ===")
    print("| Signer | Train | Test | Dev | Total |")
    print("|--------|-------|------|-----|-------|")

    signer_rows = []
    for signer in signer_split_counts:
        train_count = signer_split_counts[signer].get('train', 0)
        test_count = signer_split_counts[signer].get('test', 0)
        dev_count = signer_split_counts[signer].get('dev', 0)
        total = train_count + test_count + dev_count
        signer_rows.append((signer, train_count, test_count, dev_count, total))

    signer_rows.sort(key=lambda x: x[4], reverse=True)

    for signer, train_count, test_count, dev_count, total in signer_rows:
        print(f"| {signer:6s} | {train_count:5d} | {test_count:4d} | {dev_count:3d} | {total:5d} |")
    
    print("\n")
