import os
import shutil
from collections import defaultdict

#TODO: ADJUST YOUR PATH TO THE VIDEO FILES BEFORE RUNNING THIS FILE

# Define paths
source_root = "/home/martinvalentine/Desktop/CSLR-VSL/data/raw/VSL/full_clean_sentences"
dest_root = "/home/martinvalentine/Desktop/CSLR-VSL/data/raw/VSL/VSL_Benchmark"

splits = {
     "train": lambda signer, video: video.startswith("video1.") or video.startswith("video2."),
     "test": lambda signer, video: signer in ["Signer1", "Signer2"] and video.startswith("video3."),
     "dev": lambda signer, video: signer in ["Signer3", "Signer4"] and video.startswith("video3.")
 }

# Track copy stats
copy_counts = defaultdict(int)
sentence_counts = defaultdict(lambda: defaultdict(int))

print("Starting scripts split and copy process...\n")

# DEBUG: Print the sentences found in the source directory and videos inside each signer folder
# print("Sentences and videos in source directory:")
# for sentence in sorted(os.listdir(source_root)):
#     sentence_path = os.path.join(source_root, sentence)
#     if not os.path.isdir(sentence_path):
#         continue
#     print(f"- Sentence: {sentence}")
#
#     for signer in sorted(os.listdir(sentence_path)):
#         signer_path = os.path.join(sentence_path, signer)
#         if not os.path.isdir(signer_path):
#             continue
#         print(f"  - Signer: {signer}")
#
#         videos = sorted([
#             f for f in os.listdir(signer_path)
#             if os.path.isfile(os.path.join(signer_path, f)) and not f.startswith('.')
#         ])
#         if videos:
#             for video in videos:
#                 print(f"    - {video}")
#         else:
#             print("    (No videos found)")

# Loop through each sentence folder
for sentence in sorted(os.listdir(source_root)):
    sentence_path = os.path.join(source_root, sentence)
    if not os.path.isdir(sentence_path):
        continue

    print(f"Processing sentence: {sentence}")

    for signer in sorted(os.listdir(sentence_path)):
        signer_path = os.path.join(sentence_path, signer)
        if not os.path.isdir(signer_path):
            continue

        for video in sorted(os.listdir(signer_path)):
            video_path = os.path.join(signer_path, video)

            # Determine split
            for split_name, condition in splits.items():
                if condition(signer, video):
                    try:
                        dest_path = os.path.join(dest_root, split_name, sentence, signer)
                        os.makedirs(dest_path, exist_ok=True)
                        shutil.copy2(video_path, os.path.join(dest_path, video))
                        print(f"[{split_name.upper()}] Copied: {video_path} -> {dest_path}")
                        copy_counts[split_name] += 1
                        sentence_counts[split_name][sentence] += 1
                    except Exception as e:
                        print(f"[ERROR] Failed to copy {video_path} to {dest_path}: {e}")
                    break  # Stop checking other splits

print("\nFinished copying files.")
print("\n=== Summary ===")
for split in ["train", "test", "dev"]:
    print(f"\n{split.upper()} - Total Videos: {copy_counts[split]}")
    for sentence, count in sentence_counts[split].items():
        print(f"  {sentence}: {count} file(s)")
