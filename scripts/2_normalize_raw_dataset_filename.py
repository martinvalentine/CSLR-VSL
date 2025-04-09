import os
import shutil  # For future use if needed

'''
What this code does:
1. Normalizes folder names:
   - Removes leading/trailing whitespace.
   - Replaces spaces with dashes (-).
   - Preserves underscores (_).
   - Applies to both sentence and signer folders.
2. Renames video files inside each signer folder to follow a consistent format: video1.ext, video2.ext, etc.
'''

#TODO: ADJUST YOUR PATH TO THE VIDEO FILES BEFORE RUNNING THIS FILE

# Path to your data
root_path = "/home/martinvalentine/Desktop/CSLR-VSL/data/raw/VSL/full_clean_sentences"
print(f"Starting normalization process in: {root_path}")
print("-" * 40)

# Step 1: Rename Sentence and Signer Directories
print("Step 1: Checking and renaming sentence and signer directories...")
renamed_sentence_count = 0
renamed_signer_count = 0
skipped_count = 0

# First pass: rename sentence folders
for sentence in os.listdir(root_path):
    sentence_path = os.path.join(root_path, sentence)
    if not os.path.isdir(sentence_path):
        continue

    clean_sentence_name = sentence.strip().replace(" ", "-")
    new_sentence_path = os.path.join(root_path, clean_sentence_name)

    if sentence != clean_sentence_name:
        if os.path.exists(new_sentence_path):
            print(f"  WARNING: Cannot rename sentence '{sentence}' to '{clean_sentence_name}' — target exists.")
            skipped_count += 1
            continue
        try:
            os.rename(sentence_path, new_sentence_path)
            print(f"  Renamed sentence folder: '{sentence}' → '{clean_sentence_name}'")
            renamed_sentence_count += 1
            sentence_path = new_sentence_path  # Update path for next step
        except Exception as e:
            print(f"  ERROR renaming sentence folder '{sentence}': {e}")
            skipped_count += 1
            continue
    else:
        new_sentence_path = sentence_path  # No change

    # Now normalize signer folders under this sentence
    for signer in os.listdir(new_sentence_path):
        signer_path = os.path.join(new_sentence_path, signer)
        if not os.path.isdir(signer_path):
            continue

        clean_signer_name = signer.strip().replace(" ", "-")
        new_signer_path = os.path.join(new_sentence_path, clean_signer_name)

        if signer != clean_signer_name:
            if os.path.exists(new_signer_path):
                print(f"  WARNING: Cannot rename signer '{signer}' to '{clean_signer_name}' — target exists.")
                skipped_count += 1
                continue
            try:
                os.rename(signer_path, new_signer_path)
                print(f"    Renamed signer folder: '{signer}' → '{clean_signer_name}'")
                renamed_signer_count += 1
            except Exception as e:
                print(f"    ERROR renaming signer folder '{signer}': {e}")
                skipped_count += 1

print(f"\nRenamed {renamed_sentence_count} sentence directories.")
print(f"Renamed {renamed_signer_count} signer directories.")
print(f"Skipped {skipped_count} renames due to warnings/errors.")
print("-" * 40)

# Step 2: Normalize Video Filenames
print("Step 2: Normalizing video filenames (video1.ext, video2.ext)...")
processed_signer_dirs = 0
total_videos_renamed = 0

# Updated structure after renaming
for sentence in os.listdir(root_path):
    sentence_path = os.path.join(root_path, sentence)
    if not os.path.isdir(sentence_path):
        continue

    for signer in os.listdir(sentence_path):
        signer_path = os.path.join(sentence_path, signer)
        if not os.path.isdir(signer_path):
            continue

        print(f"  Processing videos in: {signer_path}")
        processed_signer_dirs += 1
        files_renamed_in_dir = 0

        try:
            video_files = [
                f for f in os.listdir(signer_path)
                if os.path.isfile(os.path.join(signer_path, f)) and not f.startswith('.')
            ]
            video_files_sorted = sorted(video_files)

            if not video_files_sorted:
                print(f"    No valid video files found.")
                continue

            for idx, video_file in enumerate(video_files_sorted, start=1):
                try:
                    base, ext = os.path.splitext(video_file)
                    if not ext:
                        print(f"    Skipping file without extension: {video_file}")
                        continue

                    new_name = f"video{idx}{ext.lower()}"
                    src_full_path = os.path.join(signer_path, video_file)
                    dst_full_path = os.path.join(signer_path, new_name)

                    if src_full_path != dst_full_path:
                        os.rename(src_full_path, dst_full_path)
                        files_renamed_in_dir += 1

                except Exception as e:
                    print(f"    ERROR processing file '{video_file}': {e}")

            if files_renamed_in_dir > 0:
                print(f"    Renamed {files_renamed_in_dir} video file(s).")
                total_videos_renamed += files_renamed_in_dir

        except Exception as e:
            print(f"  ERROR processing signer folder '{signer_path}': {e}")

print(f"\nProcessed {processed_signer_dirs} signer directories.")
print(f"Renamed a total of {total_videos_renamed} video files.")
print("-" * 40)
print("Normalization process complete.")
