import os
import cv2
import csv
import re
import glob # File and Directory pattern
from pathlib import Path # Path manipulation
import pandas as pd
import numpy as np
import argparse # Command line argument parsing
from tqdm import tqdm # Progress bar
from functools import partial # Partial function application
from multiprocessing import Pool # Parallel processing

class Preprocessing:
     def process_and_extract_frames(root_dir, output_dir, csv_dir, split):
          """
          Extract frames from videos and generate an annotation CSV file.
          """
          root_path = Path(root_dir) / split
          output_path = Path(output_dir) / split
          output_path.mkdir(parents=True, exist_ok=True)
          csv_path = Path(csv_dir)
          csv_path.mkdir(parents=True, exist_ok=True)

          annotations = []

          for sentence_index, sentence_folder in enumerate(sorted(root_path.iterdir())):
               if not sentence_folder.is_dir():
                    continue

               sentence_id = sentence_index + 1  # Ensure sequential numbering for sentences
               gloss_label = sentence_folder.name  # Use folder name as gloss label

               for signer_folder in sorted(sentence_folder.iterdir()):
                    if not signer_folder.is_dir():
                         continue

                    # Extract signer ID from the signer folder name
                    signer_match = re.search(r'Signer(\d+)', str(signer_folder))
                    signer_id = int(signer_match.group(1)) if signer_match else 0

                    for trial_index, video_file in enumerate(sorted(signer_folder.iterdir())):
                         if video_file.suffix.lower() in {".mp4", ".mov", ".avi"}:
                              # Extract frames from the video
                              cap = cv2.VideoCapture(str(video_file))
                              frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                              video_id = f"S{sentence_id:06d}_P{signer_id:04d}_T{trial_index+1:02d}"
                              video_output_path = Path(output_path) / video_id
                              video_output_path.mkdir(parents=True, exist_ok=True)

                              frame_index = 0
                              while cap.isOpened():
                                   ret, frame = cap.read()
                                   if not ret:
                                        break
                                   frame_filename = video_output_path / f"frame_{frame_index:04d}.png"
                                   cv2.imwrite(str(frame_filename), frame)
                                   frame_index += 1

                              cap.release()
                              print(f"Extracted {frame_count} frames from {video_file} → {video_output_path}")

                              # Add annotation entry
                              relative_path = video_output_path.relative_to(output_path.parent)
                              annotations.append([video_id, f"{relative_path}", gloss_label])

          # Save annotations to CSV
          csv_file_path = csv_path / f"{split}_annotations.csv"
          with open(csv_file_path, mode='w', newline='', encoding='utf-8') as file:
               writer = csv.writer(file)
               writer.writerow(["Video_ID", "Frames_Path", "Gloss_Label"])
               writer.writerows(annotations)
          print(f"Annotations saved to {csv_file_path}")


     def annotation2dict(dataset_root, anno_path, split):
          # Load annotation data
          df = pd.read_csv(anno_path)

          print(f"Generate information dict from {anno_path}")
          
          info_dict = {} # For storing data in dict format

          for idx, row in tqdm(df.iterrows(), total=len(df) - 1):
               video_id = str(row["Video_ID"])
               frame_folders = os.path.join(dataset_root, split, video_id) # absolute path frame
               gloss = str(row['Gloss_Label'])

               if not os.path.exists(frame_folders):
                    print('Warning: Frames folder not found -> {frame_folders}')
               
               # Count number of frame
               num_frames = len([f for f in os.listdir(frame_folders) if f.endswith(".png")])

               # Extract Signer ID (Pxxxx from Sxxxxxx_Pxxxx-Txx)
               signer_id_extracted = video_id.split('_')[1] # Extract Pxxxx
               signer_id = f"singer{signer_id_extracted[1:]}" # Convert Pxxxx to singerxxxx

               # Store structured data in dictionary format
               info_dict[idx] = {
                    "fileid": video_id,                            # Unique file ID
                    "folder": os.path.join(video_id, "*.png"),     # Folder path pattern
                    "signer": signer_id,                           # Signer ID (Pxxxx)
                    "label": gloss,                                # Label (gloss)
                    "num_frames": num_frames,                      # Number of frames
                    "original_info": f"{video_id}|{num_frames}|{gloss}"  # Original info string
               }

          return info_dict
     

     def generate_stm(info_dict, save_path):
          try:
               with open(save_path, 'w') as f:
                    for k, v in info_dict.items():
                         if not isinstance(k, int):  # Ensure key is an integer index
                              continue
                         f.writelines(f"{v['fileid']} 1 {v['signer']} 0.0 1.79769e+308 {v['label']}\n") 
                         # 0.0 1.79769e+308 mean from start to the end of the video
               
               print(f"Ground truth STM saved to {save_path}")
               print("STM generation completed successfully!")
          except Exception as e:
               print(f"An error occurred while generating the STM file: {e}")


     def gloss_dict_update(total_dict, info_dict):
          for k, v in info_dict.items():
               if not isinstance(k, int):
                    continue

               # Split the label by whitespace; if the label contains multiple tokens, count each separately
               tokens = v['label'].split()
               for token in tokens:
                    token = token.strip()
                    if not token:
                         continue
                    if token not in total_dict:
                         total_dict[token] = [next_id, 1]  # [Gloss ID, Occurrence Count]
                         next_id += 1
                    else:
                         total_dict[token][1] += 1  # Increment occurrence count
          
          return total_dict   