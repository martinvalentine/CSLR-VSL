import os
import cv2
import re
import glob # File and Directory pattern
import pandas as pd
from tqdm import tqdm # Progress bar
from multiprocessing import Pool # Parallel processing


def resize_img(img_path, dsize='256x256px'):
     dsize = tuple(int(res) for res in re.findall("\d+", dsize))
     img = cv2.imread(img_path)
     img = cv2.resize(img, dsize, interpolation=cv2.INTER_LANCZOS4)
     return img


class Preprocessing:

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
                    "folder": os.path.join(split, video_id, "*.png"),     # Folder path pattern
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
          next_id = 1  # Start from 1, as 0 is reserved for the blank token
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


     def resize_dataset(video_idx, dsize, info_dict):
          info = info_dict[video_idx]
          prefix = '/home/martinvalentine/Desktop/CSLR-VSL/data/interim/VSL_Benchmark'
          img_list = glob.glob(f"{prefix}/{info['folder']}")
          for img_path in img_list:
               rs_img = resize_img(img_path, dsize=dsize)
               rs_img_path = img_path.replace("interim/VSL_Benchmark", "processed/VSL_Benchmark/features/" + dsize)
               rs_img_dir = os.path.dirname(rs_img_path)
               if not os.path.exists(rs_img_dir):
                    os.makedirs(rs_img_dir)
                    cv2.imwrite(rs_img_path, rs_img)
               else:
                    cv2.imwrite(rs_img_path, rs_img)


     def run_mp_cmd(processes, process_func, process_args):
          with Pool(processes) as p:
               outputs = list(tqdm(p.imap(process_func, process_args), total=len(process_args)))
          return outputs


     def run_cmd(func, args):
          return func(args)
