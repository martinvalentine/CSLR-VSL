## 1. Datasource:
- Sign Language Videos: [Google Drive Link](https://drive.google.com/drive/folders/1ZiUbFvpKDWkTW5HMPRXPXpP0_N5kX7Zs?usp=sharing)

## 2. Prerequisites

- This project is implemented in Pytorch (it should be >=1.13 to be compatible with ctcdecode, or there may be errors) (**Install First**)

- ctcdecode==0.4 [[parlance/ctcdecode]](https://github.com/parlance/ctcdecode)，for beam search decode. (ctcdecode is only supported on the Linux platform.)

- [Optional] sclite [[kaldi-asr/kaldi]](https://github.com/kaldi-asr/kaldi), install kaldi tool to get sclite for evaluation. After installation, create a soft link to the sclite: 
  ```bash
  cd /CorrNet
  mkdir ./software
  ln -s PATH_TO_KALDI/tools/sctk-2.4.10/bin/sclite ./software/sclite
  ```

   You may use the Python version evaluation tool for convenience (by setting 'evaluate_tool' as 'python' in line 16 of ./configs/baseline.yaml), but sclite can provide more detailed statistics.

- You can install other required modules by conducting 
   ```bash
  pip install -r requirements.txt
   ```
  or you can install the environment I'm currently working on:
    ```bash
    conda env create -f environment.yml
    ```
  This will create a new conda env.

- You also need to create base dataset folders for pre-processing by using this command:
  ```bash
  bash ./create_folder.sh
  ```

## 3. Project Structure

```
CSLR-VSL/
├── configs/                    # Configuration YAML files for training and evaluation
│   
├── data/                       # Dataset directory with all stages of processing
│   │
│   ├── 0_raw/                  # Raw videos organized by signer and phrase
│   │
│   ├── 1_external/             # Any external datasets or related data
│   │
│   ├── 2_interim/              # Intermediate outputs (e.g., extracted frames, features)
│   │
│   ├── 3_processed/            # Final cleaned dataset ready for training
│   │
│   └── 4_splits/               # Train/dev/test CSV split files
│   
├── notebooks/                  # Jupyter notebooks for experiments and analysis
│   
├── outputs/                    # Model outputs, logs, and evaluations
│   │
│   ├── evaluation/             # Evaluation results and metrics
│   │
│   ├── logs/                   # Training and runtime logs
│   │
│   └── models/                 # Saved model checkpoints and weights
│   
├── scripts/                    # Preprocessing, data handling, and helper scripts
│   
├── src/                        # Source code root
│   └── cslr_vsl/               # Core CSLR Python package
│       │
│       ├── engine/             # Training/inference pipeline scripts
│       │
│       ├── evaluation/         # Evaluation logic (includes sclite and WER tools)
│       │   └── slr_eval/       # SLR-specific evaluation helpers and GT files
│       │
│       ├── features/           # Dataset preprocessing and feature extraction
│       │
│       ├── models/             # Model definitions (e.g., SLRNet)
│       │
│       ├── modules/            # Reusable components (e.g., BiLSTM, TCN, ResNet)
│       │   └── sync_batchnorm/  # Custom synchronized batch normalization layer
│       │
│       └── utils/              # Utilities: decoding, config, augmentation, logging, etc.
│      
├── test/                       # Inference test scripts and demo tools
│   └── weight_map_generation/  # Grad-CAM visualization ResNet variant
│   
├── third_party/                # External dependencies and binaries
│   └── software/               # Location for sclite binary (from Kaldi)
```

## 4. Data Preparation
### VSL dataset
1. Download the VSL-Sample 2014 Dataset [[Google Drive Link](https://drive.google.com/drive/folders/1ZiUbFvpKDWkTW5HMPRXPXpP0_N5kX7Zs?usp=sharing)].

2. After finishing dataset download, extract it to `data/raw/VSL_Sample/`

3. (***Optional***) If you have minor videos type, you can use the `./scripts/0_flip_video.py` to flip the videos. This script will create a new folder `data/raw/flipped_videos/` with flipped videos.

4. (***Optional***) You can review dataset by running the `./scripts/1_check_num_raw_video.py` script. This script will:
   - Show Video Lengths per Sentence → Signer → Video
   - Listing unique sentences in markdown format
   - Total Video Count

5. Next, you need to standardize the dataset folder and file names to ensure consistency and compatibility with downstream processing steps by running the `./scripts/2_standardize_dataset.py` script. This script will do:
   - Folder Normalization:
     - Cleans up sentence and signer folder names by trimming whitespace and replacing spaces with dashes (`-`).
     - Preserves underscores (`_`) and avoids overwriting existing folders.
   - Video Renaming:
     - Renames video files inside each signer folder to a consistent format: `video1.ext`, `video2.ext`, etc., based on sorted order. 
     - Skips hidden or improperly named files.

6. After standardizing the dataset, you need to split the dataset into `train/dev/test` sets. You can use the `./scripts/3_split_dataset.py` script to do this. This script will:
   - Split the dataset into train/dev/test sets based on the provided split ratio. You can custom the split ratio in the script.
   - Give the summary of the split dataset, including the number of videos in each set and the total number of videos.

7. (***Optional***) You can review the split dataset by running the `./scripts/4_check_num_split_video.py` script. This script will:
   - Show Video Lengths per Sentence → Signer → Video
   - Listing unique sentences in markdown format
   - Total Video Count

8. After splitting the dataset, you need to extract the frames from the videos. You can use the `./scripts/5_extract_frames.py` script to do this. This script will:
   - Extract frames from each video in the dataset and save them in a specified directory.
   - Create a CSV file containing the mapping between video files and their corresponding frame directories.
   - The extracted frames will be cropped to 1080×1080.
  
9. Finally, you need to generate gloss dictionary, ground truth, and resize images by running: 
   ```bash
   python ./src/cslr_vsl/features/dataset_preprocess-VSL.py --process-image --multiprocessing
   ```
  This code will:
    - Generate information dictionary for train, test, dev dataset. 
    - Generate a gloss dictionary for the dataset.
    - Create a `.stm` file containing the ground truth labels for each video.
    - Resize them to 256×256 for augmentation.

## 5. Inference

### VSL dataset

| Backbone | Dev WER  | Test WER  | Pretrained model                                            |
| -------- | ---------- | ----------- | --- |
| ResNet18 | N/A %       | N/A %        | [[Google Drive]]()                                    |

To evaluate the pretrained model, choose the dataset from VSL in line 3 in ./config/baseline.yaml first, and run the command below：   
```bash
python ./scripts/run_corrNet_pipline.py --config ./config/baseline.yaml --device 0 --load-weights your_path_to_weight.pt --phase test
```

### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:
```bash
python ./scripts/run_corrNet_pipline.py --config ./configs/baseline.yaml --device 0
```

### Visualizations
For Grad-CAM visualization, you can replace the resnet.py under "./modules" with the resnet.py under "./weight_map_generation", and then run 
```bash
python generate_cam.py
``` 
with your own hyperparameters.

### Test with one video input
Command to inference with only one video input:

```bash
python ./test/test_one_video.py --model_path /path_to_pretrained_weights --video_path /path_to_your_video --device your_device
```

The `video_path` can be the path to a video file or a dir contains extracted images from a video.

Acceptable paramters:
- `model_path`, the path to pretrained weights.
- `video_path`, the path to a video file or a dir contains extracted images from a video.
- `device`, which device to run inference, default=0.
- `language`, the target sign language, default='vsl'.
- `max_frames_num`, the max input frames sampled from an input video, default=360.

### Demo
The command to test with web GUI
```bash
python ./test/demo.py --model_path /path_to_pretrained_weights --device your_device
```

Acceptable paramters:
- `model_path`, the path to pretrained weights.
- `device`, which device to run inference, default=0.
- `language`, the target sign language, default='vsl'.
- `max_frames_num`, the max input frames sampled from an input video, default=360.

After running the command, you can visit `http://0.0.0.0:7862` to play with the demo. You can also change it into an public URL by setting `share=True` in line 176 in `demo.py`.

## 6. Citation
```latex
@inproceedings{hu2023continuous,
  title={Continuous Sign Language Recognition with Correlation Network},
  author={Hu, Lianyu and Gao, Liqing and Liu, Zekang and Feng, Wei},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  year={2023},
}
```
