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
   ``` bash
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

## 3. Implementation
The implementation for the CorrNet (line 18) is given in [./CorrNet/modules/resnet.py](https://github.com/martinvalentine/CSLR-VSL/blob/main/CorrNet/modules/resnet.py).  

It's then equipped with the BasicBlock in ResNet in line 58 [./CorrNet/modules/resnet.py](https://github.com/martinvalentine/CSLR-VSL/blob/main/CorrNet/modules/resnet.py).

## 4. Data Preparation
Currently, a full end-to-end preprocessing pipeline has not been implemented. However, you can preprocess raw videos using the following script:
[[./notebooks/pipeline_corrnet.ipynb](https://github.com/martinvalentine/CSLR-VSL/blob/main/notebooks/base_lstm_model_exp.ipynb)]

## 5. Inference

### VSL dataset

| Backbone | Dev WER  | Test WER  | Pretrained model                                            |
| -------- | ---------- | ----------- | --- |
| ResNet18 | N/A %       | N/A %        | [[Google Drive]]()                                    |


​	To evaluate the pretrained model, choose the dataset from VSL in line 3 in ./config/baseline.yaml first, and run the command below：   
```bash
python main.py --config ./config/baseline.yaml --device 0 --load-weights path_to_weight.pt --phase test
```


### Training

The priorities of configuration files are: command line > config file > default values of argparse. To train the SLR model, run the command below:
```bash
cd CorrNet/
python main.py --config ./configs/baseline.yaml --device 0
```

### Visualizations
For Grad-CAM visualization, you can replace the resnet.py under "./modules" with the resnet.py under "./weight_map_generation", and then run ```python generate_cam.py``` with your own hyperparameters.

### Test with one video input
Command to inference with only one video input:

```bash
python test_one_video.py --model_path /path_to_pretrained_weights --video_path /path_to_your_video --device your_device
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
python demo.py --model_path /path_to_pretrained_weights --device your_device
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
