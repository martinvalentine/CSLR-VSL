import numpy as np
import os
from tqdm import tqdm
from functools import partial
from cslr_vsl.utils.vsl_preprocess import Preprocessing
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')

    parser.add_argument('--dataset-prefix', type=str, default='vsl_v2',
                        help='Save prefix for ground truth file')
    parser.add_argument('--processed-feature-root', type=str,
                        default='/home/martinvalentine/Desktop/CSLR-VSL/data/processed/VSL_V2',
                        help='Path to save the processed feature')
    parser.add_argument('--dataset-root', type=str,
                        default='/home/martinvalentine/Desktop/CSLR-VSL/data/interim/frames/VSL_V2',
                        help='Path to the dataset root (where frame folders are located)')
    parser.add_argument('--annotation-prefix', type=str,
                        default='/home/martinvalentine/Desktop/CSLR-VSL/data/splits/VSL_V2/csv/{}_annotations.csv',
                        help='Path template for CSV annotations with mode placeholder (train/test/dev)')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='Resize resolution for image sequences, e.g., 256x256px')
    parser.add_argument('--process-image', '-p', action='store_true',
                        help='Enable image resizing')
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='Use multiprocessing to accelerate image preprocessing')

    args = parser.parse_args()

    modes = ["dev", "test", "train"]
    sign_dict = {}

    os.makedirs(args.processed_feature_root, exist_ok=True)

    for mode in modes:
        anno_path = args.annotation_prefix.format(mode)

        # Load annotation info from CSV and convert to dict
        info_dict = Preprocessing.annotation2dict(args.dataset_root, anno_path, split=mode)
        np.save(os.path.join(args.processed_feature_root, f"{mode}_info.npy"), info_dict)

        # Update global gloss dictionary
        Preprocessing.gloss_dict_update(sign_dict, info_dict)

        # Save ground truth STM file
        stm_path = os.path.join(args.processed_feature_root, f"{args.dataset_prefix}-ground-truth-{mode}.stm".lower())
        Preprocessing.generate_stm(info_dict, stm_path)

        # Save ground truth to evaluation folder
        eval_path = "/home/martinvalentine/Desktop/CSLR-VSL/src/cslr_vsl/evaluation/slr_eval/"
        stm_eval_path = os.path.join(eval_path, f"{args.dataset_prefix}-ground-truth-{mode}.stm".lower())
        Preprocessing.generate_stm(info_dict, stm_eval_path)

        # Resize frames if required
        video_indices = np.arange(len(info_dict))
        print(f"Resize image to {args.output_res}")
        if args.process_image:
            resize_fn = partial(Preprocessing.resize_dataset, dsize=args.output_res, info_dict=info_dict)
            if args.multiprocessing:
                Preprocessing.run_mp_cmd(10, resize_fn, video_indices)
            else:
                for idx in tqdm(video_indices):
                    Preprocessing.run_cmd(resize_fn, idx)

    # Save sorted gloss dictionary (sorted by gloss name)
    sorted_gloss = sorted(sign_dict.items(), key=lambda x: x[0])
    save_dict = {k: [i + 1, v[1]] for i, (k, v) in enumerate(sorted_gloss)}
    np.save(os.path.join(args.processed_feature_root, "gloss_dict.npy"), save_dict)
