import numpy as np
import os
from tqdm import tqdm
from functools import partial
from vsl_preprocess import Preprocessing
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Data process for Visual Alignment Constraint for Continuous Sign Language Recognition.')
    parser.add_argument('--dataset', type=str, default='VSL',
                        help='save prefix')
    parser.add_argument('--dataset-root', type=str, default='/home/martinvalentine/Desktop/CSLR-VSL/data/interim/VSL_Sample',
                        help='path to the dataset')
    parser.add_argument('--annotation-prefix', type=str, default='/home/martinvalentine/Desktop/CSLR-VSL/data/processed/VSL_Sample/csv/{}_annotations.csv',
                        help='annotation prefix')
    parser.add_argument('--output-res', type=str, default='256x256px',
                        help='resize resolution for image sequence')
    parser.add_argument('--process-image', '-p', action='store_true',
                        help='resize image')
    parser.add_argument('--multiprocessing', '-m', action='store_true',
                        help='whether adopts multiprocessing to accelate the preprocess')

    args = parser.parse_args()
    mode = ["dev", "test", "train"]
    sign_dict = dict()
    if not os.path.exists(f"./{args.dataset}"):
        os.makedirs(f"./{args.dataset}")
    for md in mode:
        # generate information dict
        information = Preprocessing.annotation2dict(f"{args.dataset_root}", f"{args.annotation_prefix.format(md)}", split=md)
        np.save(f"./{args.dataset}/{md}_info.npy", information)
        # update the total gloss dict
        Preprocessing.gloss_dict_update(sign_dict, information)
        # generate groudtruth stm for evaluation
        Preprocessing.generate_stm(information, f"./{args.dataset}"+f"/{args.dataset}-ground-truth-{md}.stm".lower())
        # resize images
        video_index = np.arange(len(information) - 1)
        print(f"Resize image to {args.output_res}")
        if args.process_image:
            if args.multiprocessing:
                Preprocessing.run_mp_cmd(10, partial(Preprocessing.resize_dataset, dsize=args.output_res, info_dict=information), video_index)

                # DEBUG
                # for idx in video_index:
                #     print(information)
                # END DEBUG

            else:
                for idx in tqdm(video_index):
                    Preprocessing.run_cmd(partial(Preprocessing.resize_dataset, dsize=args.output_res, info_dict=information), idx)
                    # resize_dataset(idx, dsize=args.output_res, info_dict=information)
    sign_dict = sorted(sign_dict.items(), key=lambda d: d[0])
    save_dict = {}
    for idx, (key, value) in enumerate(sign_dict):
        save_dict[key] = [idx + 1, value]
    np.save(f"./{args.dataset}/gloss_dict.npy", save_dict)


