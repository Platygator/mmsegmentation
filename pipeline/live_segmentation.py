"""
Created by Jan Schiffeler on 05.02.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""

import argparse
import torch

from mmcv import Config, imread, imwrite
from mmseg.apis import inference_segmentor, init_segmentor


def create_argparser():
    parser = argparse.ArgumentParser(description='Train a deeplabV3p network')
    parser.add_argument('--show_dir', required=False, default='data/results', help='Directory to save evaluated files')
    parser.add_argument('-p', '--checkpoint', required=False, help='Training state to be used')
    parser.add_argument('--eval', type=str, nargs='+',
                        help='evaluation metrics, which depends on the dataset, e.g., "mIoU"'
                             ' for generic datasets, and "cityscapes" for Cityscapes')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':

    # handling arguments
    arg = create_argparser()

    # import configuration
    cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_boulderset.py')

    if not arg['checkpoint']:
        checkpoint_file = cfg.work_dir + "/latest.pth"
    else:
        checkpoint_file = arg['checkpoint']

    print("[SETTING] Checkpoint used: ", checkpoint_file)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True


    model = init_segmentor(config=cfg, checkpoint=checkpoint_file)
    img = imread('data/boulderColabFormat/images/03_02_00750.png')
    result = inference_segmentor(model, img)
    imwrite(img=result[0], file_path='data/live_test.png')
