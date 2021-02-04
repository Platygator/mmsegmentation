"""
Created by Jan Schiffeler on 04.02.21
jan.schiffeler[at]gmail.com

Train a deeplabV3+ ResNet network to detect boulders and their edges

Changed by

Python 3.7
Library version:
    mmcv: 0.10.0

"""

from mmcv import Config
from mmcv import mkdir_or_exist as mk_e

from mmseg.apis import set_random_seed, train_segmentor
from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor

import os.path as osp
import argparse

# PARAMETERS
train_total = 100
train_log = 20
train_eval = 200
train_save = 400


def create_argparser():
    """
    setup all required arguments as in:
        parser.add_argument('-s', '--side', required=False, default=0, help='some help text')
    :return: argparse dictionary
    """
    parser = argparse.ArgumentParser(description='Train a deeplabV3p network')
    parser.add_argument('-c', '--continue', action='store_true', help='continue from latest.pth')
    args = vars(parser.parse_args())
    return args


if __name__ == '__main__':

    # handling arguments
    arg = create_argparser()

    # create the configuration
    cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_boulderset.py')
    cfg.work_dir = './work_dirs/deeplab_test'
    mk_e(osp.abspath(cfg.work_dir))

    if arg['continue']:
        cfg.resume_from = f'{cfg.work_dir}/latest.pth'

    # evaluation, printout and saving settings
    cfg.total_iters = train_total
    cfg.log_config.interval = train_log
    cfg.evaluation.interval = train_eval
    cfg.checkpoint_config.interval = train_save

    # seed
    cfg.seed = 0
    set_random_seed(0, deterministic=False)
    cfg.gpu_ids = range(1)

    # Build the dataset
    datasets = [build_dataset(cfg.data.train)]

    # Build the detector
    model = build_segmentor(cfg.model)

    # Add an attribute for visualization convenience
    model.CLASSES = datasets[0].CLASSES
    model.PALETTE = datasets[0].PALETTE

    # start training
    train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                    meta=dict())
