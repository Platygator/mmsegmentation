"""
Created by Jan Schiffeler on 04.02.21
jan.schiffeler[at]gmail.com

Test a deeplabV3+ ResNet network on the boulder data set

For single GPU only!

Changed by

Python 3.7
Library version:
    mmcv: 0.10.0

"""

import argparse
import torch

from mmcv import Config
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test


def create_argparser():
    parser = argparse.ArgumentParser(description='Train a deeplabV3p network')
    parser.add_argument('--show_dir', required=False, default='work_dir/results', help='Directory to save evaluated files')
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
        checkpoint_file = "work_dir/latest.pth"
    else:
        checkpoint_file = arg['checkpoint']

    print("[SETTING] Checkpoint used: ", checkpoint_file)

    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=cfg.data.samples_per_gpu,  # changed from 1
        workers_per_gpu=cfg.data.workers_per_gpu,
        dist=False,
        shuffle=False)

    # build the model
    cfg.model.train_cfg = None
    model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg'))

    # load checkpoint
    checkpoint = load_checkpoint(model, checkpoint_file, map_location='cpu')
    model.CLASSES = checkpoint['meta']['CLASSES']
    try:
        model.PALETTE = checkpoint['meta']['PALETTE']
    except KeyError:
        model.PALETTE = [[0, 0, 0], [128, 128, 128], [255, 255, 255], [50, 50, 50]]

    model = MMDataParallel(model, device_ids=[0])
    outputs = single_gpu_test(model, data_loader, show=False, out_dir=arg['show_dir'],
                              efficient_test=False)   # TODO check efficient_test

    _, _ = dataset.evaluate_all(outputs, arg['eval'], {})  # there are possible options
