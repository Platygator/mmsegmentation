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

import torch
import os
import glob
import numpy as np
import shutil
import cv2

from mmcv import Config
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test


def IoU(label: np.ndarray, ground_truth: np.ndarray) -> (np.ndarray, float):
    """
    Calculate Intersection of Union
    :param label: generated label
    :param ground_truth: ground truth to compare to
    :return: IoU per instance and mean IoU
    """
    iou_per_instance = np.zeros(3)
    for i, instance in enumerate([0, 1, 2]):
        org_instance = np.zeros_like(ground_truth)
        org_instance[np.where(ground_truth == instance)] = 1
        rec_instance = np.zeros_like(label)
        rec_instance[np.where(label == instance)] = 1

        intersection = np.logical_and(org_instance, rec_instance).astype('uint8')
        union = np.logical_or(org_instance, rec_instance).astype('uint8')
        iou_per_instance[i] = np.sum(intersection) / np.sum(union)

    return iou_per_instance, np.mean(iou_per_instance)


if __name__ == '__main__':

    # handling argument

    # import configuration
    cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_boulderset.py')
    show_dir = "work_dir/temp/"
    os.mkdir(show_dir)
    DATA_SET = "mmseg_tests/"

    pth_names = [os.path.basename(k) for k in glob.glob(f'work_dir/{DATA_SET}*.pth')]
    for checkpoint_file in pth_names:

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
            model.PALETTE = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]

        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, show=False, out_dir=show_dir,
                                  efficient_test=False)

        label_names = [os.path.basename(k) for k in glob.glob(f'{show_dir}*.png')]

        global_per_instance = np.zeros(3)
        global_mean = 0
        count = 0
        for label_name in label_names:
            label = cv2.imread(f'{show_dir}', 0)
            ground_truth = cv2.imread(f'{DATA_PATH}/{DATA_SET}/ground_truth/{label_name}', 0)

            if label.any():
                count += 1
                per_instance, mean = IoU(label=label, ground_truth=ground_truth)
                global_per_instance += per_instance
                global_mean += mean

        global_per_instance /= count
        global_mean /= count

        shutil.rmtree(show_dir)
        os.mkdir(show_dir)
