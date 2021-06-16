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
import numpy as np

from mmcv import Config
from mmseg.datasets import build_dataset, build_dataloader
from mmseg.models import build_segmentor
from mmcv.parallel import MMDataParallel
from mmcv.runner import load_checkpoint
from mmseg.apis import single_gpu_test


# def IoU(label: np.ndarray, ground_truth: np.ndarray) -> (np.ndarray, float):
#     """
#     Calculate Intersection of Union
#     :param label: generated label
#     :param ground_truth: ground truth to compare to
#     :return: IoU per instance and mean IoU
#     """
#     iou_per_instance = np.zeros(3)
#     for i, instance in enumerate([0, 1, 2]):
#         org_instance = np.zeros_like(ground_truth)
#         org_instance[np.where(ground_truth == instance)] = 1
#         rec_instance = np.zeros_like(label)
#         rec_instance[np.where(label == instance)] = 1
#
#         intersection = np.logical_and(org_instance, rec_instance).astype('uint8')
#         union = np.logical_or(org_instance, rec_instance).astype('uint8')
#         iou_per_instance[i] = np.sum(intersection) / np.sum(union)
#
#     return iou_per_instance, np.mean(iou_per_instance)


if __name__ == '__main__':

    # handling argument

    # import configuration
    cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_boulderset.py')
    DATA_SET = "first_full_test"

    n_epochs = 260
    step_epoch = 10
    # pth_names = [os.path.basename(k) for k in glob.glob(f'work_dir/{DATA_SET}/*.pth')]

    store_mean = []
    store_background = []
    store_stone = []
    store_border = []

    for epoch in range(step_epoch, n_epochs + step_epoch, step_epoch):

        checkpoint_file = f'work_dir/{DATA_SET}/epoch_{epoch}.pth'
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
        model.CLASSES = dataset.CLASSES
        model.PALETTE = dataset.PALETTE

        model = MMDataParallel(model, device_ids=[0])
        outputs = single_gpu_test(model, data_loader, show=False, out_dir="work_dir/tmp/",
                                  efficient_test=False)

        res_mean, res_class = dataset.evaluate_all(outputs)

        store_mean.append(res_mean["mIoU"])
        store_background.append(res_class["background"])
        store_stone.append(res_class["stone"])
        store_border.append(res_class["border"])

    np.save(f"work_dir/{DATA_SET}.npy", {"Background": store_background, "Stone": store_stone,
                                "Border": store_border, "Mean": store_mean})

    print("[INFO] Saving ", f"work_dir/{DATA_SET}.npy")
