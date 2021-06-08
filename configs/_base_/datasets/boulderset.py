# dataset settings adapted from cityscapes.py
# By Jan Schiffeler
dataset_type = 'BoulderDataset'
# data_root = 'boulderSet/'
data_root = '/home/ubuntu/dataset/boulderSet/'
img_scale = (752, 480)
# img_scale = (1440, 1080)
# TODO why does that change so much?
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
# crop_size = (512, 1024)
# TODO value check crop_size
crop_size = (256, 256)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations'),
    dict(type='Resize', img_scale=img_scale, ratio_range=(0.5, 2.0)),
    dict(type='RandomCrop', crop_size=crop_size, cat_max_ratio=0.75),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PhotoMetricDistortion'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size=crop_size, pad_val=0, seg_pad_val=255),  # TODO what is that and should it be 3 (unknown)
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=img_scale,
        # img_ratios=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=64,  # was 2
    workers_per_gpu=64,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pipeline=train_pipeline,
        split='splits/train.txt'),
    val=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='images',
        ann_dir='labels',
        pipeline=test_pipeline,
        split='splits/val.txt'),
    test=dict(
        type=dataset_type,
        data_root=data_root,
        img_dir='test_set/images',
        ann_dir='test_set/labels',
        pipeline=test_pipeline,
        split='splits/test.txt'))
