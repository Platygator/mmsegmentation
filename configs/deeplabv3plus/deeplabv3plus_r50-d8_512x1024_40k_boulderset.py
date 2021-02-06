_base_ = [
    # '../_base_/models/deeplabv3plus_r18-d8.py',
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/boulderset.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
num_classes = 3
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(
        type='ResNet',
        depth=18,
        norm_cfg=norm_cfg),
    decode_head=dict(
        c1_in_channels=64,
        c1_channels=12,
        in_channels=512,
        channels=128,
        num_classes=num_classes,
        norm_cfg=norm_cfg
    ),
    auxiliary_head=dict(
        in_channels=256,
        channels=64,
        num_classes=num_classes,
        norm_cfg=norm_cfg
    ))
