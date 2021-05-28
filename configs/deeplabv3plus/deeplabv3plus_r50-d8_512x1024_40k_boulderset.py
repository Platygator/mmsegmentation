_base_ = [
    # '../_base_/models/deeplabv3plus_r18-d8.py',
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/boulderset.py',
    '../_base_/default_runtime.py',
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
        norm_cfg=norm_cfg,
        # loss_decode=dict(
        #     type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
        #     class_weight=[0.8, 1.0, 1.2, 1.0])
    ),
    auxiliary_head=dict(
        in_channels=256,
        channels=64,
        num_classes=num_classes,
        norm_cfg=norm_cfg
    ))
runner = dict(
    type='IterBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_iters=50000) # Total number of iterations. For EpochBasedRunner use `max_epochs`
checkpoint_config = dict(  # Config to set the checkpoint hook, Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=False,  # Whethe count by epoch or not.
    interval=4000)  # The save interval.
evaluation = dict(  # The config to build the evaluation hook. Please refer to mmseg/core/evaulation/eval_hook.py for details.
    interval=4000,  # The interval of evaluation.
    metric='mIoU')
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook', by_epoch=False)
    ])