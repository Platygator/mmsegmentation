_base_ = [
    '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/boulderset.py',
    # '../_base_/default_runtime.py',
    # '../_base_/schedules/schedule_40k.py'
]

norm_cfg = dict(type='BN', requires_grad=True)
num_classes = 4
model = dict(
    pretrained='torchvision://resnet18',
    backbone=dict(
        norm_cfg=norm_cfg),
    decode_head=dict(
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0,
            class_weight=[0.98, 1.0, 1.02, 1.0])
    ),
    auxiliary_head=dict(
        num_classes=num_classes,
        norm_cfg=norm_cfg,
        loss_decode=dict(
            type='CrossEntropyLoss', use_sigmoid=False, loss_weight=0.4,
            class_weight=[0.98, 1.0, 1.02, 1.0])
    ))

# optimizer
optimizer = dict(type='SGD', lr=0.005, momentum=0.9, weight_decay=0.0005)
optimizer_config = dict()

# learning policy
lr_config = dict(policy='poly', power=0.9, min_lr=1e-4, by_epoch=True)

# Runner
runner = dict(
    type='EpochBasedRunner', # Type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=100)  # Total number of iterations. For EpochBasedRunner use `max_epochs`
checkpoint_config = dict(  # Config to set the checkpoint hook,
    # Refer to https://github.com/open-mmlab/mmcv/blob/master/mmcv/runner/hooks/checkpoint.py for implementation.
    by_epoch=True,  # Whether count by epoch or not.
    interval=4)  # The save interval.
evaluation = dict(  # The config to build the evaluation hook.
    # Please refer to mmseg/core/evaulation/eval_hook.py for details.
    interval=4,  # The interval of evaluation.
    metric='mIoU')
log_config = dict(  # config to register logger hook
    interval=50,  # Interval to print the log
    hooks=[
        # dict(type='TensorboardLoggerHook')  # The Tensorboard logger is also supported
        dict(type='TextLoggerHook', by_epoch=True)
    ])

# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True