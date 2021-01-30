_base_ = [
    '../_base_/models/deeplabv3plus_r18-d8.py',
    # '../_base_/models/deeplabv3plus_r50-d8.py',
    '../_base_/datasets/boulderset.py', '../_base_/default_runtime.py',
    '../_base_/schedules/schedule_40k.py'
]
