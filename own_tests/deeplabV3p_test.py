from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp

@DATASETS.register_module()
class BoulderDataset(CustomDataset):
  CLASSES = ("background", "stone", "border")
  PALETTE = [0, 1, 2]
  def __init__(self, split, **kwargs):
    super().__init__(img_suffix='.png', seg_map_suffix='.png',
                     split=split, **kwargs)
    assert osp.exists(self.img_dir) and self.split is not None


from mmcv import Config
import mmcv
from mmseg.apis import set_random_seed

cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_boulderset.py')
# cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r50-d8_512x1024_40k_cityscapes.py')
# cfg = Config.fromfile('configs/deeplabv3plus/deeplabv3plus_r18b-d8_512x1024_80k_cityscapes.py')



# Since we use ony one GPU, BN is used instead of SyncBN
cfg.norm_cfg = dict(type='BN', requires_grad=True)
cfg.model.backbone.norm_cfg = cfg.norm_cfg
cfg.model.decode_head.norm_cfg = cfg.norm_cfg
cfg.model.auxiliary_head.norm_cfg = cfg.norm_cfg
# modify num classes of the model in decode/auxiliary head
cfg.model.decode_head.num_classes = 8
cfg.model.auxiliary_head.num_classes = 8
# cfg.model.backbone.with_cp = True


# cfg.load_from = 'checkpoints/deeplabv3plus_r50-d8_512x1024_40k_cityscapes_20200605_094610-d222ffcd.pth'
# cfg.load_from = 'checkpoints/deeplabv3plus_r18b-d8_512x1024_80k_cityscapes_20201226_090828-e451abd9.pth'
# cfg.resume_from = 'work_dirs/deeplab_test/iter_200.pth'

# Set up working dir to save files and logs.
cfg.work_dir = './work_dirs/deeplab_test'

cfg.total_iters = 200
cfg.log_config.interval = 10
cfg.evaluation.interval = 200
cfg.checkpoint_config.interval = 200

# Set seed to facitate reproducing the result
cfg.seed = 0
set_random_seed(0, deterministic=False)
cfg.gpu_ids = range(1)

# Let's have a look at the final config used for training
print(f'Config:\n{cfg.pretty_text}')


from mmseg.datasets import build_dataset
from mmseg.models import build_segmentor
from mmseg.apis import train_segmentor


# Build the dataset
datasets = [build_dataset(cfg.data.train)]

# Build the detector
model = build_segmentor(cfg.model)
# Add an attribute for visualization convenience
model.CLASSES = datasets[0].CLASSES

# Create work_dir
mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
train_segmentor(model, datasets, cfg, distributed=False, validate=True,
                meta=dict())


from mmseg.apis import inference_segmentor, show_result_pyplot
import matplotlib.pyplot as plt

img = mmcv.imread('data/test_img/03_02_01450.png')
palette = [0, 1, 2]

model.cfg = cfg
result = inference_segmentor(model, img)
plt.figure(figsize=(8, 6))
show_result_pyplot(model, img, result, palette)
