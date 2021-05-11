from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class BoulderDataset(CustomDataset):
    CLASSES = ("background", "stone", "border", "unknown")
    # PALETTE = [[255, 0, 0], [0, 255, 0], [0, 0, 255]]
    PALETTE = [[0, 0, 0], [128, 128, 128], [255, 255, 255], [50, 50, 50]]

    def __init__(self, split, **kwargs):
        super().__init__(ignore_index=50, img_suffix='.png', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
