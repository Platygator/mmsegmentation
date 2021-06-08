from mmseg.datasets.builder import DATASETS
from mmseg.datasets.custom import CustomDataset
import os.path as osp


@DATASETS.register_module()
class BoulderDataset(CustomDataset):
    # TODO should my unknown label be named here?
    #      why does using the visual labels (commented) not work?
    CLASSES = ("background", "stone", "border", "unknown")
    # PALETTE = [[0, 0, 0], [128, 128, 128], [255, 255, 255], [50, 50, 50]]
    PALETTE = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]]

    def __init__(self, split, **kwargs):
        super().__init__(ignore_index=3, img_suffix='.png', seg_map_suffix='.png',
                         split=split, **kwargs)
        assert osp.exists(self.img_dir) and self.split is not None
