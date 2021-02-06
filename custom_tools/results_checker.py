import mmcv
import numpy as np

img = mmcv.imread('data/boulderColabFormat/labels/03_02_00750.png')
res = mmcv.imread('data/results/03_02_00750.png')
res_live = mmcv.imread('data/live_test.png')

# res = np.floor(res * res * 63.75).astype('uint8')
img = np.floor(img * 127.5).astype('uint8')
res = np.floor(res * 127.5).astype('uint8')
res_live = np.floor(res_live * 127.5).astype('uint8')

mmcv.imwrite(res, 'data/ground_truth_03_02_00750.png')
pip