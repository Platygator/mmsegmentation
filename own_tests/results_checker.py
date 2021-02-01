import mmcv
import numpy as np

img = mmcv.imread('data/boulderColabFormat/images/03_02_00750.png')
res = mmcv.imread('data/results/03_02_00750.png')

diff = img-res

print(diff.any())

mmcv.imshow(diff*127, "results", 2000)
