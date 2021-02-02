import mmcv
import numpy as np

img = mmcv.imread('data/BoulderDataset/images/03_02_00750.png')
res = mmcv.imread('data/results/03_02_00750.png')

diff = img-res

img_half = np.floor(img/2)
diff2 = img_half-res
diff2_amp = np.ceil(diff2*127.5)
print(diff.any())

mmcv.imshow(diff*127, "results", 2000)
