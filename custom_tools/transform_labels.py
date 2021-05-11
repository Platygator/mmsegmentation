"""
Created by Jan Schiffeler on 13.04.21
jan.schiffeler[at]gmail.com

Changed by



Python 3.
Library version:


"""
import cv2
import numpy as np
import glob
import os

data_root = 'boulderSet'
ann_dir = 'labels'

photo_images = [k for k in glob.glob(f'{os.path.join(data_root, ann_dir)}/*.png')]
len_img = len(photo_images)

for i, img_name in enumerate(photo_images):
    print(f"Transforming image {i+1} / {len_img} ")
    img = cv2.imread(img_name, 0)
    img[img == 128] = 1
    img[img == 255] = 2
    img[img == 50] = 3
    cv2.imwrite(img_name, img)
