### Train a deeplabV3+ model on the simulated boulder data set

Starting from scratch
```
python pipeline/train_boulder_segmentation.py -f
```

Loading a specific checkpoint version
```
python pipeline/train_boulder_segmentation.py -c /path/to/.pth
```

If no argument is given the training continues from "workdir/latest.pth"

### Testing all images in the test set

```
python pipeline/test_boulder_segementation.py --show-dir /data/results
 --eval mIou -p /path/to/.pth
```

all arguments optional. If no checkpoint file (-p) is given
 "workdir/latest.pth" is used.
 

### Infering for a single image (hardcoded as this is a template and shall be used in ROS later)

```
python pipeline/test_boulder_segementation.py --show-dir /data/results
 --eval mIou -p /path/to/.pth
```