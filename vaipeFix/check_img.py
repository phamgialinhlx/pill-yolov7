import glob
import os
import os.path as osp
import random
import json
import sys
import time
import hashlib
import posixpath

from multiprocessing.pool import Pool

import cv2
import numpy as np
import matplotlib.pyplot as plt
import torch
from PIL import ExifTags, Image, ImageOps, ImageFile
from torch.utils.data import Dataset
from tqdm import tqdm

ImageFile.LOAD_TRUNCATED_IMAGES=True

sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('/',1)[0])

# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break
def main(argv):
    if len(argv) < 2:
        print("Usage: convert_labels.py choose_server(2080 or v100)")
        return
    overwrite = False
    if len(argv) == 3:
        overwrite = argv[2]
    server = argv[1]

    mapping = {'2080': '/data/pill/competition/Yolov7/yolov7/vaipeFix/images', 'v100': '/home/pill/competition/yolov7/vaipeFix/images/', 'colab' : './content/drive/Shareddrives/Data/public_train/public_train/pill/label'}
    directory = mapping[server]
    files = os.listdir(directory)
    img_files = [posixpath.join(directory, f) for f in os.listdir(directory) if posixpath.isfile(posixpath.join(directory, f)) and f.endswith('.jpg')]
    for img_file in img_files:
        with open(img_file, "rb") as f:
            f.seek(-2, 2)
            if f.read() != b"\xff\xd9":  # corrupt JPEG
                ImageOps.exif_transpose(Image.open(img_file)).save(
                    img_file, "JPEG", subsampling=0, quality=100
                )
        

if __name__ == '__main__':
    main(sys.argv)