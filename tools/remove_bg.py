import os
import io
import cv2
import sys
import json
import math
import tqdm
import torch
import gdown
import hashlib
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import onnxruntime as ort
from contextlib import redirect_stdout
from typing import List, Optional, Union, Type
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL.ExifTags import TAGS
from PIL import ExifTags, Image, ImageOps
from rembg.bg import remove



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, help='path to csv predict file')
    parser.add_argument('--img_path_file', type=str, default='./vaipe_exif/train.txt', help='path of the file that contains path to the images')

    
    args = parser.parse_args()
