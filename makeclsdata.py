import matplotlib
import torch
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import numpy as np
import sys 
import json 
import posixpath
import cv2

src_img = '/data/pill/competition/Yolov7/yolov7/vaipeFix/images/'
src_labels = '/data/pill/competition/Yolov7/yolov7/vaipeFix/labels/'
des_img = '/data/pill/competition/Yolov7/yolov7/classify/images/'
des_labels = '/data/pill/competition/Yolov7/yolov7/classify/'

def read_labels(path, img_w, img_h):
  boxes = []
  labels = []
  with open(path) as f:
    lis = [line.rstrip().split() for line in f]
    for line in lis:
      labels.append(int(line[0]))
      ctrx = float(line[1])
      ctry = float(line[2])
      w = float(line[3])
      h = float(line[4])
      x1 = int((ctrx - w/2) * img_w)
      x2 = int((ctrx + w/2) * img_w)
      y1 = int((ctry - h/2) * img_h)
      y2 = int((ctry + h/2) * img_h)
      boxes.append([x1, y1, x2, y2])
  return labels, boxes 

class_path = des_labels + 'class.txt'

f = open(class_path, "w")
src_img_names = os.listdir(src_img)
for src_img_name in src_img_names:
  img = mpimg.imread(src_img + src_img_name)
  src_label_name = src_img_name.replace('jpg', 'txt')
  labels, boxes = read_labels(src_labels + src_label_name, img.shape[1], img.shape[0])
  for i, box in enumerate(boxes):
    img_cropped = img[box[1]:box[3], box[0]:box[2], :]
    name_img_cropped = src_img_name.replace('.jpg', '_' + str(i)) + '.jpg'
    print(name_img_cropped)
    path_img_cropped = des_img + name_img_cropped
    plt.imsave(path_img_cropped, img_cropped)
    cls = name_img_cropped + " " + str(labels[i]) + "\n"
    f.write(cls)