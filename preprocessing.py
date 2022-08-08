import os
import cv2
import numpy as np
import json
import argparse
from PIL import Image
import argparse
from utils.datasets import exif_size
import tqdm
def correct_box(x):
    return max(0.0001, min(0.9999, x))
def convert_labels(origin_path, target_path, overwrite=False):
    # read sys.argv
    # argv[1] is the server
    # argv[2] is the overwrite flag
    origin_path = os.path.join(origin_path, 'labels')
    target_path = os.path.join(target_path, 'labels')
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if overwrite:
        for file in os.listdir(target_path):
            os.remove(os.path.join(target_path, file))
        print(f'remove {target_path}')
    org_files = os.listdir(origin_path)
    json_data = []
    pbar = tqdm.tqdm(total=len(org_files))

    for i, org_file in enumerate(org_files):
        #check if is a json file
        if not org_file.endswith('.json'):
            print(org_file)
            continue
        org_file_path = os.path.join(origin_path, org_file)
        targ_file_path = os.path.join(target_path, org_file).replace('.json', '.txt')
        with open(org_file_path) as f:
            tmp = json.load(f)
            file = open(targ_file_path, "w")
            img = Image.open(org_file_path.replace('label', 'image').replace('json', 'jpg'))
            w, h = exif_size(img)
            sz = max(w, h)
            for i in tmp:
                _w = i['w'] / sz
                _h = i['h'] / sz
                _x = i['x'] / sz + _w / 2
                _y = i['y'] / sz + _h / 2
                s = str(i['label']) + ' ' + str(_x) + ' ' + str(_y) + ' ' + str(_w) + ' ' + str(_h)
                file.write(s + '\n')
            # print(file)
            file.close()
        pbar.update(1)#update progress bar
    pbar.close()
    print(f'path of target label: {target_path}')
    return target_path
def padding(origin_path, target_path, overwrite=False):
    """
    :param origin_path: path of origin image
    :param target_path: path of target image
    :return: path of target image

    """
    origin_path = os.path.join(origin_path, 'images')
    target_path = os.path.join(target_path, 'images')
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if overwrite:
        for file in os.listdir(target_path):
            os.remove(os.path.join(target_path, file))
        print(f'remove {target_path}')

    org_files = os.listdir(origin_path)
    # print(org_files)
    pbar = tqdm.tqdm(total=len(org_files))
    for i, org_file in enumerate(org_files):
        #check if the file is an image
        if not org_file.endswith('.jpg'):
            continue

        org_file_path = os.path.join(origin_path, org_file)
        img = cv2.imread(org_file_path)
        

        org_h, org_w, channels = img.shape
        sz = max(org_h, org_w)
        color = (148,0,211)
        padded = np.full((sz, sz, channels), color, dtype=np.uint8)
        padded[:org_h, :org_w] = img

        targ_file_path = os.path.join(target_path, org_file)
        cv2.imwrite(targ_file_path, padded)
        pbar.update(1)
    pbar.close()
    print(f'path of target image: {target_path}')
    return target_path
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--origin_path', type=str, help='path of origin folder containing images and labels subfolders')
    parser.add_argument('--target_path', type=str, help='path of target folder')
    parser.add_argument('--overwrite', type=bool, default=False, help='if True remove all files in target folder')
    #choose padding or convert labels
    parser.add_argument('--padding', type=bool, default=False, help='if True padding images')
    parser.add_argument('--convert_labels', type=bool, default=False, help='if True convert labels. json to txt')
    args = parser.parse_args()
    if args.padding:
        padding(args.origin_path, args.target_path, args.overwrite)
    if args.convert_labels:
        convert_labels(args.origin_path, args.target_path, args.overwrite)