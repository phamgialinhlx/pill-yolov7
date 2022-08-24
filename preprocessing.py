import os
import cv2
import numpy as np
import json
import argparse
from PIL import Image
import argparse
from utils.datasets import exif_size
import tqdm
from PIL import ExifTags, Image, ImageOps, ImageFile
from PIL.ExifTags import TAGS

ImageFile.LOAD_TRUNCATED_IMAGES=True
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break

def rotate_image(origin_path, target_path, overwrite=False):
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    if overwrite:
        for file in os.listdir(target_path):
            os.remove(os.path.join(target_path, file))
        print(f'remove {target_path}')
    img_files = [os.path.join(origin_path, f) for f in os.listdir(origin_path) if f.endswith('.jpg')]
    for img_file in tqdm.tqdm(img_files):
        with open(img_file, "rb") as f:
            targ = os.path.join(target_path, img_file.split("/")[-1])
            f.seek(-2, 2)
            img = Image.open(img_file)
            exifdata = img.getexif()
            for tag_id in exifdata:
                # get the tag name, instead of human unreadable tag id
                tag = TAGS.get(tag_id, tag_id)
                data = exifdata.get(tag_id) 
                if (tag == 'Orientation'):
                    img = ImageOps.exif_transpose(img)
            img.save(targ, "JPEG", subsampling=0, quality=100)
            

def list_all_files(dir):
    files = []
    for r, d, f in os.walk(dir):
        for file in f:
            if '.jpg' in file:
                files.append(os.path.join(r, file))
    return files

def correct_box(x):
    return max(0.0001, min(0.9999, x))
def convert_labels(origin_path, target_path, overwrite=False):
    # read sys.argv
    # argv[1] is the server
    # argv[2] is the overwrite flag
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
        img_path = targ_file_path.replace('label','image').replace('.txt','.jpg')
        with open(org_file_path) as f:
            tmp = json.load(f)
            file = open(targ_file_path, "w")
            img = Image.open(img_path)
            # print(img_path)
            w, h = exif_size(img)
            for i in tmp:
                _w = i['w'] / w
                _h = i['h'] / h
                _x = i['x'] / w + _w / 2
                _y = i['y'] / h + _h / 2 
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
    parser.add_argument('--origin_path', type=str, help='path of origin folder containing labels subfolders')
    parser.add_argument('--target_path', type=str,default='vaipe_exif', help='path of target folder')
    parser.add_argument('--overwrite', action='store_true', help='if True remove all files in target folder')
    parser.add_argument('--task', type=str, default = 'test', help='test or train')
    parser.add_argument('--convert_labels', action='store_true', help='if True convert labels from json to txt')
    args = parser.parse_args()


    if args.convert_labels:
        if args.task == 'train':
            target_path = os.path.join(args.target_path, 'train/labels')
            convert_labels(args.origin_path, target_path, args.overwrite)
    else:
        if args.task == 'test':
            target_path = os.path.join(args.target_path, 'test/images')
        else:
            target_path = os.path.join(args.target_path, 'train/images')
        rotate_image(args.origin_path, target_path, args.overwrite)
        files = list_all_files(target_path)
        #export list of files to .txt file
        file = open(os.path.join(args.target_path, f'{args.task}.txt'), 'w')
        for i in files:
            file.write(i + '\n')
        file.close()