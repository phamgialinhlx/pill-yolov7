import os
import cv2
import csv
import pandas as pd
import numpy as np
import json
import argparse
import tqdm

def convert(file, target_path):
    df = pd.read_csv(file, index_col=False)
    files = df['file_name'].unique()
    pbar = tqdm.tqdm(total=len(files))
    for f in files:
        a = df.loc[df['file_name'] == f]
        a = a.iloc[:, 1:]
        result = a.to_json(orient="records")
        parsed = json.loads(result)
        x = json.dumps(parsed, indent=4)  
        json_file = open(target_path + f, "w")
        json_file.write(x)
        json_file.close()
        pbar.update(1)
    pbar.close()
    # print(df.to_string()) 

def update_imgpath(img_path_file, file):
    df = pd.read_csv(file, index_col=False)
    files = df['file_name'].unique()
    with open(img_path_file, 'a') as f:
        pbar = tqdm.tqdm(total=len(files))
        for img_path in files:
            f.write('./train/image/' + img_path.replace('.json', '.jpg') + '\n')
            pbar.update(1)
        pbar.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file', type=str, default='./gen/label.csv', help='path to csv labels file')
    parser.add_argument('--target_path', type=str, default='./label/', help='path of target folder')
    parser.add_argument('--img_path_file', type=str, default='./vaipe_exif/train.txt', help='path of the file that contains path to the images')
    args = parser.parse_args()

    convert(args.file, args.target_path)
    update_imgpath(args.img_path_file, args.file)

