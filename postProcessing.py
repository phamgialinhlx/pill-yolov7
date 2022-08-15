import pandas as pd
import os

import json

import argparse

from loadOCR import load_OCR

def post_processing(path_to_detect_output, path_to_ocr_res):
    df = pd.read_csv(path_to_detect_output)
    df['image_id'] = df['image_name'].apply(lambda x: x.split('_')[2])
    OCR_res = load_OCR(path_to_ocr_res)
    df = df.merge(OCR_res, on='image_id', how='left')
    print(df)
    for index, row in df.iterrows():
        if row['class_id'] not in row['id']:
            df.loc[index, 'class_id'] = 107
    
    df = df.drop(columns=['id', 'filename','image_id'])
    submission_path = path_to_detect_output.replace('submission.csv', 'results.csv')
    df.to_csv(submission_path, index=False)
    return submission_path
def convert(json_file):
    result_path = os.path.join(json_file.rsplit('/', 1)[0], 'submission.csv')
    with open(json_file) as f:
        data =  json.load(f)
        df = pd.DataFrame(data)
    df[['x_min','y_min','x_max','y_max']] = pd.DataFrame(df.bbox.tolist(), index= df.index)
    df.rename(columns ={'image_id':'image_name','category_id':'class_id','score':'confidence_score'}, inplace = True)
    df['image_name'] = df['image_name'] + '.jpg'
    del df['bbox']

    df.to_csv(result_path ,index=False)
    print(f'Output saved in {result_path}')
    return result_path
def main(json_file,path_to_ocr_res):
    result_path = convert(json_file)
    submission_path = post_processing(result_path, path_to_ocr_res)
    print(submission_path)
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='', help='path to json file')
    parser.add_argument('--path_to_ocr_res', type=str, default='./ocr/ocr_test_res.csv', help='path to OCR results')
    args = parser.parse_args()
    main(args.json_file, args.path_to_ocr_res)
