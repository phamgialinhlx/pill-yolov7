from doctest import OutputChecker
import pandas as pd
import os

import json

import argparse

from loadOCR import load_OCR

def post_processing(path_to_detect_output, output_path, path_to_ocr_res):
    df = pd.read_csv(path_to_detect_output)
    df['image_id'] = df['image_name'].apply(lambda x: x.split('_')[2])
    OCR_res = load_OCR(path_to_ocr_res)
    df = df.merge(OCR_res, on='image_id', how='left')
    # print(df)
    for index, row in df.iterrows():
        #check if row['id'] is a list
        if not isinstance(row['id'], list):
            continue
        if row['class_id'] not in row['id']:
            df.loc[index, 'class_id'] = 107
    print(df)
    df = df.drop(columns=['id', 'filename','image_id'])
    submission_path = os.path.join( output_path,'results.csv')
    df.to_csv(submission_path, index=False)
    return submission_path
def convert(json_file, output_path):
    result_path = os.path.join(output_path, 'submission.csv')
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

def main(json_file,output_path, path_to_ocr_res):
    result_path = convert(json_file, output_path)
    submission_path = post_processing(result_path,  output_path, path_to_ocr_res)
    print(submission_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='', help='path to json file')
    parser.add_argument('--path_to_ocr_res', type=str, default='./runs/ocr/ocr_test_res.csv', help='path to OCR results')
    parser.add_argument('--output_path', type=str, default='', help='path to output file')
    args = parser.parse_args()
    output_path = args.output_path
    if output_path == '':
        output_path = args.json_file.rsplit('/', 1)[0]
    main(args.json_file, output_path ,args.path_to_ocr_res)
