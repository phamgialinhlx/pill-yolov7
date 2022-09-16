from doctest import OutputChecker
import pandas as pd
import os

import json

import argparse

from loadOCR import load_OCR

def pres_pill_map(pill_pres_map_path):
    pres_pill_map = json.load(open(pill_pres_map_path))
    pill_pres_map = dict()
    for k, v in pres_pill_map.items():
        for _v in v:
            pill_pres_map[_v] = k

    df = pd.DataFrame.from_dict(pill_pres_map, orient='index', columns=['prescription'])
    df['pill'] = df.index
    df.reset_index(drop=True, inplace=True)
    df['postfix'] = df['pill'].str.split('.').str[-1]
    df['image_id'] = df['pill'].str.split('.').str[0]
    df = df[['image_id', 'prescription', 'postfix']]
    return df

def convert(json_file, output_path, mapping):
    '''
    input json file best_predictions.json
    output (dataframe,submission.csv_path)
    '''
    submission_path = os.path.join(output_path, 'submission.csv')
    with open(json_file) as f:
        data =  json.load(f)
        df = pd.DataFrame(data)
    df[['x_min','y_min','x_max','y_max']] = pd.DataFrame(df.bbox.tolist(), index= df.index)
    df.rename(columns ={'category_id':'class_id','score':'confidence_score'}, inplace = True)
    del df['bbox']
    df = df.merge(mapping, on='image_id', how='left')
    df['image_name'] = df['image_id'] + '.' + df['postfix']
    backup = df.copy()
    df = df[['image_name','class_id','confidence_score','x_min','y_min','x_max','y_max']]
    df.to_csv(submission_path ,index=False)
    print(f'Output saved in {submission_path}')
    return backup, submission_path
def post_processing(path_to_detect_output, output_path, path_to_ocr_res, df = None):
    '''
    input csv file submission.csv
    output (dataframe, results.csv_path)
    '''
    if df is None:
        df = pd.read_csv(path_to_detect_output)
    OCR_res = load_OCR(path_to_ocr_res)
    df = df.merge(OCR_res, on='prescription', how='left')
    # # print(df)
    # for index, row in df.iterrows():
    #     #check if row['id'] is a list
    #     if not isinstance(row['id'], list):
    #         continue
    #     if row['class_id'] not in row['id']:
    #         df.loc[index, 'class_id'] = 107
    # # print(df)
    # df = df.drop(columns=['id', 'filename','image_id'])

    backup = df[['image_name','class_id','confidence_score','x_min','y_min','x_max','y_max','id']]
    df = df[['image_name','class_id','confidence_score','x_min','y_min','x_max','y_max']]
    result_path = os.path.join( output_path,'results.csv')
    df.to_csv(result_path, index=False)
    print(f'Output saved in {result_path}')
    return backup, result_path
def main(json_file,output_path, path_to_ocr_res, pill_pres_map):
    df,submission_path= convert(json_file, output_path, pres_pill_map(pill_pres_map))
    df2, _ =post_processing(submission_path,output_path, path_to_ocr_res, df)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--json_file', type=str, default='', help='path to json file')
    parser.add_argument('--path_to_ocr_res', type=str, default='./runs/ocr/ocr_test_res.csv', help='path to OCR results')
    parser.add_argument('--output_path', type=str, default='', help='path to output file')
    parser.add_argument('--pill_pres_map', type=str, help='path to pill_pres_map.json')
    args = parser.parse_args()
    output_path = args.output_path
    if output_path == '':
        output_path = args.json_file.rsplit('/', 1)[0]
    main(args.json_file, output_path ,args.path_to_ocr_res, args.pill_pres_map)
