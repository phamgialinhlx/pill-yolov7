
import pandas as pd
import os

import json

import argparse
from tqdm import tqdm
from loadOCR import load_OCR
import numpy as np
SIMILAR_CLS = {
    '0' : (True, [89, 99, 90, 91, 96, 3, 1, 60, 88, 48, 92]),
    '1' : (True, [89, 99, 90, 91, 96, 3, 0, 60, 88, 48, 92]),
    '2' : (False, []),
    '3' : (True, [89, 99, 90, 91, 96, 0, 1, 60, 88, 48, 92]),
    '4' : (True, []),
    '5' : (True, [59, 7, 46, 51, 45, 47, 70, 54, 63, 28, 53, 55, 57, 80, 98]),
    '6' : (True, [4, 40, 39, 86]),
    '7' : (True, [59, 46, 51, 70, 54, 63, 5, 28, 53, 55, 57, 30, 36, 45, 46, 47, 80, 98]),
    '8' : (True, [24, 63]),
    '9' : (False, [10, 82]),
    '10' : (False, [9, 82]),
    '11' : (True, []),
    '12' : (True, [13]),
    '13' : (True, [12]), # tập train có 2 viên 2 màu khác nhau, tập val có 0 viên. no hope
    '14' : (True, [58, 15, 72]), # hết thuốc chữa
    '15' : (True, [58, 14, 72]),
    '16' : (True, []), # ko có val để biết
    '17' : (True, []), # ko có val để biết
    '18' : (True, [104, 95, 68, 62]), # ko có val để biết
    '19' : (True, []),
    '20' : (True, [52]),
    '21' : (True, []), # Hơi ít val để false
    '22' : (True, []),
    '23' : (True, []),
    '24' : (True, []),
    '25' : (False, [26]),
    '26' : (False, [25, 61]),
    '27' : (True, []), # Hơi ít val để false
    '28' : (True, [59, 7, 46, 51, 45, 47, 70, 54, 63, 5, 53, 55, 57, 80, 98]),
    '29' : (False, [48]), # uy tín
    '30' : (True, []),
    '31' : (True, [41]), # Cũng uy tín nhưng hơi ít val để false
    '32' : (True, []),
    '33' : (True, []),
    '34' : (False, []), # uy tín
    '35' : (True, [65]),
    '36' : (True, [78]),
    '37' : (False, [23, 42]),
    '38' : (True, []), # Cũng uy tín mà chưa đoán mặt sau
    '39' : (True, [40, 84]),
    '40' : (False, [39]),
    '41' : (True, [31]),
    '42' : (True, []),
    '43' : (False, []), # cho học nhiều quá nên ko sai
    '44' : (True, []), # ko có val để biết
    '45' : (True, [59, 7, 46, 51, 53, 47, 70, 54, 63, 5, 28, 55, 57, 80, 98, 30, 12, 13]),
    '46' : (True, [59, 7, 53, 51, 45, 47, 70, 54, 63, 5, 28, 55, 57, 80, 98, 30]),
    '47' : (True, []),
    '48' : (True, [89, 99, 90, 91, 96, 3, 0, 1, 60, 88, 92]),
    '49' : (True, []), # ko có val để biết
    '50' : (True, []), # ko có val để biết (nhưng khả năng uy tín)
    '51' : (True, [36, 59, 7, 46, 45, 47, 70, 54, 63, 5, 28, 53, 55, 57, 78, 80, 98]),
    '52' : (False, [20]),
    '53' : (True, [59, 7, 46, 51, 45, 47, 70, 54, 63, 5, 28, 55, 57, 80, 98]),
    '54' : (True, [59, 7, 46, 51, 45, 47, 70, 53, 63, 5, 28, 55, 57, 80, 98]),
    '55' : (True, [59, 7, 46, 51, 45, 47, 70, 53, 63, 5, 28, 54, 57, 80, 98]),
    '56' : (True, []),
    '57' : (True, [59, 7, 46, 51, 45, 47, 70, 53, 63, 5, 28, 54, 55, 80, 98]),
    '58' : (True, [14, 15, 72]),
    '59' : (True, [53, 7, 46, 51, 45, 47, 70, 54, 63, 5, 28, 55, 57, 22, 45, 46, 47]),
    '60' : (True, [89, 99, 90, 91, 96, 3, 0, 1, 88, 48, 92]),
    '61' : (True, [26, 87]),
    '62' : (True, [104, 18, 95, 68]), # ko có val để biết
    '63' : (True, [22, 59, 7, 46, 51, 45, 47, 70, 54, 28, 53, 55, 57, 5, 8, 80, 98]),
    '64' : (True, [65]),
    '65' : (True, [35]),
    '66' : (True, [87]), # ko có val để biết
    '67' : (True, []), # ko có val để biết
    '68' : (True, [104, 18, 95, 62]),
    '69' : (True, []), # ko có val để biết
    '70' : (True, [30, 59, 7, 46, 51, 45, 47, 54, 63, 5, 28, 53, 55, 57, 80, 98]),
    '71' : (False, []),
    '72' : (True, [58, 15, 14]),
    '73' : (True, [75, 77, 74, 76]),
    '74' : (True, [73, 75, 77, 76]),
    '75' : (True, [33, 73, 77, 74, 76]),
    '76' : (True, [73, 75, 77, 74]),
    '77' : (True, [73, 75, 76, 74]),
    '78' : (True, [36]),
    '79' : (True, [72]),
    '80' : (True, [59, 7, 46, 51, 45, 47, 70, 54, 63, 5, 28, 55, 57, 53, 98]), # ko có val để biết mà trắng tròn nên rải hết
    '81' : (True, []),
    '82' : (True, [9, 10]),
    '83' : (False, []),
    '84' : (True, [39]),
    '85' : (True, [24]), # ko có val
    '86' : (True, [6, 4, 40, 39]), # ko có val
    '87' : (True, [66, 61, 26]),
    '88' : (True, [89, 99, 90, 91, 96, 3, 0, 1, 60, 48, 92]),
    '89' : (True, [99, 90, 91, 96, 3, 0, 1, 60, 48, 88, 92]),
    '90' : (True, [99, 89, 91, 96, 3, 0, 1, 60, 48, 88, 92]),
    '91' : (True, [99, 90, 89, 96, 3, 0, 1, 60, 48, 88, 92]),
    '92' : (True, [99, 90, 89, 96, 3, 0, 1, 60, 48, 88, 91]),
    '93' : (True, []), # ko có val
    '94' : (True, []), # ko có val
    '95' : (True, [104, 18, 68, 62]),
    '96' : (True, [99, 90, 89, 92, 3, 0, 1, 60, 48, 88, 91]), # ko có val
    '97' : (True, []),
    '98' : (True, [59, 7, 46, 51, 45, 47, 70, 54, 63, 5, 28, 53, 55, 57, 80, 98]),
    '99' : (True, [89, 99, 90, 91, 96, 3, 0, 1, 60, 88, 48, 92]), 
    '100' : (True, []), # ko đủ val để false
    '101' : (True, []), # ko có val
    '102' : (True, []), # ko có val
    '103' : (False, []),   
    '104' : (True, [18, 95, 68, 62]),
    '105' : (True, [34]),
    '106' : (True, []), # ko có val
}

def post_processing(path_to_detect_output, output_path, path_to_ocr_res, support_107 = False, reverse = False,mi=0.5, mx=1.0):
    # print('post_processing')
    df = pd.read_csv(path_to_detect_output)
    df['image_id'] = df['image_name'].apply(lambda x: x.split('_')[2])
    OCR_res = load_OCR(path_to_ocr_res)
    df = df.merge(OCR_res, on='image_id', how='left')


    sim_df = pd.DataFrame.from_dict(SIMILAR_CLS, orient='index', columns=['is_similar', 'similar_cls'])
    sim_df['class_id'] = sim_df.index.astype(int)
    df = df.merge(sim_df, on='class_id', how='left')
    add_df = pd.DataFrame(columns=df.columns)
    df['have_107'] = False
    #check if any element of array b not appear in array a
    


    # df['have_107'] = df['have_107'].apply(
    #     lambda x: True if (np.array(x['similar_cls'])) else False 
    # )
    # df = df.assign(have_107 = lambda x: (np.in1d(np.array(x['similar_cls']),np.array(x['id']))))
    # for index, row in tqdm(df.iterrows(), desc='similar_class',total=len(df)):
    #     similar_cls_row = None
    #     if row['class_id'] !=107:
    #         for id in row['similar_cls']:
    #             if id not in row['id']:
    #                 df.loc[index, 'have_107'] = True
    #                 continue
    #             similar_cls_row = row
    #         if row['have_107'] and support_107:
    #             similar_cls_row = row
    #     if similar_cls_row is not None:
    #         if reverse:
    #             similar_cls_row['confidence_score'] = 1 - similar_cls_row['confidence_score']
    #         similar_cls_row['class_id'] = 107
    #         similar_cls_row['confidence_score'] = row['confidence_score']*(mx-mi) + mi
    #         assert mx >= mi, 'need max > min'
    #         similar_cls_row = pd.DataFrame(similar_cls_row)
    #         add_df = pd.concat([add_df, similar_cls_row.T], axis=0)
    # import IPython; IPython.embed()
    
    # df = df.sort_values(by=['image_name', 'confidence_score'], ascending = [True, False])   





    # print(df)
    for index, row in df.iterrows():
        #check if row['id'] is a list
        if not isinstance(row['id'], list):
            continue
        if row['class_id'] not in row['id']:
            df.loc[index, 'class_id'] = 107
    # print(df)
    df = df.drop(columns=['id', 'filename','image_id'])
    submission_path = os.path.join( output_path,'results.csv')
    df.to_csv(submission_path, index=False)
    return submission_path
def convert(json_file, output_path):
    '''
    json_file: path to json file best_prediction.json
    return: path to csv file submission.csv
    '''
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
