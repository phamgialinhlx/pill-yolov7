import pandas as pd
import os
def post_processing(path_to_detect_output, path_to_OCR_res='./ocr/ocr_test_res.csv'):
    df = pd.read_csv(path_to_detect_output)
    df['image_id'] = df['image_name'].apply(lambda x: x.split('_')[2])
    OCR_res = pd.read_csv(path_to_OCR_res)
    drug_dict = pd.read_csv('./ocr/drug_name_dict.csv').groupby('drugname').id.apply(list).reset_index().rename(columns={'drugname': 'match'})
    OCR_res = OCR_res.merge(drug_dict, on='match', how='left')
    OCR_res = OCR_res.groupby('filename').agg({
        'id':'sum',
    }).reset_index()
    OCR_res['image_id'] = OCR_res['filename'].apply(lambda x: x[:-4].split('_')[-1])
    df = df.merge(OCR_res, on='image_id', how='left')
    # print(df.head())

    for index, row in df.iterrows():
        if row['class_id'] not in row['id']:
            df.loc[index, 'class_id'] = 107
    
    df = df.drop(columns=['id', 'filename','image_id'])
    # print(df.head())
    submission_path = path_to_detect_output.replace('submission.csv', 'results.csv')
    df.to_csv(submission_path, index=False)
    return submission_path
# post_processing('runs/detect/exp22/results.csv')
def convert(json_file):
    result_path = os.path.join(json_file.rsplit('/', 1)[0], 'submission.csv')
    with open(json_file) as f:
        data =  json.load(f)
        df = pd.DataFrame(data)
    df[['x_min','y_min','x_max','y_max']] = pd.DataFrame(df.bbox.tolist(), index= df.index)
    # df['image_id'] = df['image_id']+'.jpg'
    df.rename(columns ={'image_id':'image_name','category_id':'class_id','score':'confidence_score'}, inplace = True)
    df['image_name'] = df['image_name'] + '.jpg'
    del df['bbox']

    df.to_csv(result_path ,index=False)
    print(f'Output saved in {result_path}')
    return result_path
def main():
    result_path = convert('runs/test/yolov7_5TN_test14/best_predictions.json')
    submission_path = post_processing(result_path)
    print(submission_path)
if __name__ == '__main__':
    main()