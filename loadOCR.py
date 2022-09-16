from collections import OrderedDict
import csv
import json
import pandas as pd
# root is /data/pill/competition/Yolov7/yolov7
def load_OCR_res(path = './runs/ocr/ocr_train_res.csv'):
    OCR_res = {}
    dict_res = load_dict()
    list_drugname = [key for key in dict_res]
    with open(path) as file_in:
        next(file_in)
        for line in file_in:
            file_pres_name, _ = line.split(',', 1)
            file_pres_name = file_pres_name[:-4]
            drugname = take_key(list_drugname, line)
            if drugname == None:
                drugname = list_drugname[0]
            if file_pres_name not in OCR_res.keys():
                OCR_res[file_pres_name] = []
            if type(dict_res[drugname]) == str:
                OCR_res[file_pres_name].append(dict_res[drugname])
            else:
                for drugname in dict_res[drugname]:
                    OCR_res[file_pres_name].append(drugname)

    # sort and check unique
    for i in OCR_res:
        #OCR_res[i].sort(key = int)
        OCR_res[i] = list(set(OCR_res[i]))

    return OCR_res

def take_key(list_key, line):
    for key in list_key:
        if key in line:
            return key
    return None

def load_dict(path = './runs/ocr/drug_name_dict.csv'):
    dict_res = {}
    with open(path) as file_in:
        next(file_in)
        for line in file_in:
            value, key = line.split(',', 1)
            key = key[:-1]
            if key not in dict_res.keys():
                dict_res[key] = []
            dict_res[key].append(value)
    return dict_res

# y = load_OCR_res()
#print(y)

def take_id_from_pres(pathImage = "./vaipe_fix/images/VAIPE_P_621_17.jpg", testing=False):
    # print(pathImage)
    id_pres =  pathImage.rsplit('_')[-2]
    if testing:
        name_pres = "VAIPE_P_TEST_" + id_pres
    else:
        name_pres = "VAIPE_P_TRAIN_" + id_pres

    return name_pres

def load_OCR(path_to_OCR_res = './runs/ocr/ocr_test_res.csv'):
    # df = pd.read_csv(path_to_detect_output)
    # df['image_id'] = df['image_name'].apply(lambda x: x.split('_')[2])
    OCR_res = pd.read_csv(path_to_OCR_res)
    drug_dict = pd.read_csv('./runs/ocr/drug_name_dict.csv').groupby('drugname').id.apply(list).reset_index().rename(columns={'drugname': 'match'})
    OCR_res = OCR_res.merge(drug_dict, on='match', how='left')
    OCR_res = OCR_res.groupby('filename').agg({
        'id':'sum',
    }).reset_index()
    OCR_res['prescription'] = OCR_res['filename'].apply(lambda x: x.split('.')[0])
    return OCR_res

def main():
    # print(load_OCR_res())
    # print(take_id_from_pres())
    # print(load_dict())
    df = load_OCR().set_index('image_id').sort_index()          
    #print specific row of df
    # for i in df.iloc[20]['id']:
    #     print(i)
    #get row with index 20
    print(df.iloc['20'])
if __name__ == '__main__':
    main()
