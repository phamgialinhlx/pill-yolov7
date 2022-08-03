import csv
import json
# root is /data/pill/competition/Yolov7/yolov7
def load_OCR_res(path = './ocr/ocr_train_res.csv'):
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

def load_dict(path = './ocr/drug_name_dict.csv'):
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

def take_id_from_pres(pathImage = "./vaipe/images/VAIPE_P_621_17.jpg", testing=False):
    # print(pathImage)
    _, __, id_pres, ___ =  pathImage.split('_')
    if testing:
        name_pres = "VAIPE_P_TEST_" + id_pres
    else:
        name_pres = "VAIPE_P_TRAIN_" + id_pres

    return name_pres
#print(take_id_from_pres())

# target = torch.rand(32, 6)
# target[1:4, 1] = 107.00000000
# target = target[target[:, 1] != 107.0, :]
#print(target.shape)
