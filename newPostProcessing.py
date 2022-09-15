import pandas as pd
import os
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
import matplotlib.pyplot as plt
import argparse
import yaml
from tqdm import tqdm
from ensemble import iou, is_overlap

####### OCR PROCESS

# what is pres_name of image_name (pathImage)
def take_id_from_pres(pathImage = "VAIPE_P_621_17.jpg", prefix="VAIPE_P_TEST_"):
    id_pres =  pathImage.rsplit('_')[-2]
    name_pres = prefix + id_pres
    return name_pres

# load dict ocr
def load_dict(path = './runs/ocr/drug_name_dict.csv'):
    dict_res = {}
    df = pd.read_csv(path)
    for i, row in df.iterrows():
        value, key = row['id'], row['drugname']
        if key not in dict_res.keys():
            dict_res[key] = []
        dict_res[key].append(value)
    return dict_res

# load OCR result of train/val/test set
def load_OCR_res(path = './runs/ocr/ocr_test_res.csv', pathDict = './runs/ocr/drug_name_dict.csv'):
    OCR_res = {}
    dict_res = load_dict(pathDict)
    list_drugname = [key for key in dict_res]
    df = pd.read_csv(path)
    for i, row in df.iterrows():
      row['filename'] = row['filename'][:-4]
      if row['filename'] not in OCR_res.keys():
        OCR_res[row['filename']] = []

      if type(dict_res[row['match']]) == str:
          OCR_res[row['filename']].append(dict_res[row['match']])
      else:
          for id in dict_res[row['match']]:
              OCR_res[row['filename']].append(id)

    # unique and sort
    for i in OCR_res:
        OCR_res[i] = list(set(OCR_res[i]))
    return OCR_res

# take id potential OCR for each image
def take_id_OCR(OCR_res, nameImage = "VAIPE_P_621_17.jpg", prefix = 'VAIPE_P_TEST_'):
  id_potential_in_pres = None
  name_pres = take_id_from_pres(nameImage, prefix = prefix)
  if name_pres in OCR_res.keys():
    id_potential_in_pres = OCR_res[name_pres]
    id_potential_in_pres = np.array(id_potential_in_pres).astype(int)
  return id_potential_in_pres

# process ocr to all file csv (sub -> res)
def process_ocr(main, OCR_res, prefix = 'VAIPE_P_TEST_'):
  col = main.columns
  main = main.values.tolist()
  for x in main:
    id_potential = take_id_OCR(OCR_res, x[0], prefix) # x[0] is image_name
    if x[1] not in id_potential:
      x[1] = 107
  main = pd.DataFrame(main, columns = col)
  # print(main)
  return main

##### ENS INTER
def ens_inter(main, support, iou_threshold=0.65, is_adv=True):
  isList = isinstance(main, list)
  if not isList:
    col = main.columns
    main_list = main.values.tolist()
    support_list = support.values.tolist()
  support_dict = {}
  # make main dict from main
  for x in support_list:
    if x[0] not in support_dict.keys():
      support_dict[x[0]] = []
    support_dict[x[0]].append(x)

  ens = []
  for x in main_list:
    if is_adv == True:
      x[1] = 107 # all class adv is 107
    is_over = is_overlap(x, support_dict[x[0]], iou_threshold, 0.0)
    if is_over == True:
      ens.append(x)
  if not isList:
    ens = pd.DataFrame(ens, columns=col)
  return ens

##### ENS MAIN WITH ADV
def ens_main_adv(main, adv, conf_adv=None, iou_thres=0.65, conf_main=0.01):
  adv_img_name = adv['image_name'].unique()
  main_img_name = main['image_name'].unique()

  main_img_no_pred = []
  for name in adv_img_name:
    if name not in main_img_name:
      main_img_no_pred.append(name)

  col = adv.columns
  main = main.values.tolist()
  adv = adv.values.tolist()

  main_dict = {}
  for x in main:
    if x[0] not in main_dict.keys():
      main_dict[x[0]] = []
    main_dict[x[0]].append(x)

  for x in adv:
    x[1] = 107 # x[1] is class_id
    if x[0] in main_img_no_pred:
      main.append(x)

      # if x[0] not in main_dict.keys():
      #   main_dict[x[0]] = []
      # main_dict[x[0]].append(x)
    else:
      if conf_adv is not False:
        if x[2] < conf_adv:
          continue
      is_over = is_overlap(x, main_dict[x[0]], iou_thres, conf_main) # x[0] is image_name
      if is_over == False:
        main.append(x)
        
      # if x[0] not in main_dict.keys():
      #   main_dict[x[0]] = []
      # main_dict[x[0]].append(x)


  # # main_new = []
  # # for x in main_dict:
  # #   main_new.extend(main_dict[x])
  # # main = main_new

  pred_adv_val = pd.DataFrame(main, columns = col)
  return pred_adv_val.sort_values(by=['image_name', 'confidence_score'], ascending = [True, False])

##### ENS SUPPORT CLASS
support_class_dict = {
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

def ens_main_support_class(main, support_class_dict, OCR_res, mi=0.5, mx=1.0, support_107=False, reverse=False, prefix='VAIPE_P_TEST_'):
  files = main['image_name'].unique().tolist()
  col = main.columns
  main = main.values.tolist()
  main_dict = {}

  # make main dict
  for x in main:
    if x[0] not in main_dict.keys():
      main_dict[x[0]] = []
    main_dict[x[0]].append(x)
  # print(main_dict)
  for file in tqdm(files, desc='ens_main_support_class', total = len(files)):
    # ocr
    id_potential = take_id_OCR(OCR_res, file, prefix=prefix)
    cur = main_dict[file]

    for x in cur:
      # if x[2] > 0.2 and x[2] < 0.6:
      #   continue
      if x[1] == 107: # class_id is 107
        continue
      support_cls = support_class_dict[str(x[1])][1]
      is_have107 = False
      for id in support_cls:
        if id not in id_potential:
          is_have107 = True
          continue
        new_x = x.copy()
        if reverse:
          new_x[2] = 1 - new_x[2]
        new_x[1], new_x[2] = id, new_x[2] * (mx - mi) + mi # scale
        assert mx >= mi, 'need max > min'
        main.append(new_x)
      if is_have107 and support_107:
        new_x = x.copy()
        if reverse:
          new_x[2] = 1 - new_x[2]
        new_x[1], new_x[2] = 107, new_x[2] * (mx - mi) + mi # scale
        assert mx >= mi, 'need max > min'
        main.append(new_x)
      
  main = pd.DataFrame(main, columns = col)
  main = main.sort_values(by=['image_name', 'confidence_score'], ascending = [True, False])
  # print(main)
  return main


###### NMS
def nms(main, iou_thres=0.45, min_bbox=False):
  files = main['image_name'].unique().tolist()
  col = main.columns
  main = main.sort_values(by=['image_name', 'confidence_score'], ascending = [True, False])
  main = main.values.tolist()
  main_dict = {}
  for x in main:
    if x[0] not in main_dict.keys():
      main_dict[x[0]] = []
    main_dict[x[0]].append(x)
  new_main = []
  for file in files:
    cur_main = main_dict[file]
    cur_cls = []
    for x in cur_main:
      cur_cls.append(x[1])
    cur_cls = list(set(cur_cls))
    for cls in cur_cls:
      res = []
      for x in cur_main:
        if x[1] != cls:
          continue
        ok = True
        for i, choosed in enumerate(res):
          if iou(x, choosed) >= iou_thres:
            if min_bbox:
              res[i][3] = max(res[i][3], x[3])
              res[i][4] = max(res[i][4], x[4])
              res[i][5] = min(res[i][5], x[5])
              res[i][6] = min(res[i][6], x[6])
            ok = False
            break
        if ok:
          res.append(x)
          new_main.append(x)
  new_main = pd.DataFrame(new_main, columns=col)
  new_main = new_main.sort_values(by=['image_name', 'confidence_score'], ascending = [True, False])
  return new_main

###### mAP
def mAP(main, gt):
  files = gt['image_name'].unique().tolist()
  # files = ['VAIPE_P_0_1.jpg']
  main = main.values.tolist()
  gt = gt.values.tolist()
  pred = []
  target = []
  gt_dict = {}
  main_dict = {}
  
  for x in main:
    if x[0] not in main_dict.keys():
      main_dict[x[0]] = []
    main_dict[x[0]].append(x)
  for x in gt:
    if x[0] not in gt_dict.keys():
      gt_dict[x[0]] = []
    gt_dict[x[0]].append(x)

  for file in files:
    cur_pred = []
    cur_label = []
    if file in main_dict.keys():
      cur_pred = main_dict[file]
    if file in gt_dict.keys():
      cur_label = gt_dict[file]
    boxes_pred, labels_pred, scores_pred, boxes_target, labels_target = [], [], [], [], []
    for x in cur_pred:
      boxes_pred.append([x[3], x[4], x[5], x[6]])
      scores_pred.append(x[2])
      labels_pred.append(x[1])
    for x in cur_label:
      boxes_target.append([x[3], x[4], x[5], x[6]])
      labels_target.append(x[1])
    
    boxes_pred, labels_pred, scores_pred, boxes_target, labels_target = torch.tensor(boxes_pred), torch.tensor(labels_pred), torch.tensor(scores_pred), torch.tensor(boxes_target), torch.tensor(labels_target)
    pred.append(
      dict(
        boxes=boxes_pred, scores=scores_pred, labels=labels_pred,
      )
    )
    target.append(
      dict(
        boxes=boxes_target, labels=labels_target,
      )
    )
  metric = MeanAveragePrecision(class_metrics = True)
  metric.update(pred, target)
  x = metric.compute()
  return x

def pipeline(cfg):
    
    ocr_test_res_csv = cfg['ocr']['ocr_res']
    drug_name_dict_csv = cfg['ocr']['drugname_dict']
    label = None
    if cfg['label'] is not False:
        label = pd.read_csv(cfg['label'])
    pred_preOCR = pd.read_csv(cfg['main_sub'])
    spt_adv = pd.read_csv(cfg['adv']['support'])
    main_adv = pd.read_csv(cfg['adv']['main'])

    print("Load OCR...")
    OCR_res = load_OCR_res(ocr_test_res_csv, drug_name_dict_csv)
    print("Ensemble main with support class...")
    import IPython; IPython.embed()
    res = ens_main_support_class(pred_preOCR, support_class_dict, OCR_res, 
                                mi=cfg['ens_main_support_class']['mi_scale'], 
                                mx=cfg['ens_main_support_class']['mx_scale'],
                                support_107=cfg['ens_main_support_class']['support_107'],
                                reverse=cfg['ens_main_support_class']['reverse'],
                                prefix = cfg['ocr']['prefix'])

    adv_val_ens_inter = main_adv
    if cfg['adv']['is_ens']:
        print("Ensemble adv...")
        adv_val_ens_inter = ens_inter(main_adv, spt_adv, 
                                    iou_threshold=cfg['adv']['iou_threshold'], is_adv=True)

    print("Processing OCR...")
    res = process_ocr(res, OCR_res, prefix = cfg['ocr']['prefix'])

    print("Ensemble main with adv...")
    res = ens_main_adv(res, adv_val_ens_inter,
                        conf_adv= cfg['ens_main_adv']['conf_adv'],
                        iou_thres= cfg['ens_main_adv']['iou_thres'],
                        conf_main= cfg['ens_main_adv']['conf_main'])

    print('Run NMS...')
    if cfg['nms']['is_nms']:
        res = nms(res, iou_thres=cfg['nms']['iou_thres'], min_bbox=cfg['nms']['min_bbox'])

    if label is not None:
        print(f'number of pred is {len(res)}. calculate mAP...')
        map = mAP(res, label)
        print('map', float(map['map']))
        print('map50', float(map['map_50']))
        print('map75', float(map['map_75']))
    
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='', help='path to config postProcesing')
    parser.add_argument('--save', type=str, default='cfg/postProcessing/results.csv', help='path to save')
    args = parser.parse_args()
    f = open(args.cfg)
    cfg = yaml.load(f, Loader=yaml.SafeLoader)
    res = pipeline(cfg)
    res.to_csv(args.save)



