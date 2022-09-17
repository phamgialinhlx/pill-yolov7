import pandas as pd
import os
import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
from tqdm import tqdm
from ensemble import iou, is_overlap

##### ENS MAIN WITH SIMILAR CLASS
SIMILAR_DICT = {
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
    '13' : (True, [12]),
    '14' : (True, [58, 15, 72]),
    '15' : (True, [58, 14, 72]),
    '16' : (True, []),
    '17' : (True, []),
    '18' : (True, [104, 95, 68, 62]),
    '19' : (True, []),
    '20' : (True, [52]),
    '21' : (True, []),
    '22' : (True, []),
    '23' : (True, []),
    '24' : (True, []),
    '25' : (False, [26]),
    '26' : (False, [25, 61]),
    '27' : (True, []),
    '28' : (True, [59, 7, 46, 51, 45, 47, 70, 54, 63, 5, 53, 55, 57, 80, 98]),
    '29' : (False, [48]),
    '30' : (True, []),
    '31' : (True, [41]),
    '32' : (True, []),
    '33' : (True, []),
    '34' : (False, []),
    '35' : (True, [65]),
    '36' : (True, [78]),
    '37' : (False, [23, 42]),
    '38' : (True, []),
    '39' : (True, [40, 84]),
    '40' : (False, [39]),
    '41' : (True, [31]),
    '42' : (True, []),
    '43' : (False, []),
    '44' : (True, []),
    '45' : (True, [59, 7, 46, 51, 53, 47, 70, 54, 63, 5, 28, 55, 57, 80, 98, 30, 12, 13]),
    '46' : (True, [59, 7, 53, 51, 45, 47, 70, 54, 63, 5, 28, 55, 57, 80, 98, 30]),
    '47' : (True, []),
    '48' : (True, [89, 99, 90, 91, 96, 3, 0, 1, 60, 88, 92]),
    '49' : (True, []),
    '50' : (True, []),
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
    '62' : (True, [104, 18, 95, 68]),
    '63' : (True, [22, 59, 7, 46, 51, 45, 47, 70, 54, 28, 53, 55, 57, 5, 8, 80, 98]),
    '64' : (True, [65]),
    '65' : (True, [35]),
    '66' : (True, [87]),
    '67' : (True, []),
    '68' : (True, [104, 18, 95, 62]),
    '69' : (True, []),
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
    '80' : (True, [59, 7, 46, 51, 45, 47, 70, 54, 63, 5, 28, 55, 57, 53, 98]),
    '81' : (True, []),
    '82' : (True, [9, 10]),
    '83' : (False, []),
    '84' : (True, [39]),
    '85' : (True, [24]),
    '86' : (True, [6, 4, 40, 39]),
    '87' : (True, [66, 61, 26]),
    '88' : (True, [89, 99, 90, 91, 96, 3, 0, 1, 60, 48, 92]),
    '89' : (True, [99, 90, 91, 96, 3, 0, 1, 60, 48, 88, 92]),
    '90' : (True, [99, 89, 91, 96, 3, 0, 1, 60, 48, 88, 92]),
    '91' : (True, [99, 90, 89, 96, 3, 0, 1, 60, 48, 88, 92]),
    '92' : (True, [99, 90, 89, 96, 3, 0, 1, 60, 48, 88, 91]),
    '93' : (True, []),
    '94' : (True, []), 
    '95' : (True, [104, 18, 68, 62]),
    '96' : (True, [99, 90, 89, 92, 3, 0, 1, 60, 48, 88, 91]), 
    '97' : (True, []),
    '98' : (True, [59, 7, 46, 51, 45, 47, 70, 54, 63, 5, 28, 53, 55, 57, 80, 98]),
    '99' : (True, [89, 99, 90, 91, 96, 3, 0, 1, 60, 88, 48, 92]), 
    '100' : (True, []), 
    '101' : (True, []), 
    '102' : (True, []), 
    '103' : (False, []),   
    '104' : (True, [18, 95, 68, 62]),
    '105' : (True, [34]),
    '106' : (True, []),
}

####### OCR PROCESS

# what is pres_name of image_name (pathImage)

# load OCR result of train/val/test set

##### ENS INTER
def ens_inter(adv_name, iou_threshold=0.65, is_ens=True):

  main = pd.read_csv(f"./runs/test/{adv_name}_0/submission_OCR.csv")
  support = pd.read_csv(f"./runs/test/{adv_name}_1/submission_OCR.csv")
  main = main.drop(columns=['id'])
  support = support.drop(columns=['id'])
  if not is_ens:
    return main
  print("ensemble adv with adv...")
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
    x[1] = 107 # all class adv is 107
    is_over = is_overlap(x, support_dict[x[0]], iou_threshold, 0.0)
    if is_over == True:
      ens.append(x)
  ens = pd.DataFrame(ens, columns=col)
  return ens

##### ENS MAIN WITH ADV
def ens_main_adv(main, adv, conf_adv=False, iou_thres=0.65, conf_main=0.01):
  # input
  ### main: pandas
  ### adv: pandas
  # output
  ### ens main adv: pandas
  print("ensemble main with adv...")
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
    x[1] = 107 # x[1] is class_id, all adv class is 107
    if x[0] in main_img_no_pred:
      main.append(x)
    else:
      if conf_adv is not False:
        if x[2] < conf_adv: # x[2] is conf.
          continue
      is_over = is_overlap(x, main_dict[x[0]], iou_thres, conf_main) # x[0] is image_name
      if is_over == False:
        main.append(x)

  pred_adv_val = pd.DataFrame(main, columns = col)
  return pred_adv_val.sort_values(by=['image_name', 'confidence_score'], ascending = [True, False])


def ens_main_similar_class(main, similar_class_dict, dict_OCR, mi=0.5, mx=1.0, support_107=False, reverse=False):
   # input
  ### main: pandas
  ### similar_class_dict
  # output
  ### ens main with similar class: pandas
  print("ensemble main with similar class...")
  files = main['image_name'].unique().tolist()
  main = main.drop(columns=['id'])
  col = main.columns
  main = main.values.tolist()
  main_dict = {}

  # make main dict
  for x in main:
    if x[0] not in main_dict.keys():
      main_dict[x[0]] = []
    main_dict[x[0]].append(x)
  # print(main_dict)
  for file in tqdm(files, desc='ens_main_similar_class', total = len(files)):
    # ocr
    id_potential = dict_OCR[file]
    cur = main_dict[file]

    for x in cur:
      # if x[2] > 0.2 and x[2] < 0.6:
      #   continue
      if x[1] == 107: # class_id is 107
        continue
      similar_cls = similar_class_dict[str(x[1])][1]
      is_have107 = False
      for id in similar_cls:
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
def nms(main, iou_thres=0.45, min_bbox=False, is_nms=True):
  if not is_nms:
    return main
  print("nms...")
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
  print("map...")
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

def ens_concat_nms(base_name, base_weights, iou_thres, min_bbox):
  n_base = len(base_weights.split(' '))
  print(f"ensemble {n_base} base...")
  base = None
  for i in range(n_base):
    base_path = f"./runs/test/{base_name}_{i}/submission_OCR.csv"
    df = pd.read_csv(base_path)
    if base is None:
      base = df
    else:
      base = pd.concat([base, df], axis=0)
  base = nms(base, iou_thres=iou_thres, min_bbox=min_bbox)
  return base

def dict_ocr(base, adv_name):
  print("produce dict...")
  adv = pd.read_csv(f"./runs/test/{adv_name}_0/submission_OCR.csv")
  dict = {}
  for i, row in adv.iterrows():
    if row['image_name'] not in dict:
      dict[row['image_name']] = []
      if not isinstance(row['id'], str):
        continue
      id = str2list(row['id'])
      dict[row['image_name']] = id

  for i, row in base.iterrows():
    if row['image_name'] not in dict:
      dict[row['image_name']] = []
      if not isinstance(row['id'], str):
        continue
      id = str2list(row['id'])
      dict[row['image_name']] = id

  return dict
  
def str2list(str):
  str = str[1:-1]
  if str == '':
    return []
  else:
    res = []
    str = str.strip(" ").split(",")
    for i in str:
      res.append(int(i))
    return res

def process_ocr(main, dict_OCR):
  print("process ocr...")
  col = main.columns
  main = main.values.tolist()
  for x in main:
    id_potential = dict_OCR[x[0]] # x[0] is image_name
    if x[1] not in id_potential:
      x[1] = 107
  main = pd.DataFrame(main, columns = col)
  return main

def scale(main, mi=0.5, mx=1.0):
  print("scale...")
  col = main.columns
  main = main.values.tolist()
  for x in main:
    assert mx >= mi, 'need max > min'
    x[2] = x[2] * (mx - mi) + mi # x[2] is conf 
  main = pd.DataFrame(main, columns = col)
  return main
  
  

def pipeline(cfg):
    cfg = json.load(open(cfg))
    print(cfg)
    base = ens_concat_nms(cfg["base_name"], cfg['base_weights'],
                          iou_thres=cfg["nms"]["iou_thres_start"],
                          min_bbox=cfg["nms"]["min_bbox"])
    adv = ens_inter(cfg["adv_name"], 
                    is_ens = cfg["ens_adv"]["is_ens"], 
                    iou_threshold = cfg["ens_adv"]["iou_threshold"])
    base = scale(base, mi = cfg["normal_scale_min"], mx = cfg["normal_scale_max"])
    adv = scale(adv, mi = cfg["normal_scale_min"], mx = cfg["normal_scale_max"])
    dict_OCR = dict_ocr(base, adv_name=cfg["adv_name"])
    base = ens_main_similar_class(base, SIMILAR_DICT, dict_OCR=dict_OCR,
                                                      mi=cfg["ens_main_similar_class"]["mi_scale"],
                                                      mx=cfg["ens_main_similar_class"]["mx_scale"],
                                                      support_107=cfg["ens_main_similar_class"]["support_107"],
                                                      reverse=cfg["ens_main_similar_class"]["reverse"])
    res = ens_main_adv(base, adv, conf_adv = cfg["ens_main_adv"]["conf_adv"],
                        iou_thres=cfg["ens_main_adv"]["iou_thres"],
                        conf_main=cfg["ens_main_adv"]["conf_main"])
    res = process_ocr(res, dict_OCR)
    res = nms(res, is_nms=cfg["nms"]["is_nms"],
                iou_thres=cfg["nms"]["iou_thres_end"],
                min_bbox=cfg["nms"]["min_bbox"])

    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', type=str, default='./inference/config.json', help='path to config json')
    parser.add_argument('--save', type=str, default='cfg/postProcessing/results.csv', help='path to save')
    args = parser.parse_args()
    res = pipeline(args.cfg)
    res.to_csv(args.save, index = False)



