import pandas as pd
import argparse
import os
from pathlib import Path
from utils.general import increment_path
def is_overlap(box, gtboxes, iou_thres, conf_thres):
    area_box = (box[5] - box[3] + 1) * (box[6] - box[4] + 1)
    for gtbox in gtboxes:
        if gtbox[2] < conf_thres:
            continue
        area_gtbox = (gtbox[5] - gtbox[3] + 1) * (gtbox[6] - gtbox[4] + 1)
        x1 = max(box[3], gtbox[3])
        y1 = max(box[4], gtbox[4])
        x2 = min(box[5], gtbox[5])
        y2 = min(box[6], gtbox[6])
        w = max(0, x2 - x1 + 1)
        h = max(0, y2 - y1 + 1)
        inter = (w * h) / (area_box + area_gtbox - w * h)
        if inter >= iou_thres:
            return True
    return False
def ensemble(base, adv, iou_thres, conf_thres, data_path, save_path):
    base_pred_imgs = base['image_name'].unique()
    df = pd.read_csv(data_path, header=None)
    #split dataframe by delimiter '/'
    df[0] = df[0].str.split('/', expand=True)[-1]
    print(df.head())
    image_names = []
    print(image_names)
    name_loss = []
    for name in image_names:
        if name not in base_pred_imgs:
            name_loss.append(name)

    col = base.columns
    res = base.values.tolist()
    adv = adv.values.tolist()

    res_img = {}
    for x in res:
        if x[0] not in res_img.keys():
            res_img[x[0]] = []
        res_img[x[0]].append(x)

    for x in adv:
        if x[0] in name_loss:
            x[1] = 107
            #x[2] = x[2] * 0.002
            res.append(x)
        else:
            if is_overlap(x, res_img[x[0]], iou_thres, conf_thres) == False:
                res.append(x)

    #res = applyNMS(res, 0.85)

    for x in res:
    # x[2] = x[2]
        x[2] = x[2] * 0.2 + 0.8
    # if(x[0] == img_name):
    #print(x)

    res = pd.DataFrame(res, columns = col)
    # print(len(res))
    res.to_csv(save_path, index = False) 
    print("save to {}".format(save_path))
    res = res.values.tolist()
    return res
if __name__ == '__main__':
    # parse arguments path to csv file
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_model', type=str, default='', help='path to result.csv file of base model')
    parser.add_argument('--adv_model', type=str, default='', help='path to result.csv file of adv model')
    parser.add_argument('--iou_thres', type=float, default=0.65, help='iou threshold')
    parser.add_argument('--conf_thres', type=float, default=0.005, help='confidence threshold')
    parser.add_argument('--data_path', type=str, default='./vaipe_exif/test.txt', help='path to pill image data. eg public_test/pill')
    parser.add_argument('--save_dir', type=str, default='./runs', help='path to save ensemble.csv file')
    parser.add_argument('--save_name', type=str, default='ensemble', help='name of ensemble.csv file')
    parser.add_argument('--exist_ok', action='store_true', help='overwrite existing ensemble.csv file')


    args = parser.parse_args()
    # read base model result.csv file
    base_model_result = pd.read_csv(args.base_model)
    # read adv model result.csv file
    adv_model_result = pd.read_csv(args.adv_model)
    # ensemble
    save_dir = Path(increment_path(Path(args.save_dir) / args.save_name, exist_ok=args.exist_ok))
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = str(save_dir / "results.csv")
    ensemble_result = ensemble(base_model_result, adv_model_result, args.iou_thres, args.conf_thres, args.data_path, save_path)
    # print(ensemble_result)
    
