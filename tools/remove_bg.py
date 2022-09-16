import cv2
import torch
import argparse
import numpy as np
import pandas as pd
from rembg.bg import remove

def read_file_csv(file_path):
    df = pd.read_csv(file_path)
    return df

def crop_image(img, xyxy):
    x_min = xyxy[0]
    y_min = xyxy[1]
    x_max = xyxy[2]
    y_max = xyxy[3]
    return img[y_min:y_max, x_min:x_max, :]

def to_gray(img):
    # img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(img, 10, 230, cv2.THRESH_BINARY)
    return ret, thresh

def get_countour(image):
    # B, G, R channel splitting
    if (image.shape[-1] == 4):
        blue, green, red, _ = cv2.split(image)
    else:
        blue, green, red = cv2.split(image)
    # detect contours using blue channel and without thresholding
    contours1, hierarchy1 = cv2.findContours(image=blue, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # detect contours using green channel and without thresholding
    contours2, hierarchy2 = cv2.findContours(image=green, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    # detect contours using red channel and without thresholding
    contours3, hierarchy3 = cv2.findContours(image=red, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    return contours1, contours2, contours3

def contour_to_box(contour):
    x_min = float('inf')
    x_max = float('-inf')
    y_min = float('inf')
    y_max = float('-inf')

    for i, c in enumerate(contour):
        x = np.array(c)[:, 0 , 0]
        y = np.array(c)[:, 0 , 1]
        x1 = min(x)
        x2 = max(x) 
        y1 = min(y)
        y2 = max(y)
        x_min = min(x_min, x1)
        x_max = max(x_max, x2)
        y_min = min(y_min, y1)
        y_max = max(y_max, y2)
    w = x_max - x_min
    h = y_max - y_min
    box = np.array([x_min, y_min, w, h])
    return box

def validateSize(image, threshold=4000):
    s = image.shape[0] * image.shape[1]
    return s > threshold

def validateS(origin, new, threshold=0.6):
    s_origin = origin[2] * origin[3]
    s_new = new[2] * new[3]
    return s_new / s_origin < threshold

def mergeBBox(boxes):
    xyxy = np.array([[x , y, x + w, y + h] for x, y, w, h in boxes])
    x_min = min(xyxy[:, 0]) 
    y_min = min(xyxy[:, 1]) 
    x_max = max(xyxy[:, 2]) 
    y_max = max(xyxy[:, 3]) 
    return [x_min, y_min, x_max - x_min, y_max - y_min]

def xywh2xyxy(box):
    return [box[0], box[1], box[2] + box[0], box[3] + box[1]]

def xyxy2xywh(box):
    return [box[0], box[1], box[2] - box[0], box[3] - box[1]]

def get_new_bbox(image, kernel, origin_box):
    new_boxes = []
    for k in kernel:
        # Apply Blur
        blur = cv2.GaussianBlur(image, k, 0)
        fg_img = remove(blur)
        # Image to Grayscale
        ret, _gray = to_gray(fg_img)
        # Get countour
        c1, c2, c3 = get_countour(_gray)
        # Get countour_box
        box_img_cropped = contour_to_box(c1) 
        new_box = [box_img_cropped[0], box_img_cropped[1], box_img_cropped[2], box_img_cropped[3]]
        if new_box[0] == torch.tensor(float("inf")) or new_box[1] == torch.tensor(float("inf")) \
             or new_box[2] == torch.tensor(float("-inf")) or new_box[3] == torch.tensor(float("-inf")):
            w = image.shape[1]
            h = image.shape[0]
            new_box = [0, 0, w, h]
        elif validateS(origin_box, new_box):
            w = image.shape[1]
            h = image.shape[0]
            new_box = [0, 0, w, h]
        new_boxes.append(new_box)
    final_box = mergeBBox(new_boxes)
    final_box[0] = final_box[0] + origin_box[0]
    final_box[1] = final_box[1] + origin_box[1]
    final_box = xywh2xyxy(final_box)
    return final_box

def main(file_path, img_path_file, save_path, kernel=[(15, 15), (5, 5)]):
    df = read_file_csv(file_path)
    new_results = []
    for i, row in df.iterrows():
        image_name, class_id, confidence_score, x_min, y_min, x_max, y_max = row['image_name'], row['class_id'], row['confidence_score'], row['x_min'], row['y_min'], row['x_max'], row['y_max']
        img_path = img_path_file + image_name
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        origin_box = [x_min, y_min, x_max, y_max]
        cropped_image = crop_image(img, origin_box)
          
        if (validateSize(cropped_image)):
            final_box = get_new_bbox(cropped_image, kernel, xyxy2xywh(origin_box))
        else:
            final_box = origin_box
        new_results.append([image_name, class_id, confidence_score, final_box[0], final_box[1], final_box[2], final_box[3]])
         
    # Export new results to csv
    cols = ['image_name', 'class_id', 'confidence_score', 'x_min', 'y_min', 'x_max', 'y_max']
    new_df = pd.DataFrame(new_results, columns=cols)
    new_df.to_csv(save_path, index=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, help='path to csv predict file')
    parser.add_argument('--img_path_file', type=str, default='/data/pill/competition/Yolov7/yolov7/vaipe_exif/test/images/', help='path of the file that contains path to the images')
    parser.add_argument('--save_path', type=str, default='./a.csv', help='path to save csv')
    
    args = parser.parse_args()

    main(args.file_path, args.img_path_file, args.save_path)
