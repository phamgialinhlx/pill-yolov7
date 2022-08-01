import os
import sys 
import json 
import posixpath
from PIL import Image, ExifTags


for orientation in ExifTags.TAGS.keys():
    if ExifTags.TAGS[orientation] == 'Orientation':
        break

def exif_size(img):
    # Returns exif-corrected PIL size
    s = img.size  # (width, height)
    try:
        rotation = dict(img._getexif().items())[orientation]
        if rotation == 6:  # rotation 270
            s = (s[1], s[0])
        elif rotation == 8:  # rotation 90
            s = (s[1], s[0])
    except:
        pass

    return s

def correct_box(x):
    return max(0.0001, min(0.9999, x))

directory = '/home/pill/competition/dataset/public_train/pill/label/'
json_files = [posixpath.join(directory, f) for f in os.listdir(directory) if posixpath.isfile(posixpath.join(directory, f)) and f.endswith('.json')]
json_data = []
for i, json_file in enumerate(json_files):  
    with open(json_file) as f:
        tmp = json.load(f)
        file = open(json_file.replace(directory, './labels/').replace('json', "txt"), "w")
        img = Image.open(json_file.replace('label', "image").replace('json', 'jpg'))
        w, h = exif_size(img)
        for i in tmp:
            _w = i['w'] / w
            _h = i['h'] / h
            _x = i['x'] / w + _w / 2
            _y = i['y'] / h + _h / 2 
            # p3 = str(i['x'] + i['w'] - 1) + ' ' + str(i['y'] + i['h'] - 1)
            # p4 = str(i['x']) + ' ' + str(i['y'] + i['h'] - 1)
            s = str(i['label']) + ' ' + str(_x) + ' ' + str(_y) + ' ' + str(_w) + ' ' + str(_h)
            file.write(s + '\n')
        file.close()
#         print(file)

# print(json_data)
