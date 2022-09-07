import os
import sys 
import json 
import posixpath
import sys
from PIL import Image
sys.path.append(os.path.dirname(os.path.abspath(__file__)).rsplit('/',1)[0])
from utils.datasets import exif_size
def correct_box(x):
    return max(0.0001, min(0.9999, x))
def main(argv):
    # read sys.argv
    # argv[1] is the server
    # argv[2] is the overwrite flag
    
    if len(argv) < 2:
        print("Usage: convert_labels.py choose_server(2080 or v100)")
        return
    overwrite = False
    if len(argv) == 3:
        overwrite = argv[2]
    server = argv[1]

    mapping = {'2080': '/data/pill/competition/dataset/public_train/pill/label', 'v100': '/home/pill/competition/dataset/public_train/pill/label', 'colab' : './content/drive/Shareddrives/Data/public_train/public_train/pill/label'}
    directory = mapping[server]
    files = os.listdir(directory)
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
                s = str(i['label']) + ' ' + str(_x) + ' ' + str(_y) + ' ' + str(_w) + ' ' + str(_h)
                file.write(s + '\n')
            print(file)
            file.close()
if __name__ == '__main__':
    main(sys.argv)