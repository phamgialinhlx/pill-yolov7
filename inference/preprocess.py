import json
from os import pread
import subprocess

def main(config = './inference/config.json', data_config = './inference/data_config.json'):
    cfg = json.load(open(config))
    data_cfg = json.load(open(data_config))
    
    preprocess_infer = f"python preprocessing.py --origin_path {data_cfg['pill_infer_image_dir']} --overwrite"
    subprocess.run(preprocess_infer, shell=True)

    preprocess_train = f"python preprocessing.py --origin_path {data_cfg['pill_train_image_dir']} --overwrite --task train"
    subprocess.run(preprocess_train, shell=True)

    preprocess_labels = f"python preprocessing.py --origin_path {data_cfg['pill_train_label_dir']} --overwrite --convert_labels --task train"
    subprocess.run(preprocess_labels, shell=True)
if __name__ == '__main__':
    main()
