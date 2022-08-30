import json
import os
import subprocess

def main(config = './inference/config.json', data_config = './inference/data_config.json'):
    cfg = json.load(open(config))
    data_cfg = json.load(open(data_config))
    
    #check if data folder is empty
    
    preprocess_infer = f"python preprocessing.py --origin_path {data_cfg['pill_infer_image_dir']}"
    preprocess_train = f"python preprocessing.py --origin_path {data_cfg['pill_train_image_dir']} --task train"

    if cfg['overwrite_preprocess']:
        preprocess_infer = preprocess_infer + " --overwrite"
        preprocess_train = preprocess_train + " --overwrite"
        subprocess.run(preprocess_infer, shell=True)
        subprocess.run(preprocess_train, shell=True)
    else:
        if not(os.listdir(data_cfg['pill_infer_image_dir'])):
            subprocess.run(preprocess_infer, shell=True)

        if not(os.listdir(data_cfg['pill_train_image_dir'])):
            subprocess.run(preprocess_train, shell=True)
        
    preprocess_labels = f"python preprocessing.py --origin_path {data_cfg['pill_train_label_dir']} --overwrite --convert_labels --task train"
    subprocess.run(preprocess_labels, shell=True)

if __name__ == '__main__':
    main()
