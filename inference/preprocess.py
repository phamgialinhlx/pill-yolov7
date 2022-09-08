import json
import os
import subprocess
import yaml
from yaml.loader import SafeLoader
def main(config = './inference/config.json', data_config = './inference/data_config.json'):
    cfg = json.load(open(config))
    data_cfg = json.load(open(data_config))

    # Open the file and load the file
    with open(cfg['data']) as f:
        targ = yaml.load(f, Loader=SafeLoader)
        targ_train = targ['train'].rsplit('.', 1)[0]
        # targ_val = targ['val']
        targ_test = targ['test'].rsplit('.', 1)[0]
    #check if data folder is empty
    
    preprocess_infer = f"python preprocessing.py --origin_path {data_cfg['pill_infer_image_dir']} --target_path {targ_test.rsplit('/', 1)[0]}"
    preprocess_train = f"python preprocessing.py --origin_path {data_cfg['pill_train_image_dir']} --target_path {targ_train.rsplit('/',1)[0]} --task train"

    if cfg['overwrite_preprocess']:
        if os.path.exists(f"{targ_test.rsplit('/',1)[0]}/test.cache"):  
            subprocess.run(f"rm -rf {targ_test.rsplit('/',1)[0]}/test.cache", shell=True)
        preprocess_infer = preprocess_infer + " --overwrite"
        preprocess_train = preprocess_train + " --overwrite"
        subprocess.run(preprocess_infer, shell=True)
        subprocess.run(preprocess_train, shell=True)
    else:
        if not(os.path.isdir(targ_test)):
            subprocess.run(preprocess_infer, shell=True)

        if not(os.listdir(data_cfg['pill_train_image_dir'])):
            subprocess.run(preprocess_train, shell=True)
        
    preprocess_labels = f"python preprocessing.py --origin_path {data_cfg['pill_train_label_dir']} --overwrite --convert_labels --task train"
    subprocess.run(preprocess_labels, shell=True)

if __name__ == '__main__':
    main()
