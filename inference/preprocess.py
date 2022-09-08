from cgi import print_arguments
import json
import os
import subprocess
import yaml
from yaml.loader import SafeLoader
def main(config = './inference/config.json', data_config = './inference/data_config.json'):
    cfg = json.load(open(config))
    data_cfg = json.load(open(data_config))


    # Open the file and load the config files
    with open(cfg['data']) as f:
        targ = yaml.load(f, Loader=SafeLoader)

        targ_train = targ['train'].rsplit('.', 1)[0]
        
        targ_test = targ['test'].rsplit('.', 1)[0]
        targ_folder = targ_train.rsplit('/', 1)[0]
    

    # Preprocess the images with metadata
    preprocess_infer = f"python preprocessing.py --origin_path {data_cfg['pill_infer_image_dir']} --target_path {targ_folder}"
    preprocess_train = f"python preprocessing.py --origin_path {data_cfg['pill_train_image_dir']} --target_path {targ_folder} --task train"

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

        
        if not(os.path.isdir(os.path.join(targ_folder, 'train'))):
            subprocess.run(preprocess_train, shell=True)
        # print(targ_train)

    #Convert the labels

    preprocess_labels = f"python preprocessing.py --origin_path {data_cfg['pill_train_label_dir']} --overwrite --convert_labels --task train"
    subprocess.run(preprocess_labels, shell=True)
    
    # Add gen data
    gen_img_path = os.path.join(targ_folder, 'gen', 'images')
    gen_label_path = os.path.join(targ_folder, 'gen', 'labels')
    if not os.path.exists(gen_img_path):
        os.makedirs(gen_img_path)
    
    if os.listdir(gen_img_path) == []:
        if not os.path.exists(os.path.join(targ_folder, 'gen.zip')):
            id = '1-0qyIFhFg1IDxQcnGNVk6k-9NWyWWZ1H'
            subprocess.run(f"gdown {id} --output {os.path.join(targ_folder, 'gen.zip')}", shell=True)
        subprocess.run(['unzip', os.path.join(targ_folder, 'gen.zip'), '-d', os.path.join(targ_folder, 'gen')])
    process_gen_label = f"python tools/csv2json.py --target_path {gen_label_path} --img_path_file {os.path.join(targ_folder, 'train.txt')}"
    subprocess.run(process_gen_label, shell=True)
    
    #Merge directories
    if not os.path.exists(os.path.join(targ_folder, 'train_gen')):
        # subprocess.run(f"rm -rf {os.path.join(targ_folder, 'train_gen')}", shell=True)
        print('Processing gen...')
        subprocess.run(f"rsync -a -v {os.path.join(targ_folder, 'gen') }/ {os.path.join(targ_folder, 'train_gen')}/", shell=True)
        print('Processing train...')
        subprocess.run(f"rsync -a -v {os.path.join(targ_folder, 'train') }/ {os.path.join(targ_folder, 'train_gen')}/", shell=True)
    
    preprocess_gen_label = f"python preprocessing.py --origin_path {os.path.join(targ_folder, 'train_gen', 'labels')} --overwrite --task train_gen"
    subprocess.run(preprocess_gen_label, shell=True)

    # if not os.path.exists(os.path.join(targ_folder,'train_gen.txt')):
    #     #create train_gen.txt
    #     with open(os.path.join(targ_folder,'train_gen.txt'), 'w') as f:
    #         pass

    # subprocess.run(f"cat {os.path.join(targ_folder,'gen.txt')} > {os.path.join(targ_folder,'train.txt')} > {os.path.join(targ_folder,'train_gen.txt')}", shell=True)
if __name__ == '__main__':
    main()
