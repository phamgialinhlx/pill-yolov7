import json
import subprocess

def main(config = './inference/config.json', data_config = './inference/data_config.json'):
    cfg = json.load(open(config))
    data_cfg = json.load(open(data_config))
    build_dict = f"python build_dict.py --data_dir {data_cfg['prescription_train_label_dir']}"
    ocr = f"python ocr.py --extract_ocr --data_dir {data_cfg['prescription_infer_image_dir']}"
    preprocess = f"python preprocessing.py --origin_path {data_cfg['pill_infer_image_dir']} --overwrite"
    subprocess.run(build_dict, shell=True)
    subprocess.run(ocr, shell=True)
    subprocess.run(preprocess, shell=True)
if __name__ == '__main__':
    main()
