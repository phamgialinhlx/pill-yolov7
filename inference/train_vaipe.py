import json
import subprocess

def main(config = './inference/config.json', data_config = './inference/data_config.json'):
    cfg = json.load(open(config))
    data_cfg = json.load(open(data_config))
    train = f"python train.py --workers 8 --device {cfg['device']} --batch-size {cfg['batch_size']} --data {cfg['data']} --img-size {cfg['img_size']}  --cfg {cfg['cfg']} --name yolov7x-vaipe --hyp {cfg['hyp']} --cache-images --weights '' --notest"
    subprocess.run(train, shell=True)
if __name__ == '__main__':
    main()
