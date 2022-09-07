import json
import subprocess

def main(config = './inference/config.json', data_config = './inference/data_config.json'):
    cfg = json.load(open(config))
    # data_cfg = json.load(open(data_config))
    base = f"python train.py --epochs {cfg['epochs']} --device {cfg['device']} --batch-size {cfg['batch_size']} --data {cfg['data']} --img-size {cfg['img_size']}  --cfg {cfg['cfg']} --name {cfg['base_name']} --hyp {cfg['hyp']} --cache-images --weights '' --exist-ok"
    adv = f"python train.py --epochs {cfg['epochs']} --device {cfg['device']} --batch-size {cfg['batch_size']} --data {cfg['data']} --img-size {cfg['img_size']}  --cfg {cfg['cfg']} --name {cfg['adv_name']} --hyp {cfg['hyp']} --cache-images --weights '' --exist-ok --single-cls"
    subprocess.run(base, shell=True)
    # subprocess.run(adv, shell=True)
if __name__ == '__main__':
    main()
