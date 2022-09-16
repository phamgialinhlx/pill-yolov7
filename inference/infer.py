import json
import subprocess

def main(config = './inference/config.json', data_config = './inference/data_config.json'):
    cfg = json.load(open(config))
    data_cfg = json.load(open(data_config))
    base = f"python test.py --weight {cfg['base_weights']} --exist-ok --device {cfg['device']} --name base --data {cfg['data']} --img-size {cfg['img_size']} --batch-size {cfg['batch_size']} --save-json --task test --no-trace"
    
    adv = f"python test.py --weight {cfg['adv_weights']} --exist-ok --device {cfg['device']} --name adv --data {cfg['data']} --img-size {cfg['img_size']} --batch-size {cfg['batch_size']} --save-json --task test"
    post_processing_base = f"python postProcessing.py --pill_pres_map {data_cfg['pill_pres_map']} --json_file ./runs/test/base/best_predictions.json"
    post_processing_adv = f"python postProcessing.py --pill_pres_map {data_cfg['pill_pres_map']} --json_file ./runs/test/adv/best_predictions.json"
    ensemble = f"python ensemble.py --base_model ./runs/test/base/results.csv --adv_model ./runs/test/adv/results.csv --exist_ok"

    subprocess.run(post_processing_base, shell=True)
    subprocess.run(post_processing_adv, shell=True)
    subprocess.run(ensemble, shell=True)
if __name__ == '__main__':
    main()
