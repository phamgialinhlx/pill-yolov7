import json
import subprocess
import os
def main(config = './inference/config.json', data_config = './inference/data_config.json'):
    cfg = json.load(open(config))
    data_cfg = json.load(open(data_config))
    base_weights = cfg['base_weights'].split(' ',)
    if base_weights==['']:
        base_weights = []
    # print(base_weights)
    
    adv_weights = cfg['adv_weights'].split(' ')
    if adv_weights==['']:
        adv_weights = []
    # print(adv_weights)
    for i, base_weight in enumerate(base_weights):
        # print(f"Running inference on {base_weight} at {i}")
        base_name = f"{cfg['base_name']}_{i}"
        base = f"python test.py --weight {base_weight} --name {base_name} --exist-ok --device {cfg['device']} --data {cfg['data']} --img-size {cfg['img_size']} --batch-size {cfg['batch_size']} --save-json --task test"
        post_processing_base = f"python postProcessing.py --json_file ./runs/test/{base_name}/best_predictions.json --pill_pres_map {data_cfg['pill_pres_map']}"
        subprocess.run(base, shell=True)
        subprocess.run(post_processing_base, shell=True)
    
    for i, adv_weight in enumerate(adv_weights):
        adv_name = f"{cfg['adv_name']}_{i}"
        adv = f"python test.py --weight {adv_weight} --name {adv_name} --exist-ok --device {cfg['device']} --data {cfg['data']} --img-size {cfg['img_size']} --batch-size {cfg['batch_size']} --save-json --task test"
        post_processing_adv = f"python postProcessing.py --json_file ./runs/test/{adv_name}/best_predictions.json --pill_pres_map {data_cfg['pill_pres_map']}"
        subprocess.run(adv, shell=True)
        subprocess.run(post_processing_adv, shell=True)
    ensemble = f"python ensemble.py --base_model ./runs/test/base/results.csv --adv_model ./runs/test/adv/results.csv --exist_ok"
    if not os.path.exists('./runs/ensemble'):
        os.makedirs('./runs/ensemble')

    ensemble = f"python ensemble_postprocess.py --cfg {config} --save './runs/ensemble/results.csv'"
    subprocess.run(ensemble, shell=True)
if __name__ == '__main__':
    main()
