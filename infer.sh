python test.py --weight ./runs/train/yolov7-exif-gen3/weights/best.pt --exist-ok --device 0 --name base --data data/vaipe_exif.yaml --img-size 640 --save-json --task test 
python postProcessing.py --json_file ./runs/test/base/best_predictions.json
python test.py --weight ./runs/train/yolov7-exif-detection3/weights/best.pt --exist-ok --device 0 --name detection --data data/vaipe_exif.yaml --img-size 640 --save-json --task test 
python postProcessing.py --json_file ./runs/test/detection/best_predictions.json
python ensemble.py --base_model ./runs/test/base/results.csv --adv_model ./runs/test/detection/results.csv --data_path /data/pill/competition/dataset/public_test/pill/images 