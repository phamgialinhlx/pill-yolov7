#!/bin/bash

for i in "$@"; do
  case $i in
    --weight=*)
      WEIGHT="${i#*=}"
      shift # past argument=value
      ;;
    --device=*)
      DEVICE="${i#*=}"
      shift # past argument=value
      ;;
    --name=*)
      NAME="${i#*=}"
      shift # past argument=value
      ;;
    --data=*)
      DATA="${i#*=}"
      shift # past argument=value
      ;;
    --img-size=*)
      IMG_SIZE="${i#*=}"
      shift # past argument=value
      ;;
    --data-path=*)
      DATA_PATH="${i#*=}"
      shift # past argument=value
      ;;
    -*|--*)
      echo "Unknown option $i"
      exit 1
      ;;
    *)
      ;;
  esac
done

echo "WEIGHT  = ${WEIGHT}"

if [[ -n $1 ]]; then
    echo "Last line of file specified as non-opt/last argument:"
    tail -1 $1
fi

python test.py \
    --weight ${WEIGHT} \
    --exist-ok \
    --device ${DEVICE} \
    --name base \
    --data ${DATA} \
    --img-size 640 \
    --save-json \
    --task test

# python postProcessing.py --json_file ./runs/test/base/best_predictions.json
# python test.py --weight ./runs/train/yolov7-exif-detection3/weights/best.pt --exist-ok --device 0 --name detection --data data/vaipe_exif.yaml --img-size 640 --save-json --task test 
# python postProcessing.py --json_file ./runs/test/detection/best_predictions.json
# python ensemble.py --base_model ./runs/test/base/results.csv --adv_model ./runs/test/detection/results.csv --data_path /data/pill/competition/dataset/public_test/pill/images 