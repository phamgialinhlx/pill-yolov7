## Training

``` bash
# train p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/vaipe.yaml --img 640 640 --cfg cfg/training/yolov7-vaipe.yaml --weights '' --name yolov7-vaipe --hyp data/hyp.scratch.p5.yaml

# train p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/vaipe.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-vaipe.yaml --weights '' --name yolov7-w6-vaipe --hyp data/hyp.scratch.p6.yaml
```


## Transfer learning

``` bash
# finetune p5 models
python train.py --workers 8 --device 0 --batch-size 32 --data data/vaipe.yaml --img 640 640 --cfg cfg/training/yolov7-vaipe.yaml --weights 'yolov7_training.pt' --name yolov7-vaipe --hyp data/hyp.scratch.custom.yaml

# finetune p6 models
python train_aux.py --workers 8 --device 0 --batch-size 16 --data data/vaipe.yaml --img 1280 1280 --cfg cfg/training/yolov7-w6-vaipe.yaml --weights 'yolov7-w6_training.pt' --name yolov7-w6-vaipe --hyp data/hyp.scratch.custom.yaml
```