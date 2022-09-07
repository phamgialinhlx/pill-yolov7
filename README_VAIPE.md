# Pill Gates - VAIPE: Medicine Pill Image Recognition Challenge
## Config data path

### 1. Create data_config.json file in [inference](inference) folder.
File [data_config.json](inference/data_config.json) contains path to train label and image directories of prescription and pill as well as inference (test) image directories of prescription and pill. Example [data_config.json](inference/data_config.example.json) file.

### 2. About [config.json](inference/config.json) file
There are several configs for trainning. The models trained on these configs are then ensembled in test phase. To train each config, put the content in file [config.json](inference/config.json) 

## Pipeline

1. Preprocess data [preprocess.py](inference/preprocess.py) (Run once)
```
python inference/preprocess.py 
```
2. Process OCR [infer_ocr.py](inference/infer_ocr.py) (Run once)
```
python inference/infer_ocr.py 
```
3. Train and Inference (Test):
    - Train model with each config: Change config and run [train_vaipe.py](inference/train_vaipe.py)
    ```
    python inference/train_vaipe.py
    ```
    - Inference:  run [infer.py](inference/infer.py) 
    ```
    python inference/infer.py
    ```

## Docker

### Build docker image
```
docker build . -t ai4vn:latest
```
### Run 
1. Preprocess
``` 
bash scripts/preprocess.sh 
```
2.  Process OCR
```
bash scripts/infer_ocr.sh
```
3. Train and Inference
    - Train
        - Change config
        - Then run 
        ```
        bash scripts/train.sh
        ```
    - Inference
        ```
        bash scripts/infer.sh
        ```
## Results
- Model weight stored in [runs/train](runs/train)
- File [results.csv](runs/ensemble/results.csv)

