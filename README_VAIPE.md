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
### Add wandb api key
In [Dockerfile](Dockerfile), provide your wandb API key in line 20
### Build docker image
```
docker build . -t ai4vn:latest
```
### Run 
1. Preprocess
    - Change volume dataset (line 3) to your local directory train and test (infer) dataset.
    - Then run
    ``` 
    bash scripts/preprocess.sh 
    ```
2.  Process OCR
    - Change volume dataset (line 3) to your local directory inference test image.
    - Then run
    ```
    bash scripts/infer_ocr.sh
    ```
3. Train and Inference
    - Train
        - Change file [config.json](inference/config.json) content with each config in folder [cfg/docker](cfg/docker)
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

