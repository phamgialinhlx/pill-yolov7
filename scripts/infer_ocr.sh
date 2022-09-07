docker run --rm --gpus all \
-v $PWD/vaipe_exif:/app/vaipe_exif \
-v /home/pill/competition/yolov7/vaipe_exif/public_test_new/prescription/image:/home/pill/competition/yolov7/vaipe_exif/public_test_new/prescription/image \
-v $PWD/runs:/app/runs \
-v $PWD/inference:/app/inference \
--shm-size 8G \
ai4vn:latest \
inference/infer_ocr.py 