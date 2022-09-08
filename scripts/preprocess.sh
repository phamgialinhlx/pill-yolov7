docker run --rm --gpus all \
-v $PWD/vaipe_exif:/app/vaipe_exif \
-v /home/pill/competition/dataset:/home/pill/competition/dataset \
-v /home/pill/competition/yolov7/vaipe_exif/public_test_new/pill/image:/home/pill/competition/yolov7/vaipe_exif/public_test_new/pill/image \
-v $PWD/runs:/app/runs \
-v $PWD/inference:/app/inference \
--user $(id -u):$(id -g) \
--shm-size 8G \
ai4vn:latest \
inference/preprocess.py 