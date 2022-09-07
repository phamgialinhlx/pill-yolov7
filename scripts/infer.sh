docker run --rm --gpus all \
-v $PWD/vaipe_exif:/app/vaipe_exif \
-v $PWD/runs:/app/runs \
-v $PWD/inference:/app/inference \
--shm-size 8G \
ai4vn:latest \
inference/infer.py 