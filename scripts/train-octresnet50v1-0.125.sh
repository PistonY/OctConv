#!/usr/bin/env bash
python ../train_script.py \
    --data-dir /system1/Dataset/rec --batch-size 128 --alpha 0.125 --model oct_resnet50v1 \
    --dtype float16 --num-gpus 4 -j 12 --num-epochs 100 --lr 0.4 --warmup-epochs 5 --model hybrid \
    --no-wd --log-interval 800