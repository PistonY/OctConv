#!/usr/bin/env bash
python  ../train_script.py \
    --data-dir /home/ubuntu/piston/imagenet_rec --batch-size 256 --alpha 0 \
    --model mobilenetv3_large --dtype float32 --num-gpus 1 -j 32 --num-epochs 100 \
    --lr 0.256 --warmup-epochs 5 --mode hybrid --lr-mode cosine --log-interval 200

