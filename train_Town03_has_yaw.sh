#!/bin/bash

DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210403
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210403/Town03/12_val0_test1.json
A=5
MAP=Town03
NAME=carla_${MAP}_B10_A${A}_T20_10Hz_has_yaw
# don't forget to enable yaw in dataset
# and then disable afterwards

    # proxy=binary_mask_proxy \
    # objective=symmetric_cross_entropy \
export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_train.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    dataset.params.name=$NAME \
    bijection.params.A=$A \
    dataset.params.A=$A \
    dataset.params.B=10 \
    optimizer.params.plot_before_train=False \
    optimizer.params.save_before_train=True \
    optimizer.params.epochs=4 \
    optimizer.params.evaluate_period=500 \
