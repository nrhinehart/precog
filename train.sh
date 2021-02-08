#!/bin/bash

DATA_PATH=/media/external/data/precog_generate/datasets/20210201
SPLIT_PATH=/media/external/data/precog_generate/splits/20210201/Town03/12_val0_test1.json
NAME=carla_Town03_B10_A3_T20

export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_train.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    dataset.params.name=$NAME \
    bijection.params.A=3 \
    dataset.params.A=3 \
    optimizer.params.plot_before_train=False \
    optimizer.params.save_before_train=True \
    optimizer.params.epochs=5 \
    optimizer.params.evaluate_period=500 \
