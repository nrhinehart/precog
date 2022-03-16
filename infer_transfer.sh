#!/bin/bash

MODEL_DIRECTORY=/home/fireofearth/data/esp_transfer_results/2021-04/04-12-19-53-24_carla_Town03_B10_A5_T20_10Hz_no_yaw
DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210403
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210403/Town03/12_val0_test1.json
RUN_NAME=carla_Town03_B10_A5_T20_10Hz_modelTransfer

export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_infer.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    model.directory=$MODEL_DIRECTORY \
    main.label=$RUN_NAME \
    main.compute_metrics=true \
    main.plot=true \

sleep 3

MODEL_DIRECTORY=/home/fireofearth/data/esp_transfer_results/2021-04/04-12-23-29-40_carla_Town04_B10_A5_T20_10Hz_no_yaw
DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210402
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210402/n_present5/Town04/12_val0_test1.json
RUN_NAME=carla_Town04_B10_A5_T20_10Hz_modelTransfer

export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_infer.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    model.directory=$MODEL_DIRECTORY \
    main.label=$RUN_NAME \
    main.compute_metrics=true \
    main.plot=true \

sleep 3

MODEL_DIRECTORY=/home/fireofearth/data/esp_transfer_results/2021-04/04-13-04-15-38_carla_Town05_B10_A5_T20_10Hz_no_yaw
DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210402
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210402/n_present5/Town05/12_val0_test1.json
RUN_NAME=carla_Town05_B10_A5_T20_10Hz_modelTransfer

export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_infer.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    model.directory=$MODEL_DIRECTORY \
    main.label=$RUN_NAME \
    main.compute_metrics=true \
    main.plot=true \

sleep 3

MODEL_DIRECTORY=/home/fireofearth/data/esp_transfer_results/2021-04/04-13-07-03-09_carla_Town06_B10_A5_T20_10Hz_no_yaw
DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210402
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210402/n_present5/Town06/12_val0_test1.json
RUN_NAME=carla_Town06_B10_A5_T20_10Hz_modelTransfer

export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_infer.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    model.directory=$MODEL_DIRECTORY \
    main.label=$RUN_NAME \
    main.compute_metrics=true \
    main.plot=true \

sleep 3

MODEL_DIRECTORY=/home/fireofearth/data/esp_transfer_results/2021-04/04-13-11-24-29_carla_Town07_B10_A5_T20_10Hz_no_yaw
DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210402
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210402/n_present5/Town07/12_val0_test1.json
RUN_NAME=carla_Town07_B10_A5_T20_10Hz_modelTransfer

export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_infer.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    model.directory=$MODEL_DIRECTORY \
    main.label=$RUN_NAME \
    main.compute_metrics=true \
    main.plot=true \

sleep 3

MODEL_DIRECTORY=/home/fireofearth/data/esp_transfer_results/2021-04/04-13-15-30-19_carla_Town10HD_B10_A5_T20_10Hz_no_yaw
DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210402
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210402/Town10HD/12_val0_test1.json
RUN_NAME=carla_Town10HD_B10_A5_T20_10Hz_modelTransfer

export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_infer.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    model.directory=$MODEL_DIRECTORY \
    main.label=$RUN_NAME \
    main.compute_metrics=true \
    main.plot=true \
