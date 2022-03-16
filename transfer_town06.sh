
MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-07-21-06-48_carla_Town01_B10_A5_T20
DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210402
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210402/n_present5/Town06/12_val0_test1.json
A=5
MAP=Town06
NAME=carla_${MAP}_B10_A${A}_T20_10Hz_no_yaw

export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_transfer.py \
    model.directory=$MODEL_DIRECTORY \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    dataset.params.name=$NAME \
    optimizer.params.epochs=4 \
    optimizer.params.evaluate_period=500 \
