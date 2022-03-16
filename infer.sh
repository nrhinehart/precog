## testing model on generated dataset
# MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2021-02/02-01-22-35-29_split_dataset_precog.bijection.social_convrnn.SocialConvRNN_
# MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2021-02/02-02-17-47-55_split_dataset_precog.bijection.social_convrnn.SocialConvRNN_
# MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2021-02/test_with_w200
# MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2021-02/test_with_no_lidar
# MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2021-02/carla_Town03_B5_A3_T20_W200

## testing Town10HD single T-intersection
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-02/02-22-14-39-49_carla_Town10_B10_A5_T20_no_yaw
# DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210217/30vehicles
# SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210217/30vehicles/12_val0_test1.json

## testing Town10HD single T-intersection
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-02/02-23-21-55-35_carla_Town10_B10_A5_T20_no_yaw_small
# DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210223/30vehicles_small
# SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210223/30vehicles_small/12_val0_test1.json

## testing Town10HD single T-intersection
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-02/02-24-10-59-31_carla_Town10_B10_A5_T20_no_yaw
# DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210223/30vehicles
# SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210223/30vehicles/12_val0_test1.json

## eval different configurations on Town03
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-01-11-20-57_carla_Town03_B10_A5_T20_no_yaw
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-03-10-33-48_carla_Town03_B10_A5_T20_10Hz_no_yaw
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-03-14-21-03_carla_Town03_B10_A5_T20_10Hz_has_yaw
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-03-18-02-39_carla_Town03_B10_A5_T20_10Hz_no_lidar
# DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210403
# SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210403/Town03/12_val0_test1.json

## eval over all maps
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-06-18-22-30_carla_Town04_B10_A5_T20_10Hz_no_yaw
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-06-23-54-31_carla_Town05_B10_A5_T20_10Hz_no_yaw
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-07-10-55-17_carla_Town06_B10_A5_T20_10Hz_no_yaw
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-07-15-42-37_carla_Town07_B10_A5_T20_10Hz_no_yaw
# MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-02-21-47-45_carla_Town10HD_B10_A5_T20_10Hz_no_yaw
# DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210402
# SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210402/Town10HD/12_val0_test1.json

MODEL_DIRECTORY=/home/fireofearth/data/esp_transfer_results/2021-04/04-13-15-30-19_carla_Town10HD_B10_A5_T20_10Hz_no_yaw

DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210403
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210403/Town03/12_val0_test1.json

DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210218/30vehicles
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210218/selected_test.json
RUN_NAME=carla_Town10_B10_A5_T20_10Hz_selected

export CUDA_VISIBLE_DEVICES=-1; python $PRECOGROOT/precog/esp_infer.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    +dataset.params.test_cap=20 \
    model.directory=$MODEL_DIRECTORY \
    main.compute_metrics=true \
    main.plot=true \
    main.label=$RUN_NAME \
