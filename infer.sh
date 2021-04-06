#MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-11/carla_town01_B10_A5_T20_precog_SocialConvRNN
#MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-11/carla_town01_B10_A5_T20_test_gen_dataset

## testing continue train on small dataset
#MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-12/12-13-19-56-19_carla_town01_A5_T20_test_precog.bijection.social_convrnn.SocialConvRNN_
#MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-12/12-13-21-27-10_carla_town01_A5_T20_test_precog.bijection.social_convrnn.SocialConvRNN_

## testing continue train on original dataset
## original (before continue train)
# MODEL_DIRECTORY=/media/external/data/esp_train_results/2020-12/carla_town01_B10_A5_T20_precog_SocialConvRNN
# MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-12/update2.carla_town01_B10_A5_T20_precog_SocialConvRNN
## after continue train

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

MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-03/03-31-10-57-35_carla_Town03_B10_A5_T20_has_yaw
MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-03/03-31-16-39-27_carla_Town03_B10_A5_T20_no_yaw
MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-02-15-03-51_carla_Town04_B10_A5_T20_10Hz_no_yaw

MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-03-22-57-54_carla_Town03_B10_A5_T20_10Hz_symm
MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-05-10-56-51_carla_Town10HD_B10_A5_T20_10Hz_symm
MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-05-17-43-16_carla_Town04_B10_A5_T20_10Hz_symm
MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-05-20-31-35_carla_Town05_B10_A5_T20_10Hz_symm
MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-03-10-33-48_carla_Town03_B10_A5_T20_10Hz_no_yaw

DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210402
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210402/n_present5/Town05/12_val0_test1.json

DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210403
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210403/Town03/12_val0_test1.json

python $PRECOGROOT/precog/esp_infer.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    model.directory=$MODEL_DIRECTORY \
    main.compute_metrics=true \
    main.plot=true \
