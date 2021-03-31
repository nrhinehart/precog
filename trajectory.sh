
MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-02/02-22-14-39-49_carla_Town10_B10_A5_T20_no_yaw
DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210218/30vehicles
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210218/selected_test.json

MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-02/02-24-10-59-31_carla_Town10_B10_A5_T20_no_yaw
DATA_PATH=/home/fireofearth/data/precog_generate/datasets/20210223/1scenario
SPLIT_PATH=/home/fireofearth/data/precog_generate/splits/20210223/1scenario/1scenario.json

python $PRECOGROOT/precog/esp_infer_trajectory.py \
    epochs=228 \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    model.directory=$MODEL_DIRECTORY \
