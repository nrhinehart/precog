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
MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2021-02/test_with_w200
DATA_PATH=/media/external/data/precog_generate/datasets/20210201
SPLIT_PATH=/media/external/data/precog_generate/splits/20210201/Town03/12_val0_test1.json
python $PRECOGROOT/precog/esp_infer.py \
    dataset=split_dataset \
    dataset.params.data_path=$DATA_PATH \
    dataset.params.split_path=$SPLIT_PATH \
    model.directory=$MODEL_DIRECTORY \
    main.compute_metrics=true \
