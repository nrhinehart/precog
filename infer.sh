#MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-11/carla_town01_B10_A5_T20_precog_SocialConvRNN
#MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-11/carla_town01_B10_A5_T20_test_gen_dataset
## testing continue train
#MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-12/12-13-19-56-19_carla_town01_A5_T20_test_precog.bijection.social_convrnn.SocialConvRNN_
#MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-12/12-13-21-27-10_carla_town01_A5_T20_test_precog.bijection.social_convrnn.SocialConvRNN_
## testing continue train on full network
MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-12/carla_town01_B10_A5_T20_precog_SocialConvRNN
MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-12/12-15-16-27-42_carla_town01_A5_T20_test_precog.bijection.social_convrnn.SocialConvRNN_
python $PRECOGROOT/precog/esp_infer.py \
    dataset=main \
    model.directory=$MODEL_DIRECTORY \
    main.compute_metrics=true \
