MODEL_DIRECTORY=/home/gutturale/code/data/esp_train_results/2020-10/10-13-21-21-16_carla_town01_A1_T20_test_precog.bijection.social_convrnn.SocialConvRNN_
python $PRECOGROOT/precog/esp_infer.py \
    dataset=carla_town01_A5_T20_test \
    model.directory=$MODEL_DIRECTORY \
    # bijection.params.A=5 \
