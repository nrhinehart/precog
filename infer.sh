MODEL_DIRECTORY=/home/gutturale/code/data/esp_train_results/2020-10/carla_town01_epoch10_B1_A5_T20_precog_SocialConvRNN
python $PRECOGROOT/precog/esp_infer.py \
    dataset=carla_town01_A5_T20_test \
    model.directory=$MODEL_DIRECTORY \
    main.compute_metrics=false \
