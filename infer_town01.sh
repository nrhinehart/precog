
MODEL_DIRECTORY=/home/fireofearth/data/esp_train_results/2021-04/04-07-21-06-48_carla_Town01_B10_A5_T20

RUN_NAME=carla_Town10_B10_A5_T20

export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_infer.py \
    dataset=main \
    model.directory=$MODEL_DIRECTORY \
    main.compute_metrics=true \
    main.plot=true \
    main.label=$RUN_NAME \
