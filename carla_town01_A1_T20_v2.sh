python $PRECOGROOT/precog/esp_train.py \
    dataset=carla_town01_A1_T20_v2 \
    main.eager=False \
    bijection.params.A=1 \
    optimizer.params.plot_before_train=True \
    optimizer.params.save_before_train=True \
