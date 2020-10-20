python $PRECOGROOT/precog/esp_train.py \
    dataset=carla_town01_A5_T20_test \
    main.eager=False \
    bijection.params.A=5 \
    optimizer.params.plot_before_train=True \
    optimizer.params.save_before_train=True \
    optimizer.params.epochs=10 \
