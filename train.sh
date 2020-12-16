
export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_train.py \
    dataset=carla_town01_A5_T20_test \
    bijection.params.A=5 \
    optimizer.params.plot_before_train=False \
    optimizer.params.save_before_train=True \
    optimizer.params.epochs=10 \
    optimizer.params.evaluate_period=2000 \
