
export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_train.py \
    dataset=main \
    bijection.params.A=5 \
    optimizer.params.plot_before_train=False \
    optimizer.params.save_before_train=True \
    optimizer.params.epochs=10 \
    optimizer.params.evaluate_period=500 \
