
MODEL_DIRECTORY=/home/fireofearth/code/data/esp_train_results/2020-12/carla_town01_B10_A5_T20_precog_SocialConvRNN
export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_continue_train.py \
    model.directory=$MODEL_DIRECTORY \
    dataset=main \
    bijection.params.A=5 \
    optimizer.params.plot_before_train=False \
    optimizer.params.save_before_train=True \
    optimizer.params.epochs=10 \
    optimizer.params.evaluate_period=500 \
