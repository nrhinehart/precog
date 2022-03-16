#!/bin/bash

export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_train.py \
    dataset=main \
    optimizer.params.plot_before_train=True \
    optimizer.params.save_before_train=True \
    optimizer.params.epochs=10 \
