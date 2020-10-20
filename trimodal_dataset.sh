python $PRECOGROOT/precog/esp_train.py \
    bijection=social_convrnn \
    dataset=trimodal_dataset \
    bijection.params.A=1 \
    dataset.params.B=20 \
    main.eager=false