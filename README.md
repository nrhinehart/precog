[![License: CC BY-NC-ND 4.0](https://img.shields.io/badge/License-CC%20BY--NC--ND%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

# Purposes

1. Serve as a partial reimplementation of the ESP multi-agent forecasting model
2. Train a Deep Imitative Model for CARLA autonomous driving task. See [https://github.com/nrhine1/deep_imitative_models](https://github.com/nrhine1/deep_imitative_models).

Paper:
[http://openaccess.thecvf.com/content_ICCV_2019/papers/Rhinehart_PRECOG_PREdiction_Conditioned_on_Goals_in_Visual_Multi-Agent_Settings_ICCV_2019_paper.pdf](http://openaccess.thecvf.com/content_ICCV_2019/papers/Rhinehart_PRECOG_PREdiction_Conditioned_on_Goals_in_Visual_Multi-Agent_Settings_ICCV_2019_paper.pdf)

# Primary files
`precog/esp_train.py` Interface to train a model
`precog/esp_infer.py` Interface to perform test-time inference (plotting and metrics computation)

# Setup

```bash
export PRECOGCONDAENV=pre3
conda create -n $PRECOGCONDAENV python=3.6.6
conda activate $PRECOGCONDAENV
source precog_env.sh
pip install -r requirements.txt
```

Potentially install the `nrhine1/deep_imitative_models` repo.

# Training ESP on a toy context-free single-agent dataset.
```bash
export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_train.py \
bijection=social_convrnn \
dataset=trimodal_dataset \
bijection.params.A=1 \
dataset.params.B=20 \
main.eager=false
```

# Training ESP on data collected with `carla_agent.py` with the deep_imitative_models repo

```bash
export CUDA_VISIBLE_DEVICES=0; python $PRECOGROOT/precog/esp_train.py \
dataset=carla_town01_A1_T20_v2 \
main.eager=False \
bijection.params.A=1 \
optimizer.params.plot_before_train=True \
optimizer.params.save_before_train=True
```

# Preparing NuScenes data.
Download the nuscenes dataset, then use the script `preprocess_nuscenes.py`

# Tips
Recall that the log-likelihood is insensitive to sample quality. If you're not using a sample-penalizing metric, it will take longer training time to observe higher-quality samples.

# Citation
```
@InProceedings{Rhinehart_2019_ICCV,
author = {Rhinehart, Nicholas and McAllister, Rowan and Kitani, Kris and Levine, Sergey},
title = {PRECOG: PREdiction Conditioned on Goals in Visual Multi-Agent Settings},
booktitle = {The IEEE International Conference on Computer Vision (ICCV)},
month = {October},
year = {2019}
}
```
