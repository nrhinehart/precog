#!/bin/bash

# Job name:
#SBATCH --job-name=precog

# Account:
# QoS: must be savio_long for jobs > 3 days!
# Partition (NEED TO SET TO A PARTION WITH GPUS, SEE TABLE HERE: https://research-it.berkeley.edu/services/high-performance-computing/user-guide/savio-user-guide)
# Wall clock limit (7 days in this case):

#SBATCH --account=co_rail
#SBATCH --qos=rail_2080ti3_normal
#SBATCH --partition=savio3_2080ti
#SBATCH --time=7-00:00:00

## SBATCH --account=fc_rail
## SBATCH --qos=savio_normal
## SBATCH --partition=savio2_1080ti
## SBATCH --time=3-00:00:00

# Here’s how to request two CPUs for each GPU: the total of CPUs requested results from multiplying two settings: the number of tasks (“--ntasks=”) and CPUs per task ("--cpus-per-task=").
# Processors per task (needs to be 2x the number of GPUS)
#SBATCH --cpus-per-task=2
#SBATCH --output=array_job_%A_task_%a.out
#SBATCH --error=array_job_%A_task_%a.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=nrhinehart@berkeley.edu

### 8-sized sweep
## SBATCH --array=1-8
## Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
## SBATCH --gres=gpu:2

### 2-sized sweep
#SBATCH --array=1-2
# Number of GPUs, this can be in the format of "gpu:[1-4]", or "gpu:K80:[1-4] with the type included
#SBATCH --gres=gpu:1

echo "I am task $SLURM_ARRAY_TASK_ID"
conda init bash
conda activate pre3
# Hardcoded precog root
source ~/dev/precog/precog_env.sh
# SHARED=("${PRECOGROOT}/precog/esp_train.py" dataset=carla_town01_A1_T20_v2 dataset.params.root_path=/global/scratch/nrhinehart/data/dim_release_results/2020-03 main.eager=False bijection.params.A=1 hardware=savio_cluster);

SHARED=("${PRECOGROOT}/precog/esp_train.py" dataset=carla_town01_A1_T20_v2 dataset.params.root_path=/global/scratch/nrhinehart/data/dim_release_results/2020-03 main.eager=False bijection.params.A=1 hardware=savio_cluster);

# We require cuda 10.0
module load cuda/10.0
module load cudnn/7.5

case $SLURM_ARRAY_TASK_ID in
    1) python ${SHARED[@]} main.tag="slurm_${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}_whiskerfix_lightfix_F8" bijection.params.cnnconf.F=8;;
    2) python ${SHARED[@]} main.tag="slurm_${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}_whiskerfix_lightfix_F16" bijection.params.cnnconf.F=16;;
esac
