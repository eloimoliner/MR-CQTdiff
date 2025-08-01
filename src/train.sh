#!/bin/bash

export HYDRA_FULL_ERROR=1 

module load mamba
source activate /scratch/work/molinee2/conda_envs/env2024

# main config
conf=conf_FMA.yaml

n="train_example"

PATH_EXPERIMENT=experiments/$n
mkdir -p $PATH_EXPERIMENT

#network="MR-cqtdiff_44k"
network="ncsnpp_44k"
#network="cqtdiff+_44k"

python train.py --config-name=$conf \
  network=$network \

