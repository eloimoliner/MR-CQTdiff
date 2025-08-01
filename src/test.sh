#!/bin/bash

module load mamba
source activate /scratch/work/molinee2/conda_envs/env2024  


conf=conf_OpenSinger.yaml

tester=unconditional

n="test"

PATH_EXPERIMENT=experiments/$n


#choose model and dataset

#network=MR-cqtdiff+_44k
#ckpt="../checkpoints/OpenSinger_MRcqtdiff_500k.pt"
#ckpt="../checkpoints/FMA_MRcqtdiff_500k.pt"

network=ncsnpp_44k
ckpt="../checkpoints/OpenSinger_MRcqtdiff_500k.pt"
#ckpt="../checkpoints/FMA_MRcqtdiff_500k.pt"

#network=cqtdiff+_44k
#ckpt="../checkpoints/OpenSinger_MRcqtdiff_500k.pt"
#ckpt="../checkpoints/FMA_MRcqtdiff_500k.pt"


mkdir -p $PATH_EXPERIMENT

python test_unconditional.py --config-name=$conf \
  model_dir=$PATH_EXPERIMENT \
  tester=$tester \
  tester.checkpoint=$ckpt \
  tester.overriden_name="2D_OpenSinger_HD_CQT_waveform_IS2_100k_v2" \
  tester.unconditional.num_samples=4 \
