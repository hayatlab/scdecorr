#!/bin/bash

ulimit -a
free -h

pwd; hostname; date
  
# Define a timestamp function
timestamp() {
  date +"%T" # current time
}

timestamp # print timestamp


# Model name: densenet11, densenet21, densenet29 and densenet63
arch=densenet11
#Number of input features
in_features=2000

#We recommend the user to normalize the feature before training the model
data=/data/human_lungs/adata.h5ad
# Path where model is saved after training
checkpoint=/checkpoints/human_lungs/v1/
cfg_path='/cfg/human_lungs/model_config.yaml'
# ckpt path if training has to be resumed from earlier ckpt
load_checkpoint=''


n_domains=16
batch_obs_name=batch
projector="512-512-512"
cl=btwins
optim=adam
batch_size=512
epochs=1000



python ../scdecorr/multi_domain_single_gpu_train.py \
   -a $arch -cl $cl -optim $optim --batch-size $batch_size --epochs $epochs \
   --use_dsbn --projector $projector --in_features $in_features --cfg_path $cfg_path \
   --checkpoint-dir $checkpoint --n_domains $n_domains --batch_obs_name $batch_obs_name --model_name scdecorr_v1 $data