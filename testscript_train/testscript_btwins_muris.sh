#!/usr/local_rwth/bin/zsh

module load cuda

source ~/.zshrc

pwd; hostname; date
  

  
# Define a timestamp function
timestamp() {
  date +"%T" # current time
}

# do something..
timestamp # print timestamp
# do something else...
timestamp # print another timestamp


conda activate tf_gpu
# Model name: densenet11, densenet21, densenet29 and densenet63
arch=densenet21
#Number of input features
in_features=5000
#We recommend the user to normalize the feature before training the model
data=/home/noco0013/projects/rito-single-cell-integration-contrastive/Miscell/data/data/Muris/muris_sample_filter.h5ad
checkpoint=/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet21_muris_hvg_5000_unscaled_multi_dom_train_dsbn_minibatch=2048_optim=adam/
load_checkpoint=/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet21_muris_hvg_5000_unscaled_multi_dom_train_dsbn_minibatch=2048_optim=adam/epoch_539/checkpoint.pth

n_domains=2
batch_obs_name=batch_int
projector="1024-1024-1024"
cl=btwins
optim=adam
#data_obsm_name=X_normalized_log_counts_unscaled

python ../multi_domain_train.py -a $arch -cl $cl -optim $optim --batch-size 2048 --epochs 1000 --use_dsbn --projector $projector --in_features $in_features --load_checkpoint $load_checkpoint --checkpoint-dir $checkpoint --n_domains $n_domains --batch_obs_name $batch_obs_name --model_name barlow_twins_densenet21_1.2 $data

#python ../train.py -a $arch -cl $cl -optim $optim --batch-size 2048 --epochs 726 --projector $projector --in_features $in_features --load_checkpoint $load_checkpoint --checkpoint-dir $checkpoint --batch_obs1 $batch_obs1 --batch_obs2 $batch_obs2 --batch_obs_name $batch_obs_name --data_obsm_name $data_obsm_name --model_name barlow_twins_densenet21_1.2 $data
