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
arch=densenet11
#Number of input features
in_features=2000
#We recommend the user to normalize the feature before training the model
data=/home/noco0013/projects/rito-single-cell-integration-contrastive/Miscell/data/human_pancreas_norm_complexBatch_original_adata_X_unscaled.h5ad
checkpoint=/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet11_pancreas_unscaled_multi_dom_train_dsbn_minibatch=512_projector_512-512-512_optim=adam/
#load_checkpoint=/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet21_pancreas_multi_dom_train_dsbn_minibatch=512_optim=adam/epoch_988/checkpoint.pth

n_domains=9
batch_obs_name=tech_int
projector="512-512-512"
cl=btwins
optim=adam

python ../multi_domain_train.py -a $arch -cl $cl -optim $optim --batch-size 512 --epochs 500 --use_dsbn --projector $projector --in_features $in_features --checkpoint-dir $checkpoint --n_domains $n_domains --batch_obs_name $batch_obs_name --model_name barlow_twins_densenet21_1.2 $data

#python ../train.py -a $arch -cl $cl -optim $optim --batch-size 2048 --epochs 726 --projector $projector --in_features $in_features --load_checkpoint $load_checkpoint --checkpoint-dir $checkpoint --batch_obs1 $batch_obs1 --batch_obs2 $batch_obs2 --batch_obs_name $batch_obs_name --data_obsm_name $data_obsm_name --model_name barlow_twins_densenet21_1.2 $data
