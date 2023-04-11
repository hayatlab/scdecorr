#!/usr/local_rwth/bin/zsh
module load cuda

source ~/.zshrc

pwd; hostname; date
  

  
# Define a timestamp function
timestamp() {
  date +"%T" # current time
}


timestamp # print another timestamp


conda activate tf_gpu

# Model name: densenet11, densenet21, densenet29 and densenet63
arch=densenet21
#Number of input features
in_features=2000

#We recommend the user to normalize the feature before training the model
#data=/home/noco0013/projects/rito-single-cell-integration-contrastive/Miscell/data/Broad_heart+HCA_heart/lv/merged_adata.h5ad
#data=/home/noco0013/projects/contrastive_learning/data/kidney/X_all_adata.npy
#data=/home/noco0013/projects/contrastive_learning/data/kidney/M.vst_common.401.subset.batch.zinb.h5ad
checkpoint=/home/noco0013/projects/rito-single-cell-integration-contrastive/SimCLR/checkpoint_densenet21_kidney_st_separate_standardization_two_loaders_mmd_dsbn_minibatch=2048_optim=adam/
load_checkpoint=/home/noco0013/projects/rito-single-cell-integration-contrastive/SimCLR/checkpoint_densenet21_kidney_st_separate_standardization_two_loaders_mmd_dsbn_minibatch=2048_optim=adam/epoch_155/checkpoint.pth
batch_obs1=KPBU
batch_obs2=CD
batch_obs_name=batch_prefix_binary
data_obsm_name=X_norm

cl=simclr
optim=adam

#python ../train.py -a $arch -cl $cl -optim $optim --batch-size 2048 --epochs 300 --use_mmd --use_dsbn --in_features $in_features --checkpoint-dir $checkpoint --batch_obs1 $batch_obs1 --batch_obs2 $batch_obs2 --batch_obs_name $batch_obs_name --data_obsm_name $data_obsm_name --model_name simclr_densenet21_0.4 $data

python ../train.py -a $arch -cl $cl -optim $optim --batch-size 2048 --epochs 300 --use_mmd --in_features $in_features --load_checkpoint $load_checkpoint --checkpoint-dir $checkpoint --batch_obs1 $batch_obs1 --batch_obs2 $batch_obs2 --batch_obs_name $batch_obs_name --data_obsm_name $data_obsm_name --model_name simclr_densenet21_0.3 $data

