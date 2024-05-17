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
in_features=2000
#We recommend the user to normalize the feature before training the model
data=/home/noco0013/projects/rito-single-cell-integration-contrastive/Miscell/data/Broad_heart+HCA_heart/lv/merged_adata.h5ad
#data=/home/noco0013/projects/contrastive_learning/data/kidney/X_all_adata.npy

batch_obs1=HCA
batch_obs2=BR

#load_checkpoint=/home/noco0013/projects/rito-single-cell-integration-contrastive/barlowtwins/checkpoint_densenet21_broad+hca_two_loaders_mmd_alpha=1_beta=1_minibatch=2048/epoch_936/checkpoint.pth

python ../simclr_train.py -a $arch --batch-size 512 --epochs 300 --in_features $in_features --batch_obs1 $batch_obs1 --batch_obs2 $batch_obs2 $data

#python ../train.py -a $arch -cl $cl -optim $optim --batch-size 2048 --epochs 1000 --in_features $in_features --load_checkpoint $load_checkpoint --checkpoint-dir $checkpoint --batch_obs1 $batch_obs1 --batch_obs2 $batch_obs2 --model_name barlow_twins_densenet21_1.2 $data

