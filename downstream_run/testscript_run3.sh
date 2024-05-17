#!/usr/local_rwth/bin/zsh

module load cuda

source ~/.zshrc

pwd; hostname; date
  
conda activate scib_new

  
# Define a timestamp function
timestamp() {
  date +"%T" # current time
}

# do something..
timestamp # print timestamp
# do something else...
timestamp # print another timestamp


python run3.py