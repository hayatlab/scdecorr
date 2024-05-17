#!/usr/local_rwth/bin/zsh

source ~/.zshrc

pwd; hostname; date



# Define a timestamp function
timestamp() {
  date +"%T" # current time
}


conda activate scrna_models
echo "Activated scrna_models env"


timestamp # print timestamp

echo "Executing run_compute_features_2.py"

python run_compute_features_2.py
echo "Execution run_compute_features_2 finished"

timestamp # print timestamp

