#!/usr/local_rwth/bin/zsh

module load cuda

source ~/.zshrc

pwd; hostname; date

conda activate scib_new
echo "Activated scib_new env"

# Define a timestamp function
timestamp() {
  date +"%T" # current time
}

timestamp # print timestamp

echo "Executing run_compute_features.py"

python run_compute_features.py

echo "Execution run_compute_features.py finished"
timestamp # print timestamp
