#!/usr/local_rwth/bin/zsh


source ~/.zshrc

pwd; hostname; date



# Define a timestamp function
timestamp() {
  date +"%T" # current time
}


conda activate LIGER
echo "Activated LIGER env"


timestamp # print timestamp

echo "Executing run_compute_features_1.py"

python run_compute_features_1.py
echo "Execution run_compute_features_1.py finished"

timestamp # print timestamp

